"""Read-only diagnostics of five largest packed-window-control OOF errors.

Selection is diagnostic, never an exclusion/tuning rule. Full raw clips are
processed exactly as the pinned preparation code, then selected windows replayed.
"""
from pathlib import Path
import csv, hashlib, importlib.util, json, struct, sys, warnings, zlib
import numpy as np
import torch
H=Path(__file__).resolve().parent;S=H.parent;ROOT=S.parents[2]
C=S/'cnn-multiwindow695';R=S/'cnn-record-loss692';O=H/'analysis'
sys.path.insert(0,str(C/'bundle'))
from model import GdiCNN,channels
torch.set_num_threads(4)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(zip(('min','p05','median','p95','max'),map(float,np.quantile(a,[0,.05,.5,.95,1])))) if a.size else None

d=arrays(C/'bundle/data.npz');z=arrays(R/'analysis/window_control/unpacked/results/oof_predictions.npz')
old=arrays(C/'analysis/cnn_oof.npz')
for k in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(d[k],z[k]);np.testing.assert_array_equal(d[k],old[k])
err=np.abs(z['predictions'][:,0]-d['y']);chosen=np.argsort(-err.max(axis=1),kind='stable')[:5]
sources={str(p):sha(p) for p in (C/'bundle/data.npz',R/'analysis/window_control/unpacked/results/oof_predictions.npz',C/'analysis/cnn_oof.npz')}
save(H/'protocol.json',dict(selection='Top five records by maximum absolute side OOF error, stable original row tie order',
    model='packed record window_control; not the still-running record_mean model',records=chosen.tolist(),
    cpu_only=True,no_training_or_exclusion=True,no_input_modification=True,protected_final_opened=False,
    replay_tolerance=1e-4,replay_precision='CPU float64 same weights, already established reference',
    diagnostic_reference='Only outer training windows; descriptive overlapping-window quantiles, not person-equal performance'))
prep=module('prepared_input_reader',C/'prepare_inputs.py');ns=prep.source_functions()
archive=ROOT/'outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip'
reader=module('archive_reader',ROOT/'tools/stanford-sample-check/inspect_archive.py')
groups={'openpose/'+str(d['video_id'][i])+'-processed':int(i) for i in chosen};members={g:{} for g in groups}
print('SELECTED',chosen.tolist(), 'archive',str(archive.resolve()),flush=True)
with archive.open('rb',buffering=4*1024*1024) as stream:
    count,_,offset=reader.directory_info(stream);stream.seek(offset)
    for _ in range(count):
        header=stream.read(46);assert header[:4]==b'PK\x01\x02'
        f=struct.unpack('<4s6H3L5H2L',header);name=stream.read(f[10]).decode();stream.seek(f[11]+f[12],1)
        group='/'.join(name.split('/')[:2])
        if group in groups and name.endswith('_keypoints.json'):
            frame=int(name.rsplit('_',2)[-2])
            if frame<=1499:
                assert not f[3]&1 and f[4] in (0,8)
                members[group][frame]=dict(name=name,offset=f[16],size=f[9],compressed=f[8],crc=f[7],method=f[4])
print('RAW_INDEX_READY',flush=True)
left=[28,29,26,27,24,25,38,39,50,52,54,56];right=[22,23,20,21,18,19,44,45,51,53,55,57]
feature_names=['ankle_x','ankle_y','knee_x','knee_y','hip_x','hip_y','toe_x','toe_y','knee_angle','ankle_angle','toe_ankle_distance','signed_ankle_gap']
models={};bounds={};cases=[];rows=[];maxreplay=0
for rank,i in enumerate(chosen,1):
    i=int(i);fold=int(d['fold'][i]);wi=np.flatnonzero(d['window_record']==i);x=d['x'][wi];starts=d['window_start'][wi]
    orig=int(d['original_row_index'][i]);p=C/'records'/('%04d.npz'%orig);proof=read(p.with_suffix('.json'))
    assert sha(p)==proof['artifact_sha256'];sources[str(p)]=sha(p);sources[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
    cache=arrays(p);np.testing.assert_array_equal(cache['multi_x'],x);np.testing.assert_array_equal(cache['window_start'],starts)
    group='openpose/'+str(d['video_id'][i])+'-processed';ms=members[group];last=0
    while last+1 in ms and last<1499:last+=1
    raw=np.full((last,75),np.nan);digest=hashlib.sha256();ambiguous=[]
    with archive.open('rb') as stream:
        for frame in range(1,last+1):
            m=ms[frame];stream.seek(m['offset']);header=stream.read(30);assert header[:4]==b'PK\x03\x04'
            nl,el=struct.unpack_from('<HH',header,26);assert stream.read(nl).decode()==m['name'];stream.seek(el,1)
            packed=stream.read(m['compressed']);content=zlib.decompress(packed,-15) if m['method']==8 else packed
            assert len(content)==m['size'] and zlib.crc32(content)&0xffffffff==m['crc'];digest.update(content)
            candidates=[]
            for person in json.loads(content)['people']:
                a=np.array(person['pose_keypoints_2d'],dtype=np.float64)
                if a.shape==(75,) and np.isfinite(a).all() and a[::3].max()<320:candidates.append(a)
            if len(candidates)==1:raw[frame-1]=candidates[0]
            elif len(candidates)>1:ambiguous.append(frame)
    raw=raw[:np.flatnonzero(np.isfinite(raw).any(axis=1))[-1]+1]
    assert digest.hexdigest()==proof['raw_content_sha256'] and len(raw)==proof['raw_frames_used']
    assert ambiguous==proof['ambiguous_frames'];np.testing.assert_equal(raw[cache['frame_ids']-1],cache['source_raw_matched'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
        xy=ns['drop_confidence_cols'](raw);xy[xy==0]=np.nan
        filtered=ns['filter_frames'](ns['impute_frames'](xy))
    rp=filtered[:,18:20];rk=filtered[:,20:22];lp=filtered[:,24:26];lk=filtered[:,26:28]
    scale=(np.linalg.norm(rp-rk,axis=1)+np.linalg.norm(lp-lk,axis=1))/2
    reconstructed=np.stack([np.stack([processed[k:k+124,left],processed[k:k+124,right]]) for k in starts]).astype(np.float32)
    np.testing.assert_array_equal(reconstructed,x)
    if fold not in bounds:
        train=d['x'][d['fold'][d['window_record']]!=fold]
        bounds[fold]=np.array([np.quantile(train[...,j],[.005,.995]) for j in range(12)])
        del train
    bound=bounds[fold];feature=[]
    for j,name in enumerate(feature_names):
        v=x[...,j];feature.append(dict(name=name,train_p005=float(bound[j,0]),train_p995=float(bound[j,1]),
            case_min=float(v.min()),case_max=float(v.max()),outside_fraction=float(((v<bound[j,0])|(v>bound[j,1])).mean())))
    preds={}
    for label,checkpoint in [('control',R/('analysis/window_control/unpacked/results/fold%d/refit/latest.pt'%fold)),('historical',C/('received/checkpoints/fold%d.pt'%fold))]:
        key=(fold,label)
        if key not in models:
            ck=torch.load(checkpoint,map_location='cpu',weights_only=True);m=GdiCNN().double().eval();m.load_state_dict(ck['model'],strict=True)
            assert str(d['person_id'][i]) not in ck['identity']['train_people']
            models[key]=(m,ck);sources[str(checkpoint)]=sha(checkpoint)
        m,ck=models[key]
        with torch.inference_mode():
            v=channels(torch.from_numpy(x).double()).reshape(-1,12,124)
            window_raw=(m(v)[:,0].reshape(-1,2)*float(ck['span'])+float(ck['lo'])).numpy()
        slope,intercept=ck['history'][-1]['calibration'];window_corrected=window_raw*slope+intercept
        target=z if label=='control' else old
        delta=float(np.abs(window_corrected.mean(0)-target['predictions'][i,0]).max());maxreplay=max(maxreplay,delta)
        assert delta<1e-4
        preds[label]=dict(raw=window_raw.mean(0).tolist(),corrected=window_corrected.mean(0).tolist(),
            calibration=[slope,intercept],window_raw=window_raw.tolist(),window_corrected=window_corrected.tolist(),replay_max_error=delta,
            window_raw_min=window_raw.min(0).tolist(),window_raw_max=window_raw.max(0).tolist())
    for w,k in enumerate(starts):
        chunk=raw[k:k+124].reshape(124,25,3);required=chunk[:,[14,11,13,10,12,9,19,22]]
        missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[k:k+124]),[14,11,13,10,12,9,19,22]))
        ss=scale[k:k+124];confidence=required[...,2]
        for side in range(2):
            rows.append(dict(rank=rank,row=i,fold=fold,start0=int(k),side=['left','right'][side],truth=float(d['y'][i,side]),
                control_raw=preds['control']['window_raw'][w][side],control_corrected=preds['control']['window_corrected'][w][side],
                historical_raw=preds['historical']['window_raw'][w][side],historical_corrected=preds['historical']['window_corrected'][w][side],
                required_missing_max=missing,confidence_median=float(np.nanmedian(confidence)),confidence_below02_fraction=float(np.mean((~np.isfinite(confidence))|(confidence<.2))),
                thigh_scale_min=float(ss.min()),thigh_scale_median=float(np.median(ss)),thigh_scale_max=float(ss.max()),
                xy_abs_max=float(np.abs(x[w,side,:,:8]).max())))
    used=np.unique(np.concatenate([np.arange(k,k+124) for k in starts]))
    case=dict(rank=rank,row=i,original_row=orig,fold=fold,person_id=str(d['person_id'][i]),video_id=str(d['video_id'][i]),
        truth=d['y'][i].tolist(),max_absolute_side_error=float(err[i].max()),windows=len(wi),starts0=starts.tolist(),
        raw_frames=len(raw),raw_content_sha256=digest.hexdigest(),raw_preprocess_exact_match=True,
        selected_frames_thigh_scale_pixels=stats(scale[used]),features=feature,predictions=preds)
    cases.append(case);print('CASE',rank,'row',i,'truth',case['truth'],'raw',preds['control']['raw'],'historical',preds['historical']['corrected'],flush=True)
O.mkdir(exist_ok=True)
with (O/'windows.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
save(O/'diagnostics.json',dict(cases=cases,source_hashes=sources,cpu_replay_max_error=maxreplay,
    cpu_replay_tolerance=1e-4,raw_preprocessing_exact_in_all_cases=True,window_predictions_averaged_before_calibration=True,
    archive_resolved=str(archive.resolve()),protected_final_data_opened=False,optimizer_steps=0,
    limitations=['Worst-case selection is descriptive, not evidence of a general exclusion rule',
    'Pixel scales and pose confidence do not establish anatomical validity without video/reference',
    'Training quantiles count overlapping windows; descriptive support checks only',
    'No causal preprocessing ablation or retraining in this analysis']))
print('INPUT_DIAGNOSTICS_FINISHED',maxreplay,flush=True)
