"""Authorized development695 only: official preprocessing and matched124 features.
Read original selected videos, never protected labels. Cache every record to resume.
"""
from pathlib import Path
import ast, hashlib, json, struct, sys, time, warnings, zlib
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[2]
H=ROOT/'gdi-research-harness'
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def source_functions():
    s=HERE/'bundle/sources';meta=load(s/'provenance.json')
    for n,row in meta['files'].items():assert sha(s/Path(n).name)==row['sha256']
    tree=ast.parse((s/'video_process_utils.py').read_text())
    # np.NaN was removed in NumPy 2; spelling-only adaptation, mathematical code unchanged.
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='np' and node.attr=='NaN':node.attr='nan'
    body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign))]
    ns={'np':np,'interpolate':interpolate,'gaussian_filter1d':gaussian_filter1d,'itertools':__import__('itertools')}
    exec(compile(ast.Module(body=body,type_ignores=[]),'pinned_official_utils','exec'),ns)
    nb=load(s/'process_raw_videos.ipynb')
    code=next(''.join(c['source']) for c in nb['cells'] if ''.join(c.get('source',[])).startswith('def process_video_and_add_cols'))
    exec(compile(code,'pinned_official_features','exec'),ns)
    return ns


def context():
    with np.load(STUDY/'cnn-followup695/bundle/data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files if k!='x'}
    assert len(set(d['person_id']))==695 and len(d['y'])==1064
    assert sha(STUDY/'expansion695/train_only.npz')==load(STUDY/'expansion695/analysis/gpu/verification.json')['dataset_sha256']
    old=load(H/'control/desktop-sessions/20261026T100042Z-9d41cd/native124_tasks.json')['tasks']
    old_map={r['video_id']:(i,r) for i,r in enumerate(old) if r['split']=='train'}
    new_map={r['video_id']:r for r in load(H/'studies/20261027-gdi-linkage/development_linkage.json')['records']}
    records=[]
    for i,v in enumerate(d['video_id']):
        if v in old_map:
            seq,r=old_map[v];assert r['patient_id']==d['person_id'][i]
            p=H/('jobs/native124-cache-v1/records/row_%04d.npz'%seq)
        else:
            r=new_map[v];assert r['patient_id']==d['person_id'][i] and r['reserved_role']=='development_candidate'
            p=H/r['artifact'];assert sha(p)==r['sha256']
        with np.load(p,allow_pickle=False) as z:frames=z['frame_ids124'].copy()
        assert len(frames)==124 and (np.diff(frames)==1).all()
        records.append({'index':i,'video_id':str(v),'person_id':str(d['person_id'][i]),
                        'folder':'openpose/'+str(v)+'-processed','frames':frames.tolist(),
                        'source_input':p.relative_to(ROOT).as_posix(),'source_sha256':sha(p)})
    return d,records


def main():
    start=time.monotonic();cache=HERE/'records';cache.mkdir(exist_ok=True)
    ns=source_functions();d,records=context();byfolder={r['folder']:r for r in records}
    archive=ROOT/'outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip'
    stat={'size':archive.stat().st_size,'mtime_ns':archive.stat().st_mtime_ns}
    save(HERE/'input_sources.json',{'archive':str(archive),'archive_stat':stat,'records':records,
                                  'prepare_code_sha256':sha(__file__),'source_provenance_sha256':sha(HERE/'bundle/sources/provenance.json')})
    left=[28,29,26,27,24,25,38,39,50,52,54,56]
    right=[22,23,20,21,18,19,44,45,51,53,55,57]
    checked=0
    def process(row,members,stream):
        nonlocal checked
        dst=cache/('%04d.npz'%row['index']);proof=dst.with_suffix('.json')
        if proof.exists():
            p=load(proof);assert p['video_id']==row['video_id'] and p['source_sha256']==row['source_sha256']
            assert p['archive_stat']==stat and p['prepare_code_sha256']==sha(__file__)
            if p['eligible']:assert sha(dst)==p['artifact_sha256']
            checked+=1;return
        # Source converter reads frames 1..1499 and stops at first absent JSON file.
        last=0
        while last+1 in members and last<1499:last+=1
        raw=np.full((last,75),np.nan,dtype=np.float64)
        ambiguous=[];crc=hashlib.sha256()
        for frame in range(1,last+1):
            m=members[frame];stream.seek(m['offset']);header=stream.read(30)
            assert header[:4]==b'PK\x03\x04'
            nl,el=struct.unpack_from('<HH',header,26);assert stream.read(nl).decode()==m['name'];stream.seek(el,1)
            packed=stream.read(m['compressed']);content=zlib.decompress(packed,-15) if m['method']==8 else packed
            assert len(content)==m['size'] and zlib.crc32(content)&0xffffffff==m['crc']
            crc.update(content);candidates=[]
            for person in json.loads(content)['people']:
                a=np.array(person['pose_keypoints_2d'],dtype=np.float64)
                if a.shape==(75,) and np.isfinite(a).all() and a[::3].max()<320:candidates.append(a)
            # Unlike source's accidental last-person overwrite, ambiguous frames remain missing.
            if len(candidates)==1:raw[frame-1]=candidates[0]
            elif len(candidates)>1:ambiguous.append(frame)
        observed=np.flatnonzero(np.isfinite(raw).any(axis=1))
        if len(observed):raw=raw[:observed[-1]+1]
        ids=np.asarray(row['frames']);reason=None;processed=None;missing=None;multi=[]
        if not len(raw) or ids.min()<1 or ids.max()>len(raw):reason='matched_frames_outside_official_converter_extent'
        else:
            matched=raw[ids-1]
            missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](matched),[14,11,13,10,12,9,19,22]))
            if missing>0.25:reason='matched_window_over_25pct_missing_required_coordinates'
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
                x=np.stack([processed[ids-1][:,left],processed[ids-1][:,right]])
                if not np.isfinite(x).all():reason='nonfinite_required_feature_after_official_processing'
        if processed is not None:
            for k in range(0,500-124,31):
                chunk=raw[k:k+124]
                if len(chunk)!=124:continue
                miss=ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](chunk),[14,11,13,10,12,9,19,22])
                xx=np.stack([processed[k:k+124,left],processed[k:k+124,right]])
                # Advanced indexing above gives time x channel for each side.
                if miss<=.25 and np.isfinite(xx).all():multi.append(k)
        proofdata={**{k:row[k] for k in ('index','video_id','person_id','source_sha256')},
                   'archive_stat':stat,'prepare_code_sha256':sha(__file__),'raw_content_sha256':crc.hexdigest(),
                   'raw_frames_used':len(raw),'canonical_frame_start':int(ids[0]),'canonical_frame_end':int(ids[-1]),
                   'ambiguous_frames':ambiguous,'missing_fraction_max_coordinate':missing,
                   'eligible':reason is None,'exclusion_reason':reason,'eligible_sliding_window_starts':multi}
        if reason is None:
            assert x.shape==(2,124,12)
            np.savez_compressed(dst,x=x.astype(np.float32),frame_ids=ids,
                                source_raw_matched=raw[ids-1].astype(np.float64))
            proofdata['artifact_sha256']=sha(dst)
        save(proof,proofdata);checked+=1
        if checked%25==0:print(json.dumps({'processed':checked,'total':1064,'seconds':round(time.monotonic()-start,1)}),flush=True)
    sys.path.insert(0,str(ROOT/'tools/stanford-sample-check'))
    from inspect_archive import directory_info
    with archive.open('rb',buffering=4*1024*1024) as directory,archive.open('rb') as data:
        count,_,offset=directory_info(directory);directory.seek(offset)
        current=None;members={}
        for _ in range(count):
            header=directory.read(46);assert header[:4]==b'PK\x01\x02'
            f=struct.unpack('<4s6H3L5H2L',header);name=directory.read(f[10]).decode();directory.seek(f[11]+f[12],1)
            group='/'.join(name.split('/')[:2])
            if group!=current:
                if current in byfolder:process(byfolder[current],members,data)
                current=group;members={}
            if group in byfolder and name.endswith('_keypoints.json'):
                frame=int(name.rsplit('_',2)[-2])
                if frame<=1499:
                    assert not f[3]&1 and f[4] in (0,8)
                    members[frame]={'name':name,'offset':f[16],'size':f[9],'compressed':f[8],'crc':f[7],'method':f[4]}
        if current in byfolder:process(byfolder[current],members,data)
    assert checked==1064 and stat=={'size':archive.stat().st_size,'mtime_ns':archive.stat().st_mtime_ns}
    proofs=[load(cache/('%04d.json'%i)) for i in range(1064)]
    keep=np.array([p['eligible'] for p in proofs]);idx=np.flatnonzero(keep)
    x=np.stack([np.load(cache/('%04d.npz'%i),allow_pickle=False)['x'] for i in idx])
    result={k:v[keep] for k,v in d.items()};result.update(x=x,original_row_index=idx)
    np.savez_compressed(HERE/'bundle/data.npz',**result)
    from collections import Counter
    audit={'records_total':1064,'people_total':695,'records_usable':int(keep.sum()),
           'people_usable':len(set(d['person_id'][keep])), 'excluded_reasons':dict(Counter(p['exclusion_reason'] for p in proofs if not p['eligible'])),
           'matched_same124_frame_ids':True,'all_fixed_folds_preserved':True,
           'fold_people':{str(f):len(set(d['person_id'][keep&(d['fold']==f)])) for f in range(5)},
           'sliding_windows_available_for_retained_records':sum(len(p['eligible_sliding_window_starts']) for p in proofs if p['eligible']),
           'records_without_sliding_window':sum(not p['eligible_sliding_window_starts'] for p in proofs if p['eligible']),
           'ambiguous_frames_total':sum(len(p['ambiguous_frames']) for p in proofs),
           'data_sha256':sha(HERE/'bundle/data.npz'),'prepare_seconds':time.monotonic()-start,
           'actual_training_steps':0,'protected_labels_opened':False}
    save(HERE/'input_audit.json',audit);print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
