"""Frozen-model diagnostic interventions; never choose inputs by resulting score."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;ROOT=STUDY.parents[2]
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD');OUT=DATA/'cause';OUT.mkdir(exist_ok=True)
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle';sys.path.insert(0,str(BUNDLE))
import infer,preprocess
def csvread(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')
def csvsave(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(v):
    v=np.asarray(v);return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),min=float(v.min()),max=float(v.max()))

def development():
    dev=STUDY.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695';prior=STUDY.parent/'20261031-ljmu-normal-cnn/diagnostics'
    source=json.loads((prior/'sources.json').read_text());assert infer.sha(BUNDLE/'models.json')==source['models_sha256'];assert infer.sha(dev/'bundle/data.npz')==source['dev_sha256']
    with np.load(dev/'bundle/data.npz',allow_pickle=False) as d:
        y=d['y'];people=d['person_id'];fold=d['fold'];videos=d['video_id']
    with np.load(prior/'development_all_models.npz',allow_pickle=False) as z:
        predictions=z['predictions'];assert np.array_equal(z['person_id'],people)
    oof=predictions[fold,np.arange(len(y))];trained=np.stack([predictions[f,i] for i in range(len(y)) for f in range(5) if f!=fold[i]]).reshape(len(y),4,2).mean(1)
    with np.load(dev/'analysis/cnn_oof.npz',allow_pickle=False) as z:
        assert np.array_equal(z['y'],y) and np.array_equal(z['person_id'],people)
        replay=float(np.abs(oof-z['predictions'][:,0]).max());assert replay<1e-4
    bins=[]
    for lo,hi in [(0,60),(60,70),(70,80),(80,90),(90,100),(100,120)]:
        mask=(y>=lo)&(y<hi);bins.append(dict(lower=lo,upper=hi,side_scores=int(mask.sum()),people=len(np.unique(np.repeat(people,2)[mask.ravel()])),true_mean=float(y[mask].mean()),oof_mean=float(oof[mask].mean()),oof_bias=float((oof-y)[mask].mean()),training_mean=float(trained[mask].mean())))
    group=[dict(person=str(p),true=float(y[people==p].mean()),oof=float(oof[people==p].mean())) for p in np.unique(people)]
    yy=np.array([r['true'] for r in group]);pp=np.array([r['oof'] for r in group])
    result=dict(records=len(y),people=len(group),side_scores=y.size,bins=bins,person_equal_MAE=float(np.abs(yy-pp).mean()),person_mean_prediction_vs_true_slope=float(np.polyfit(yy,pp,1)[0]),person_mean_correlation=float(np.corrcoef(yy,pp)[0,1]),oof_replay_max_error=replay,model_manifest_same=True,meaning='Development high/low GDI regression behaviour; not healthy/abnormal classification ground truth',sources={'development':str(dev/'bundle/data.npz'),'saved_all_models':str(prior/'development_all_models.npz')})
    save(HERE/'development.json',result);csvsave(HERE/'development_bins.csv',bins);print('Development recomputation complete',flush=True)

def main():
    infer.verify_bundle();infer.configure();development()
    conditions={r['record_id']:r for r in csvread(STUDY/'conditions_shift0.csv')};reviews=csvread(STUDY/'reviewed_shift0.csv')
    baseline=np.load(DATA/'runs/confidence_primary/prepared/prepared.npz',allow_pickle=False)
    ids=baseline['record_id'].tolist();records={};generation=[];features=[];keys=[];rng=np.random.default_rng(20261031)
    variants=['baseline','translate','scale_half','scale_double','horizontal_reflection','confidence_mask_0_1','jitter_bbox_1pct','toe_gap10','static_pose']
    for r in reviews:
        key=r['record_id'];folder=Path(r['json_dir']);selection={a['filename']:int(a['person_index']) for a in csvread(Path(r['selection_csv']))};paths=sorted(folder.glob('*_keypoints.json'));raw=np.full((len(paths),25,3),np.nan)
        for i,p in enumerate(paths):
            ix=selection[p.name]
            if ix>=0:raw[i]=np.asarray(json.loads(p.read_text())['people'][ix]['pose_keypoints_2d']).reshape(25,3)
        records[key]=raw;finite=np.isfinite(raw[:,:,:2])&(raw[:,:,:2]!=0);height=float(conditions[key]['bbox_height_pixels'])
        for variant in variants:
            a=raw.copy()
            if variant=='translate':a[:,:,:2][finite]+=173.25
            elif variant=='scale_half':a[:,:,:2][finite]*=.5
            elif variant=='scale_double':a[:,:,:2][finite]*=2
            elif variant=='horizontal_reflection':
                valid=finite[:,:,0];a[:,:,0][valid]=2*float(conditions[key]['width'])-a[:,:,0][valid]
            elif variant=='confidence_mask_0_1':a[:,:,:2][np.isfinite(a[:,:,2])&(a[:,:,2]<.1)]=0
            elif variant=='jitter_bbox_1pct':
                noise=rng.normal(0,.01*height,size=a[:,:,:2].shape);a[:,:,:2][finite]+=noise[finite]
            elif variant=='toe_gap10':a[57:67,[19,22],:2]=0
            elif variant=='static_pose':
                complete=np.flatnonzero(np.isfinite(a[:,preprocess.REQUIRED,:2]).all(axis=(1,2))&(a[:,preprocess.REQUIRED,:2]!=0).all(axis=(1,2)))
                if not len(complete):continue
                a[:]=a[complete[len(complete)//2]]
            try:ex=preprocess.extract_windows(a.reshape(len(a),75))
            except ValueError as err:generation.append(dict(record_id=key,variant=variant,status='no_eligible_windows',reason=str(err)));continue
            if variant=='baseline':
                mask=baseline['window_record']==ids.index(key);assert np.array_equal(ex['x'],baseline['x'][mask]);assert np.array_equal(ex['window_start'],baseline['window_start'][mask])
            generation.append(dict(record_id=key,variant=variant,status='prepared',windows=len(ex['x'])))
            for x,start in zip(ex['x'],ex['window_start']):features.append(x);keys.append((key,variant,int(start)))
    # Temporal sensitivity on identical centers and observed support; no extrapolation.
    for key in ids:
        ws=np.flatnonzero(baseline['window_record']==ids.index(key));last=int(baseline['window_start'][ws].max())+124;timeline=np.full((2,last,12),np.nan,dtype=np.float32)
        for w in ws:
            st=int(baseline['window_start'][w]);timeline[:,st:st+124]=baseline['x'][w]
        for w in ws:
            st=int(baseline['window_start'][w]);center=st+61.5;maxpos=center+(np.arange(124)-61.5)*1.2;lo=int(np.floor(maxpos.min()));hi=int(np.ceil(maxpos.max()))
            if lo<0 or hi>=last or not np.isfinite(timeline[:,lo:hi+1]).all():continue
            for rate in [.8,1.,1.2]:
                pos=center+(np.arange(124)-61.5)*rate;l=np.floor(pos).astype(int);h=np.ceil(pos).astype(int);frac=(pos-l)[None,:,None];x=(timeline[:,l]*(1-frac)+timeline[:,h]*frac).astype(np.float32)
                if rate==1:assert np.array_equal(x,baseline['x'][w])
                features.append(x);keys.append((key,'time_'+str(rate),st))
    x=np.stack(features);scores=[]
    for fold in range(5):
        raw=infer.predict_windows(x,fold);slope,intercept=json.loads((BUNDLE/'models.json').read_text())['folds'][fold]['calibration_slope_intercept'];scores.append(raw.astype(np.float64)*slope+intercept);print('Diagnostic inference fold',fold,'windows',len(x),flush=True)
    score=np.stack(scores);np.savez_compressed(OUT/'perturbations.npz',x=x,predictions=score,record_id=np.array([k[0] for k in keys]),variant=np.array([k[1] for k in keys]),window_start=np.array([k[2] for k in keys]))
    mean=score.mean(axis=(0,2));lookup={k:i for i,k in enumerate(keys)};result=[]
    for variant in sorted({k[1] for k in keys}-{'baseline','time_1.0'}):
        ref='time_1.0' if variant.startswith('time_') else 'baseline'
        for key in ids:
            pairs=[(i,lookup[(key,ref,start)]) for i,(r,v,start) in enumerate(keys) if r==key and v==variant and (key,ref,start) in lookup]
            if not pairs:continue
            ii,jj=zip(*pairs);base=float(mean[list(jj)].mean());new=float(mean[list(ii)].mean());result.append(dict(record_id=key,video_id=conditions[key]['video_id'],cluster_id=conditions[key]['cluster_id'],label=conditions[key]['label'],view=conditions[key]['view'],variant=variant,common_windows=len(pairs),baseline_score=base,changed_score=new,delta=new-base))
    csvsave(HERE/'sensitivity_records.csv',result);save(HERE/'generation.json',generation)
    summary={}
    for variant in sorted({r['variant'] for r in result}):
        selected=[r for r in result if r['variant']==variant];v=np.array([r['delta'] for r in selected]);summary[variant]=dict(records=len(v),mean_delta=float(v.mean()),mean_abs_delta=float(np.abs(v).mean()),max_abs_delta=float(np.abs(v).max()),by_label={label:stats([r['delta'] for r in selected if r['label']==label]) for label in ['Normal Gait','Abnormal Gait'] if any(r['label']==label for r in selected)})
    # Compare baseline output directly with untouched original five-fold CSV.
    saved=csvread(DATA/'runs/confidence_primary/predictions/window_predictions.csv');errors=[]
    for row in saved:
        ix=lookup[(row['record_id'],'baseline',int(row['start_offset']))] if 'start_offset' in row else lookup[(row['record_id'],'baseline',int(row['frame_first'])-int(next(r['annotation_frame_first'] for r in reviews if r['record_id']==row['record_id'])))]
        errors.extend(abs(score[int(row['fold']),ix]-[float(row['left_GDI']),float(row['right_GDI'])]))
    error=float(max(errors));assert error<1e-4
    save(HERE/'sensitivity_summary.json',dict(variants=summary,baseline_features_bitwise_equal=True,baseline_prediction_replay_max_error=error,comparison='Same record and common window starts; changes are diagnostic counterfactuals, not corrected estimates',seed=20261031))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
