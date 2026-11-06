import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;BASE=H/'studies/20261028-missing-augmentation-learning'
sys.stdout.reconfigure(encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def people(a,ids):return np.array([a[ids==p].mean(axis=0) for p in sorted(set(ids))])
def stats(a):return {'mean':float(np.mean(a)),'median':float(np.median(a))}
def delta(a,b):
    d=a-b;rng=np.random.default_rng(20261026);ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5]);return {'reduction':float(d.mean()),'bootstrap95':ci.tolist(),'people_improved':int((d>0).sum()),'people_worsened':int((d<0).sum())}
def main():
    assert load(S/'inference.json')['verified'];assert sha(S/'features.npz')==load(S/'inference.json')['features_sha256']
    receipt={a['path']:a['sha256'] for a in load(BASE/'study_receipt.json')['artifacts']};tracked={}
    for name in ['models.npz','predictions.csv']:
        p=BASE/name;assert sha(p)==receipt[p.relative_to(H).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
    with np.load(S/'features.npz',allow_pickle=False) as z:ids=z['patient_ids'].copy();vid=list(map(str,z['video_ids']));source_rows=z['source_rows'].copy();features={k:z[k].copy() for k in z.files if k not in ['patient_ids','video_ids','source_rows']}
    with np.load(BASE/'models.npz',allow_pickle=False) as z:models={k:z[k].copy() for k in z.files if k.startswith('fold')}
    source={}
    for row in csv.DictReader((BASE/'predictions.csv').open(encoding='utf-8')):
        if row['arm']=='baseline' and row['seed']=='20261028' and row['position'] in ['front','last'] and row['video_id'] in set(vid):source[row['video_id'],row['position']]=row
    assert len(source)==2*len(vid)
    y=np.array([[float(source[v,'front']['actual_L']),float(source[v,'front']['actual_R'])] for v in vid]);folds=np.array([int(source[v,'front']['fold']) for v in vid])
    for i,v in enumerate(vid):assert source[v,'front']['patient_id']==str(ids[i])
    pred={k:np.full((len(ids),2),np.nan) for k in features};zfeat={k:np.full((len(ids),512),np.nan) for k in features}
    for fold in range(5):
        ix=folds==fold;pre=f'fold{fold}__'
        for k,f in features.items():
            z=(f[ix]-models[pre+'scaler_mean'])/models[pre+'scaler_scale'];zfeat[k][ix]=z;pred[k][ix]=z@models[pre+'coef'].T+models[pre+'intercept']
    replay=0.
    for pos in ['front','last']:
        expected=np.array([[float(source[v,pos]['prediction_L']),float(source[v,pos]['prediction_R'])] for v in vid]);err=float(abs(expected-pred['original_'+pos]).max());assert err<1e-8;replay=max(replay,err)
    metrics={};vectors={}
    for arm in ['original','control','aligned']:
        pp={p:pred[arm+'_'+p] for p in ['front','last']};pp['mean2']=(pp['front']+pp['last'])/2
        e={p:people(abs(a-y).mean(1),ids) for p,a in pp.items()};gap=people(abs(pp['last']-pp['front']).mean(1),ids);fdelta=zfeat[arm+'_last']-zfeat[arm+'_front'];fgap=people(np.sqrt((fdelta*fdelta).mean(1)),ids)
        metrics[arm]={'person_mae':{p:float(a.mean()) for p,a in e.items()},'score_gap':stats(gap),'feature_gap':stats(fgap),'prediction_person_sd':{p:float(people(a.mean(1),ids).std(ddof=1)) for p,a in pp.items()}}
        vectors[arm]={**{'mae_'+p:v for p,v in e.items()},'score_gap':gap,'feature_gap':fgap}
    comparisons={}
    for ref,can in [('original','control'),('control','aligned'),('original','aligned')]:comparisons[can+'_vs_'+ref]={k:delta(v,vectors[can][k]) for k,v in vectors[ref].items()}
    manifest=load(S/'cycle_manifest.json');rows=[manifest[int(i)] for i in source_rows]
    durations={p:{'median':float(np.median([a['duration_frames'][p] for a in rows])),'min':min(a['duration_frames'][p] for a in rows),'max':max(a['duration_frames'][p] for a in rows),'median_abs_start_shift':float(np.median([abs(a['start_shift_frames'][p]) for a in rows]))} for p in ['front','last']}
    missing={k:float(people(np.array([a['missing_fraction'][k] for a in rows]),ids).mean()) for k in ['control_front','control_last','aligned_front','aligned_last']}
    same=sum(all(r['aligned_bounds'][p]==r['control_bounds'][p] for p in ['front','last']) for r in rows)
    with (S/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','video_id','fold','condition','position','actual_L','actual_R','prediction_L','prediction_R'])
        for arm in ['original','control','aligned']:
            for p in ['front','last','mean2']:
                values=pred[arm+'_'+p] if p!='mean2' else (pred[arm+'_front']+pred[arm+'_last'])/2
                for i,v in enumerate(vid):w.writerow([ids[i],v,folds[i],arm,p,*y[i],*values[i]])
    np.savez_compressed(S/'person_metrics.npz',patient_ids=np.array(sorted(set(ids))),**{arm+'__'+k:v for arm,dd in vectors.items() for k,v in dd.items()})
    for p in [Path(__file__),S/'plan.md',S/'features.npz',S/'cycle_manifest.json']:tracked[p.relative_to(R).as_posix()]=sha(p)
    save('analysis_sources.json',tracked)
    result={'people':len(set(ids)),'records':len(vid),'people_per_original_fold':{str(k):len(set(ids[folds==k])) for k in range(5)},'new_model_fits':0,'baseline_replay_max':replay,'metrics':metrics,'comparisons':comparisons,'selected_two_cycle_duration_frames':durations,'selected_period_agreement25':sum(r['period_agreement25'] is True for r in rows),'identical_control_and_aligned_records':same,'missing_fractions':missing,'generalizes_to_all175':False,'clinical_gait_events_validated':False,'time_resampling_changes_cadence':True,'protected_data_used':False}
    save('result.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
