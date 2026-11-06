import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;BASE=H/'studies/20261028-missing-augmentation-learning';Q=H/'studies/20261028-position-quality'
sys.stdout.reconfigure(encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def people(v,ids):return np.array([np.mean(v[ids==p],axis=0) for p in sorted(set(ids))])
def stats(v):return dict(mean=float(np.mean(v)),median=float(np.median(v)),p10=float(np.percentile(v,10)),p90=float(np.percentile(v,90)))
def rank(x):
    _,inv,counts=np.unique(x,return_inverse=True,return_counts=True);mid=np.cumsum(counts)-(counts-1)/2;return mid[inv]
def rho(x,y):
    a,b=rank(x),rank(y)
    return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def compare(a,b):
    d=a-b;rng=np.random.default_rng(20261026);ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5]);return dict(reduction=float(d.mean()),bootstrap95=ci.tolist(),people_reduced=int((d>0).sum()),people_increased=int((d<0).sum()))
def main():
    assert load(S/'preparation.json')['verified'];assert sha(S/'features.npz')==load(S/'preparation.json')['features_sha256']
    tracked={}
    for folder,names in [(BASE,['models.npz','predictions.csv']),(Q,['quality_rows.csv'])]:
        sealed={a['path']:a['sha256'] for a in load(folder/'study_receipt.json')['artifacts']}
        for name in names:
            p=folder/name;assert sha(p)==sealed[p.relative_to(H).as_posix()];tracked[p.relative_to(R).as_posix()]=sha(p)
    with np.load(S/'features.npz',allow_pickle=False) as z:
        ids=z['patient_ids'].copy();vid=list(map(str,z['video_ids']));features={k:z[k].copy() for k in z.files if k not in ['patient_ids','video_ids','original_indices']}
    with np.load(BASE/'models.npz',allow_pickle=False) as z:models={k:z[k].copy() for k in z.files if k.startswith('fold')}
    records=[r for r in csv.DictReader((BASE/'predictions.csv').open(encoding='utf-8')) if r['arm']=='baseline' and r['seed']=='20261028' and r['position'] in ['front','last']];old={(r['video_id'],r['position']):r for r in records};assert len(records)==406
    folds=np.array([int(old[v,'front']['fold']) for v in vid]);pred={k:np.full((203,2),np.nan) for k in features};zfeat={k:np.full((203,512),np.nan) for k in features}
    for fold in range(5):
        ix=folds==fold;assert len(set(ids[ix]))==35 and not set(ids[ix])&set(ids[~ix]);pre=f'fold{fold}__'
        for k,x in features.items():
            zz=(x[ix]-models[pre+'scaler_mean'])/models[pre+'scaler_scale'];zfeat[k][ix]=zz;pred[k][ix]=zz@models[pre+'coef'].T+models[pre+'intercept']
    replay=0.
    for p in ['front','last']:
        target=np.array([[float(old[v,p]['prediction_L']),float(old[v,p]['prediction_R'])] for v in vid]);err=float(np.max(np.abs(target-pred[p])));assert err<1e-8;replay=max(replay,err)
    fd=zfeat['last']-zfeat['front'];pd=pred['last']-pred['front'];fsize=lambda v:people(np.sqrt(np.mean(v*v,axis=1)),ids);psize=lambda v:people(np.abs(v).mean(1),ids)
    original_f,original_p=fsize(fd),psize(pd);out={};maxdecomp=0.;values={'original_feature_gap':original_f,'original_score_gap':original_p}
    for kind in ['translation','scale','both']:
        item={}
        for field,dic,measure,original in [('feature',zfeat,fsize,original_f),('score',pred,psize,original_p)]:
            ff,fl,lf,ll=(dic[k] for k in ['front',kind+'_FL',kind+'_LF','last']);n=((fl-ff)+(ll-lf))/2;g=((lf-ff)+(ll-fl))/2;err=float(np.max(np.abs((ll-ff)-n-g)));assert err<1e-8;maxdecomp=max(maxdecomp,err)
            nsize,gsize=measure(n),measure(g);common_f,common_l=measure(lf-ff),measure(ll-fl);common_mean=(common_f+common_l)/2
            item[field]={'global_trajectory_component':stats(nsize),'remaining_component':stats(gsize),'common_front_trajectory_gap':stats(common_f),'common_last_trajectory_gap':stats(common_l),'mean_common_trajectory_gap':stats(common_mean),'common_trajectory_vs_original':compare(original,common_mean),'interaction_component':stats(measure(ll-lf-fl+ff))}
            values[kind+'_'+field+'_common_mean']=common_mean
        out[kind]=item
    cf=fsize(zfeat['confidence_last']-zfeat['confidence_front']);cp=psize(pred['confidence_last']-pred['confidence_front']);confidence={'feature_gap':stats(cf),'score_gap':stats(cp),'feature_vs_original':compare(original_f,cf),'score_vs_original':compare(original_p,cp)};values['confidence_feature_gap']=cf;values['confidence_score_gap']=cp
    metadata=load(S/'diagnostic_rows.json');rawstats={k:stats(people(np.array([a[k] for a in metadata]),ids)) for k in ['knee_unaligned_abs_deg','knee_distribution_abs_deg','scale_last_over_front_median','root_front_span','root_last_span']}
    qrows={}
    for r in csv.DictReader((Q/'quality_rows.csv').open(encoding='utf-8')):
        if r['video_id'] in set(vid) and r['position'] in ['front','last']:
            assert r['split']=='train';qrows[r['video_id'],r['position']]=r
    assert len(qrows)==406;quality={};correlations={}
    metrics=['lower_confidence_mean','lower_below_0_3_fraction','all_joint_missing_fraction','projected_bone_length_cv','relative_lower_acceleration_p90','knee_angle_range_deg','shoulder_width_over_torso']
    for k in metrics:
        front=people(np.array([float(qrows[v,'front'][k]) for v in vid]),ids);last=people(np.array([float(qrows[v,'last'][k]) for v in vid]),ids);diff=abs(last-front)
        quality[k]={'front_mean':float(front.mean()),'last_mean':float(last.mean())};correlations[k]={'absolute_change_vs_feature_gap':rho(diff,original_f),'absolute_change_vs_score_gap':rho(diff,original_p)}
    for k in ['knee_unaligned_abs_deg','knee_distribution_abs_deg']:
        d=people(np.array([x[k] for x in metadata]),ids);correlations[k]={'vs_feature_gap':rho(d,original_f),'vs_score_gap':rho(d,original_p)}
    with (S/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','video_id','fold','condition','prediction_L','prediction_R'])
        for key,p in pred.items():
            for i,v in enumerate(vid):w.writerow([ids[i],v,folds[i],key,*p[i]])
    np.savez_compressed(S/'person_diagnostics.npz',patient_ids=np.array(sorted(set(ids))),**values)
    for p in [Path(__file__),S/'plan.md',S/'features.npz']:tracked[p.relative_to(R).as_posix()]=sha(p)
    save('analysis_sources.json',tracked)
    result={'people':175,'records':203,'folds':5,'new_model_fits':0,'baseline_replay_max':replay,'decomposition_max_error':maxdecomp,'original_feature_gap':stats(original_f),'original_score_gap':stats(original_p),'crossovers':out,'confidence_common':confidence,'raw_pose_diagnostics':rawstats,'reused_quality_training175':quality,'exploratory_correlations':correlations,'protected_or_validation_data_computed':False,'causal_camera_vs_true_motion_identified':False,'counterfactual_accuracy_not_evaluated':True}
    save('result.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
