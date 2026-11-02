import csv,json,hashlib,time,sys
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent
CV=H/'control/desktop-sessions/20261026T041807Z-f050df';SHORT=H/'control/desktop-sessions/20261026T130506Z-e876e7'
sys.stdout.reconfigure(encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def persons(v,ids):return np.array([v[ids==p].mean() for p in sorted(set(ids))])
def comparison(a,b):
    d=a-b;rng=np.random.default_rng(20261026)
    ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1),[2.5,97.5])
    return dict(reduction=float(d.mean()),paired_bootstrap95=ci.tolist(),people_improved=int((d>0).sum()),people_worsened=int((d<0).sum()))
def fit(x,y):
    mu=x.mean(0);scale=x.std(0);scale[scale==0]=1
    z=(x-mu)/scale;center=z.mean(0);ym=y.mean(0);a=z-center;b=y-ym
    u,s,vh=np.linalg.svd(a,full_matrices=False);coef=(vh.T@((s/(s*s+1))[:,None]*(u.T@b))).T;intercept=ym-coef@center
    dual=np.linalg.solve(a@a.T+np.eye(len(a)),b);direct=(a.T@dual).T
    np.testing.assert_allclose(coef,direct,rtol=0,atol=1e-7)
    return dict(scaler_mean=mu,scaler_scale=scale,coef=coef,intercept=intercept),float(np.sum(s*s/(s*s+1))),float(np.max(np.abs(coef-direct)))
def predict(x,p):return ((x-p['scaler_mean'])/p['scaler_scale'])@p['coef'].T+p['intercept']
def main():
    start=time.perf_counter();assert not (S/'result.json').exists();assert load(S/'preparation.json')['verified']
    assert sha(S/'features.npz')==load(S/'preparation.json')['features_sha256']
    for folder,filename in ((CV,'internal_cv_protocol.json'),(SHORT,'length_internal_oof.csv')):
        sealed={a['path']:a['sha256'] for a in load(folder/'receipt.json')['artifacts']};assert sha(folder/filename)==sealed[(folder/filename).relative_to(H).as_posix()]
    protocol=load(CV/'internal_cv_protocol.json');old=list(csv.DictReader((SHORT/'length_internal_oof.csv').open(encoding='utf-8')));lookup={r['video_id']:r for r in old};assert len(old)==len(lookup)==203
    with np.load(S/'features.npz',allow_pickle=False) as f:
        ids,vid=(f[k].copy() for k in ('patient_ids','video_ids'))
        x={name:{pos:f[pos+'_'+name].copy() for pos in ('front','middle','last')} for name in ('baseline','valid_pool','rhythm')}
    x['combined']={p:np.concatenate([x['baseline'][p],x['rhythm'][p]],axis=1) for p in x['baseline']}
    assert len(vid)==203 and len(set(ids))==175 and set(map(str,vid))==set(lookup)
    y=np.array([[float(lookup[str(v)]['actual_L']),float(lookup[str(v)]['actual_R'])] for v in vid]);folds=np.array([int(lookup[str(v)]['fold']) for v in vid])
    for i,v in enumerate(vid):assert str(ids[i])==lookup[str(v)]['patient_id']
    for k,pp in enumerate(protocol['fold_people']):assert set(map(str,ids[folds==k]))==set(pp)
    pred={m:{p:np.full_like(y,np.nan) for p in x[m]} for m in x};params={};fit_info=[];coeferr=0.;visited=np.zeros(203,dtype=int)
    for fold in range(5):
        te=folds==fold;tr=~te;assert len(set(ids[te]))==35 and len(set(ids[tr]))==140 and not set(ids[te])&set(ids[tr]);visited[te]+=1
        for m,xx in x.items():
            p,dof,err=fit(xx['front'][tr],y[tr]);coeferr=max(coeferr,err)
            for pos in xx:pred[m][pos][te]=predict(xx[pos][te],p)
            params.update({f'{m}_fold{fold}__{k}':v for k,v in p.items()});fit_info.append(dict(method=m,fold=fold,train_people=140,held_people=35,train_records=int(tr.sum()),held_records=int(te.sum()),effective_df=dof))
    assert (visited==1).all()
    oldpred=np.array([[float(lookup[str(v)]['short124_L']),float(lookup[str(v)]['short124_R'])] for v in vid])
    replay=float(np.max(np.abs(oldpred-pred['baseline']['front'])));assert replay<1e-7
    np.savez_compressed(S/'fold_models.npz',**params)
    with np.load(S/'fold_models.npz',allow_pickle=False) as f:
        for m in x:
            for k in range(5):
                p={n:f[f'{m}_fold{k}__{n}'] for n in ('scaler_mean','scaler_scale','coef','intercept')}
                for pos in x[m]:np.testing.assert_allclose(predict(x[m][pos][folds==k],p),pred[m][pos][folds==k],rtol=0,atol=1e-10)
    errors={};ranges={};stats={}
    for m,pp in pred.items():
        pp['mean3']=np.stack(list(pp.values())).mean(0)
        errors[m]={p:persons(np.abs(v-y).mean(1),ids) for p,v in pp.items()}
        ranges[m]=persons(np.ptp(np.stack([pp[p] for p in ('front','middle','last')]),axis=0).mean(1),ids)
        stats[m]=dict(person_mae={p:float(e.mean()) for p,e in errors[m].items()},range_mean=float(ranges[m].mean()),range_median=float(np.median(ranges[m])),prediction_sd_mean3=float(np.std(persons(pp['mean3'].mean(1),ids),ddof=1)),fold_mae={str(k):{p:float(persons(np.abs(v[folds==k]-y[folds==k]).mean(1),ids[folds==k]).mean()) for p,v in pp.items()} for k in range(5)})
    comparisons={}
    for ref,can in (('baseline','valid_pool'),('baseline','rhythm'),('baseline','combined'),('valid_pool','combined')):
        comparisons[can+'_vs_'+ref]=dict(mae={p:comparison(errors[ref][p],errors[can][p]) for p in errors[ref]},range=comparison(ranges[ref],ranges[can]),front_folds_improved=sum(stats[can]['fold_mae'][str(k)]['front']<stats[ref]['fold_mae'][str(k)]['front'] for k in range(5)))
    with (S/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','video_id','fold','method','position','actual_L','actual_R','prediction_L','prediction_R'])
        for m,pp in pred.items():
            for pos,values in pp.items():
                for i in range(203):w.writerow([ids[i],vid[i],folds[i],m,pos,*y[i],*values[i]])
    diag=load(S/'diagnostics.json');flat={p:[sum(d[p]['flat_spectra'][i] for d in diag) for i in range(3)] for p in ('front','middle','last')}
    undef={p:sum(d[p]['undefined_bilateral_correlation'] for d in diag) for p in flat}
    tracked={p.relative_to(R).as_posix():sha(p) for p in (Path(__file__),S/'plan.md',S/'features.npz',CV/'internal_cv_protocol.json',SHORT/'length_internal_oof.csv')};save('analysis_sources.json',tracked)
    result=dict(people=175,records=203,folds=5,source_partition='existing train only',validation37_used=False,protected38_or223_used=False,alpha=1,new_candidate_methods=3,new_candidate_fold_fits=15,baseline_fold_refits_for_replay=5,baseline_oof_replay_max_difference=replay,independent_coef_max_difference=coeferr,metrics=stats,comparisons=comparisons,fit_details=fit_info,flat_signal_counts=flat,undefined_correlation_counts=undef,elapsed_seconds=time.perf_counter()-start,independent_final_performance=False)
    save('result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('fit_details',)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
