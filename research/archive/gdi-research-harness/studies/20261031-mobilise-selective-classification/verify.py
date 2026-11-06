"""Independent artifact, person-split, scaler, rule and prediction replay checks."""
import study as s
from study import HERE,BASE,SETS,STABILITY,FEATURES,SEEDS,MODELS,KEY,np,pd,joblib,sha,save
import json,itertools,os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

def w(ids):
    values=list(ids);counts=Counter(values)
    a=np.fromiter((1/counts[i] for i in values),dtype=float,count=len(values))
    return a/a.sum()
def lg(p):
    p=np.minimum(np.maximum(p,1e-6),1-1e-6);return np.log(p/(1-p))
def binary(a,target):return (a<=3 if target=='sppb' else a>20).astype(int)

def verify_rules(rules,p,internal,y,ids,frame,prefix):
    delta=0.
    for policy,rule in rules.items():
        t=rule['threshold'];pos=y==1;neg=y==0
        tpr=np.dot(w(ids[pos]),internal[pos]>=t);fpr=np.dot(w(ids[neg]),internal[neg]>=t)
        candidates=np.r_[0.,np.unique(internal),1.00000001]
        if policy=='sens80':
            assert tpr>=.8-1e-12
            nxt=candidates[candidates>t]
            if len(nxt):assert np.dot(w(ids[pos]),internal[pos]>=nxt[0])<.8+1e-12
        if policy=='fpr10':
            assert fpr<=.1+1e-12
            prev=candidates[candidates<t]
            if len(prev):assert np.dot(w(ids[neg]),internal[neg]>=prev[-1])>.1-1e-12
        if policy=='p50':assert t==.5
        im=abs(lg(internal)-lg(t));om=abs(lg(p)-lg(t))
        for reject,cut in rule['reject_margin'].items():
            reject=float(reject)
            if reject==0:assert cut==-1.
            else:
                # Verify the mathematical weighted-quantile inequalities, without
                # changing the saved cutoff or its strict rejection convention.
                ww=w(ids);assert np.any(im==cut)
                assert np.sum(ww[im<cut])<=reject+1e-12
                assert np.sum(ww[im<=cut])>=reject-1e-12
            expected=np.where(om<cut,-1,(p>=t).astype(int))
            np.testing.assert_array_equal(expected,frame[f'{prefix}_{policy}_r{int(reject*100)}'])
    return delta

def run():
    protocol=json.loads((HERE/'protocol.json').read_text());assert sha(HERE/'study.py')==protocol['code_sha256']
    for path,digest in protocol['source_hashes'].items():assert sha(path)==digest,path
    rp=json.loads((HERE/'regression/protocol.json').read_text());assert rp['code_sha256']==sha(HERE/'regression_control.py')
    data=pd.read_csv(BASE/'features.csv');data=data[data.draw.eq(0)]
    assert not data.duplicated(['target'] if 'target' in data else KEY+['band','k']).any()
    people=pd.read_csv(BASE/'person-folds.csv');assert not people.participantid.duplicated().any()
    assert data.groupby('participantid').fold.nunique().eq(1).all()
    jobs=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    assert len(list((HERE/'folds').glob('*.json')))==300
    def check(task):
        target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}';original=json.loads((SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json').read_text())
        led=json.loads((HERE/'folds'/f'{tag}.json').read_text());out=pd.read_csv(HERE/'folds'/f'{tag}.csv');ins=np.load(HERE/'folds'/f'{tag}-inner.npz')
        assert sha(HERE/'folds'/f'{tag}.csv')==led['predictions_sha256'];assert sha(HERE/'folds'/f'{tag}-inner.npz')==led['inner_sha256']
        cell=data[data[target+'_ok']&data.band.eq(band)&data.k.eq(k)].sort_values(KEY);tr=cell[cell.fold.ne(fold)];te=cell[cell.fold.eq(fold)]
        assert led['train_people']==original['train_people'] and led['test_people']==original['test_people'] and led['inner_people']==original['inner_people']
        assert set(tr.participantid)==set(led['train_people']) and set(te.participantid)==set(led['test_people'])
        assert not set(tr.participantid)&set(te.participantid)
        np.testing.assert_array_equal(out[KEY].to_numpy(),te[KEY].to_numpy())
        col='total_sppb_score' if target=='sppb' else 'tugtime';np.testing.assert_array_equal(out.actual,te[col])
        np.testing.assert_array_equal(ins['participantid'],tr.participantid);np.testing.assert_array_equal(ins['y'],binary(tr[col].to_numpy(),target))
        np.testing.assert_array_equal(out.positive,binary(te[col].to_numpy(),target))
        for i,sp in enumerate(original['inner_people']):
            assert set(ins['participantid'][ins['inner']==i])==set(sp['valid']);assert set(sp['train'])|set(sp['valid'])==set(tr.participantid)
        prior=np.dot(w(tr.participantid),ins['y']);np.testing.assert_allclose(out.prior,prior,rtol=0,atol=1e-12)
        mx=0.;x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();ww=w(tr.participantid)
        for name in ['LR','RF']:
            z=led['models'][name];assert sha(z['path'])==z['sha256'];bundle=joblib.load(z['path']);m=bundle['model'];c=bundle['calibrator']
            xm=np.sum(x*ww[:,None],axis=0);var=np.sum((x-xm)**2*ww[:,None],axis=0)
            np.testing.assert_allclose(m['scale'].mean_,xm,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(m['scale'].var_,var,rtol=1e-12,atol=1e-12)
            zz=lg(ins[name+'_oof']);np.testing.assert_allclose(c['scale'].mean_,np.dot(ww,zz),rtol=1e-12,atol=1e-12)
            raw=m.predict_proba(xt)[:,1];cal=c.predict_proba(lg(raw)[:,None])[:,1]
            for variant,ip,op in [('raw',ins[name+'_oof'],raw),('cal',ins[name+'_crosscal'],cal)]:
                err=float(np.max(abs(out[name+'_'+variant]-op)));mx=max(mx,err);assert err<1e-12
                verify_rules(z['rules'][variant],op,ip,ins['y'],ins['participantid'],out,name+'_'+variant)
        return mx
    with ThreadPoolExecutor(max_workers=4) as pool:errors=list(pool.map(check,jobs))
    # Refit a preselected example for both targets, independent of performance.
    refits=[]
    for target in ['sppb','tug']:
        tag=f'{target}-long-5-f0-s{SEEDS[0]}';led=json.loads((HERE/'folds'/f'{tag}.json').read_text());ins=np.load(HERE/'folds'/f'{tag}-inner.npz')
        cell=data[data[target+'_ok']&data.band.eq('long')&data.k.eq(5)].sort_values(KEY);tr=cell[cell.fold.ne(0)];te=cell[cell.fold.eq(0)]
        for name in ['LR','RF']:
            m=s.fit(s.make(name,SEEDS[0]+9),tr[FEATURES].to_numpy(),ins['y'],tr.participantid.to_numpy());b=joblib.load(led['models'][name]['path'])
            err=float(np.max(abs(m.predict_proba(te[FEATURES].to_numpy())-b['model'].predict_proba(te[FEATURES].to_numpy()))));assert err<1e-12
            c=s.calibration(lg(ins[name+'_oof']),ins['y'],ins['participantid']);z=lg(ins[name+'_oof'])[:,None]
            ce=float(np.max(abs(c.predict_proba(z)-b['calibrator'].predict_proba(z))));assert ce<1e-10
            refits.append({'target':target,'model':name,'refit_max_difference':err,'calibrator_refit_max_difference':ce})
    regmax=0.
    for target,band,k,fold in itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5)):
        tag=f'{target}-{band}-{k}-f{fold}';led=json.loads((HERE/'regression'/f'{tag}.json').read_text());out=pd.read_csv(HERE/'regression'/f'{tag}.csv');ins=np.load(HERE/'regression'/f'{tag}-inner.npz')
        assert sha(HERE/'regression'/f'{tag}.csv')==led['predictions_sha256']
        te=data[data[target+'_ok']&data.band.eq(band)&data.k.eq(k)&data.fold.eq(fold)].sort_values(KEY)
        for name,z in led['models'].items():
            assert sha(z['original_model'])==z['original_sha256'] and sha(z['calibrator'])==z['calibrator_sha256']
            m=joblib.load(z['original_model'])
            if hasattr(m['model'],'n_jobs'):m['model'].n_jobs=1
            pred=m.predict(te[FEATURES].to_numpy());c=joblib.load(z['calibrator']);p=c.predict_proba((pred*(-1 if target=='sppb' else 1))[:,None])[:,1]
            err=float(np.max(abs(p-out[name+'_cal'])));regmax=max(regmax,err);assert err<1e-12
            verify_rules(z['rules'],p,ins[name+'_crosscal'],ins['y'],ins['participantid'],out,name+'_cal')
    # Recompute saved summary from row-level predictions using separate metric formulas.
    from analyze import load
    allp=load();metrics=pd.read_csv(HERE/'metrics-by-seed.csv');metricmax=0.
    for target in ['sppb','tug']:
        for seed in SEEDS:
            d=allp[(allp.target==target)&(allp.band=='long')&(allp.k==5)&(allp.seed==seed)]
            for method in ['RF_cal_sens80_r0','RF_cal_sens80_r20','SVRreg_cal_fpr10_r0']:
                dec=d[method].to_numpy();y=d.positive.to_numpy();ids=d.participantid.to_numpy()
                manual={'defer_rate':np.dot(w(ids),dec==-1),'miss_rate':np.dot(w(ids[y==1]),dec[y==1]==0),'detect_rate':np.dot(w(ids[y==1]),dec[y==1]==1),'fpr':np.dot(w(ids[y==0]),dec[y==0]==1),'positive_defer_rate':np.dot(w(ids[y==1]),dec[y==1]==-1)}
                row=metrics[(metrics.target==target)&(metrics.band=='long')&(metrics.k==5)&(metrics.seed==seed)&(metrics.method==method)].iloc[0]
                for key,value in manual.items():metricmax=max(metricmax,abs(value-row[key]));assert abs(value-row[key])<1e-12
    save(HERE/'verification.json',{'status':'passed','classifier_jobs':300,'regression_control_jobs':60,'source_hashes_checked':len(protocol['source_hashes']),
         'all_saved_models_replayed':720,'classifier_max_probability_difference':max(errors),'regression_max_probability_difference':regmax,'metric_max_difference':metricmax,
         'refits':refits,'leakage':'outer test people absent from training, calibration and rule selection; train scaler moments independently recomputed',
         'analysis_code_sha256':sha(HERE/'analyze.py'),'verification_code_sha256':sha(__file__)})
    print('VERIFICATION_PASSED',max(errors),regmax,metricmax,flush=True)

if __name__=='__main__':
    with s.threadpool_limits(limits=1):run()
