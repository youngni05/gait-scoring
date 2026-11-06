from study import *

def mean(values,ids):return float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())

def check_scale(m,x,ids):
    center=np.average(x,axis=0,weights=weights(ids));variance=np.average((x-center)**2,axis=0,weights=weights(ids))
    np.testing.assert_allclose(m['scale'].mean_,center,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(m['scale'].var_,variance,rtol=1e-10,atol=1e-12)
    assert m['model'].n_estimators==128 and m['model'].min_samples_leaf==20 and m['model'].max_features==1.

def check_threshold(p,y,ids,cut,policy):
    neg=y==0;pos=~neg
    if policy=='fpr10':
        assert mean((p[neg]>=cut).astype(float),ids[neg])<=.1+1e-12
        candidates=np.r_[0.,np.unique(p),1.00000001];index=np.flatnonzero(candidates==cut);assert len(index)==1
        if index[0]>0:assert mean((p[neg]>=candidates[index[0]-1]).astype(float),ids[neg])>.1-1e-12
    elif policy=='sens80':assert mean((p[pos]>=cut).astype(float),ids[pos])>=.8-1e-12
    else:assert cut==.5

def check(task,refit=False):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';l=json.loads((STORE/'folds'/f'{tag}.json').read_text())
    assert l['code_sha256']==sha(HERE/'study.py') and l['predictions_sha256']==sha(STORE/'folds'/f'{tag}.csv.gz') and l['inner_sha256']==sha(STORE/'folds'/f'{tag}-inner.npz')
    d=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz');ins=np.load(STORE/'folds'/f'{tag}-inner.npz');tr,te,inner,split=old.cell_data(target,'long',5,fold);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    assert set(tr.participantid).isdisjoint(te.participantid)
    for key,val in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}.items():np.testing.assert_array_equal(ins[key],val)
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d['class'],old.classes(te[col].to_numpy(),target))
    for i in range(3):assert set(ids[inner!=i]).isdisjoint(ids[inner==i])
    count=0;error=0.;refits=0
    for name in METHODS:
        info=l['models'][name];assert sha(info['path'])==info['sha256'];m=joblib.load(info['path']);check_scale(m,x[name],ids);raw=old.probabilities(m,xt[name],'RF');count+=1
        np.testing.assert_allclose(raw,d[[f'{name}_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        oof=ins[name+'_oof'];cross=ins[name+'_crosscal']
        for i,mi in enumerate(info['inner_models']):
            a=inner!=i;b=~a;assert sha(mi['path'])==mi['sha256'];model=joblib.load(mi['path']);check_scale(model,x[name][a],ids[a]);prediction=old.probabilities(model,x[name][b],'RF');err=float(np.max(abs(prediction-oof[b])));assert err<1e-12;error=max(error,err);count+=1
        t=old.temperature(oof,y,ids);assert abs(t-info['temperature'])<1e-12;cal=old.temper(raw,t)
        np.testing.assert_allclose(cal,d[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for i in range(3):
            a=inner!=i;b=~a;t_i=old.temperature(oof[a],y[a],ids[a]);assert abs(t_i-info['cross_temperatures'][i])<1e-12
            np.testing.assert_allclose(cross[b],old.temper(oof[b],t_i),rtol=0,atol=1e-12)
        for variant,ip,op in [('raw',oof,raw),('cal',cross,cal)]:
            for policy,cut in info['thresholds'][variant].items():
                check_threshold(ip[:,old.lowclass(target)],(y==old.lowclass(target)).astype(int),ids,cut,policy)
                np.testing.assert_array_equal(op[:,old.lowclass(target)]>=cut,d[f'{name}_{variant}_{policy}'].astype(bool))
        if name in l['source_files']:
            src=l['source_files'][name]
            for key in ['ledger','inner','predictions']:assert sha(src[key])==src[key+'_sha256']
            oldins=np.load(src['inner']);np.testing.assert_array_equal(oldins[src['source_name']+'_oof'],oof)
        if refit and name in ['SpeedBouts','SpeedCadence']:
            for i in [0,9]:
                a=inner!=i if i==0 else np.ones(len(tr),bool);m=old.fit(old.make('RF',seed+fold*10+i),x[name][a],y[a],ids[a],'RF')
                z=x[name][inner==0] if i==0 else xt[name];expected=oof[inner==0] if i==0 else raw
                np.testing.assert_array_equal(old.probabilities(m,z,'RF'),expected);refits+=1
    return {'models':count,'refits':refits,'error':error}

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv');assert len(m)==90 and len(mt)==40;worst=0.
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        d=pd.concat([pd.read_csv(STORE/'folds'/f'{target}-f{f}-s{seed}.csv.gz') for f in range(5)]).sort_values(['fold']+KEY).reset_index(drop=True)
        assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939) and len(d)==(7356 if target=='sppb' else 5688)
        y=d['class'].to_numpy();ids=d.participantid.to_numpy();truth=np.eye(4)[y];low=y==old.lowclass(target)
        for _,r in m[m.target.eq(target)&m.seed.eq(seed)].iterrows():
            name=r.method;p=d[[f'{name}_p{j}' for j in range(4)]].to_numpy();pred=p.argmax(1);assert p.min()>=0 and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-12)
            vectors={'accuracy':(pred==y).astype(float),'brier':((p-truth)**2).sum(1),'rps':((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1),'log_loss':-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),'extreme_error':(abs(pred-y)==3).astype(float),'low_miss':np.where(low,pred!=old.lowclass(target),np.nan),'fpr':np.where(~low,pred==old.lowclass(target),np.nan)}
            recalls=[]
            for j in range(4):
                recall=mean((pred[y==j]==j).astype(float),ids[y==j]);assert abs(recall-r[f'recall_{j}'])<1e-10;recalls.append(recall)
            assert abs(np.mean(recalls)-r.balanced_recall)<1e-10
            if name!='Prior':warn=d[f'{name}_fpr10'].to_numpy();vectors.update(fpr10_miss=np.where(low,warn==0,np.nan),fpr10_fpr=np.where(~low,warn==1,np.nan))
            for key,value in vectors.items():err=abs(mean(value,ids)-r[key]);assert err<1e-10;worst=max(worst,err)
        pairs=pd.read_csv(STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy();assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all() and (ids[a]!=ids[b]).all()
        np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b]);np.testing.assert_array_equal(d.fold.to_numpy()[a],d.fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
        for name in METHODS:
            risk=d[f'{name}_cal_p{old.lowclass(target)}'].to_numpy();diff=risk[a]-risk[b];row=mt[mt.target.eq(target)&mt.seed.eq(seed)&mt.method.eq(name+'_cal')].iloc[0]
            assert abs(mean((diff>0).astype(float)+.5*(diff==0),ids[a])-row.pair_order_accuracy)<1e-10
    return worst

def run(mode):
    initialize();start=time.time();p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(HERE/'study.py')
    for path,digest in p['sources'].items():assert sha(path)==digest
    results=[]
    for i,t in enumerate(tasks(mode),1):
        while not (STORE/'folds'/f'{t[0]}-f{t[1]}-s{t[2]}.json').exists():time.sleep(1)
        results.append(check(t,t[1]==0 and t[2] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error=verify_metrics()
    else:metric_error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(results),'model_replays':sum(r['models'] for r in results),'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),'independent_metric_error':metric_error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start});print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
