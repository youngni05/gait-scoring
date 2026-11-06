from study import *
spec=importlib.util.spec_from_file_location('reference_verifier',base.PREV/'verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def mean(values,ids):return float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())

def check(task,refit=False):
    target,cohort,seed=task;tag=f'{target}-{cohort}-s{seed}';l=json.loads((STORE/'folds'/f'{tag}.json').read_text())
    assert l['code_sha256']==sha(HERE/'study.py')
    assert sha(STORE/'folds'/f'{tag}.csv.gz')==l['predictions_sha256'] and sha(STORE/'folds'/f'{tag}-inner.npz')==l['inner_sha256']
    d=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz');ins=np.load(STORE/'folds'/f'{tag}-inner.npz');tr,te,inner,splits=cell(target,cohort);x,xt=inputs(tr),inputs(te)
    ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    assert set(tr.cohort)==set(COHORTS)-{cohort} and te.cohort.eq(cohort).all() and set(ids).isdisjoint(te.participantid)
    assert splits==l['inner_people'] and sorted(ids.tolist())!=[]
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d.original_fold,te.fold)
    for key,value in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}.items():np.testing.assert_array_equal(ins[key],value)
    for i,s in enumerate(splits):
        assert set(s['train']).isdisjoint(s['valid']) and set(s['train'])==set(ids[inner!=i]) and set(s['valid'])==set(ids[inner==i])
        assert set(s['train']).isdisjoint(te.participantid) and set(s['valid']).isdisjoint(te.participantid)
    count=0;error=0.;refits=0
    for fold,source in l['reference_files'].items():
        assert sha(source['predictions'])==source['predictions_sha256'] and sha(source['ledger'])==source['ledger_sha256']
        sl=json.loads(Path(source['ledger']).read_text());sub=d.original_fold.eq(int(fold)).to_numpy();sd=pd.read_csv(source['predictions'])
        selected=te.loc[sub,KEY].merge(sd,on=KEY,validate='one_to_one',how='left')
        for name,original,kind in [('MixedCoreRF','Bout4','Core'),('MixedTurnRF','Turn5','Turn')]:
            info=sl['models'][original];assert sha(info['path'])==info['sha256'];m=joblib.load(info['path']);raw=old.probabilities(m,xt[kind][sub],'RF');count+=1
            for variant,p in [('raw',raw),('cal',old.temper(raw,info['temperature']))]:
                delta=float(np.max(abs(p-d.loc[sub,[f'{name}_{variant}_p{j}' for j in range(4)]].to_numpy())));assert delta<1e-12;error=max(error,delta)
                for policy,cut in info['thresholds'][variant].items():np.testing.assert_array_equal(p[:,old.lowclass(target)]>=cut,d.loc[sub,f'{name}_{variant}_{policy}'].astype(bool))
            for variant in ['raw','cal']:np.testing.assert_allclose(d.loc[sub,[f'{name}_{variant}_p{j}' for j in range(4)]],selected[[f'{original}_{variant}_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    for name,kind in [('LocoCoreRF','Core'),('LocoTurnRF','Turn')]:
        info=l['models'][name];assert sha(info['path'])==info['sha256'];m=joblib.load(info['path']);v.verify_scaler(m,x[kind],ids)
        assert m['model'].n_estimators==128 and m['model'].min_samples_leaf==20 and m['model'].max_features==1.
        raw=old.probabilities(m,xt[kind],'RF');count+=1;np.testing.assert_allclose(raw,d[[f'{name}_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for i,mi in enumerate(info['inner_models']):
            a=inner!=i;b=~a;assert sha(mi['path'])==mi['sha256'];model=joblib.load(mi['path']);v.verify_scaler(model,x[kind][a],ids[a]);np.testing.assert_allclose(old.probabilities(model,x[kind][b],'RF'),ins[name+'_oof'][b],rtol=0,atol=1e-12);count+=1
        oof=ins[name+'_oof'];cross=ins[name+'_crosscal'];t=old.temperature(oof,y,ids);assert abs(t-info['temperature'])<1e-12
        np.testing.assert_allclose(old.temper(raw,t),d[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for i in range(3):
            a=inner!=i;b=~a;ct=old.temperature(oof[a],y[a],ids[a]);assert abs(ct-info['cross_temperatures'][i])<1e-12
            np.testing.assert_allclose(cross[b],old.temper(oof[b],ct),rtol=0,atol=1e-12)
        for variant,ip,op in [('raw',oof,raw),('cal',cross,old.temper(raw,t))]:
            for policy,cut in info['thresholds'][variant].items():
                low=old.lowclass(target);v.independently_check_threshold(ip[:,low],(y==low).astype(int),ids,cut,policy);np.testing.assert_array_equal(op[:,low]>=cut,d[f'{name}_{variant}_{policy}'].astype(bool))
        if refit:
            for i in [0,9]:
                a=inner!=i if i==0 else np.ones(len(tr),bool);model=old.fit(old.make('RF',seed+COHORTS.index(cohort)*10+i),x[kind][a],y[a],ids[a],'RF')
                test=x[kind][inner==0] if i==0 else xt[kind];expected=oof[inner==0] if i==0 else raw
                np.testing.assert_array_equal(old.probabilities(model,test,'RF'),expected);refits+=1
    return {'models':count,'error':error,'refits':refits}

def verify_metrics():
    metrics=pd.read_csv(HERE/'metrics-by-seed.csv');cm=pd.read_csv(HERE/'cohort-metrics-by-seed.csv');matches=pd.read_csv(HERE/'matched-by-seed.csv')
    assert len(metrics)==90 and len(cm)==360 and len(matches)==40
    worst=0.
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        d=pd.concat([pd.read_csv(STORE/'folds'/f'{target}-{c}-s{seed}.csv.gz') for c in COHORTS]).sort_values(['cohort']+KEY).reset_index(drop=True)
        assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939) and len(d)==(7356 if target=='sppb' else 5688)
        y=d['class'].to_numpy();ids=d.participantid.to_numpy()
        for cohort in [None]+COHORTS:
            use=np.ones(len(d),bool) if cohort is None else d.cohort.eq(cohort).to_numpy()
            sub=metrics[metrics.target.eq(target)&metrics.seed.eq(seed)] if cohort is None else cm[cm.target.eq(target)&cm.seed.eq(seed)&cm.cohort.eq(cohort)]
            yy=y[use];ii=ids[use]
            for _,row in sub.iterrows():
                name=row.method;p=d.loc[use,[f'{name}_p{j}' for j in range(4)]].to_numpy();truth=np.eye(4)[yy];pred=p.argmax(1);low=yy==old.lowclass(target)
                arrays={'brier':((p-truth)**2).sum(1),'rps':((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1),'log_loss':-np.log(np.maximum(p[np.arange(len(yy)),yy],1e-12)),'accuracy':(pred==yy).astype(float),'extreme_error':(abs(pred-yy)==3).astype(float),'low_miss':np.where(low,pred!=old.lowclass(target),np.nan),'fpr':np.where(~low,pred==old.lowclass(target),np.nan)}
                if name!='Prior':
                    warning=d.loc[use,f'{name}_fpr10'].to_numpy();arrays.update(fpr10_miss=np.where(low,warning==0,np.nan),fpr10_fpr=np.where(~low,warning==1,np.nan))
                for key,value in arrays.items():delta=abs(mean(value,ii)-row[key]);assert delta<1e-10;worst=max(worst,delta)
        pairs=pd.read_csv(STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy()
        assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all() and (ids[a]!=ids[b]).all()
        np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b]);np.testing.assert_array_equal(d.original_fold.to_numpy()[a],d.original_fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
        for name in METHODS:
            row=matches[matches.target.eq(target)&matches.seed.eq(seed)&matches.method.eq(name+'_cal')].iloc[0];risk=d[f'{name}_cal_p{old.lowclass(target)}'].to_numpy();delta=risk[a]-risk[b]
            assert abs(mean((delta>0).astype(float)+.5*(delta==0),ids[a])-row.pair_order_accuracy)<1e-10
    return worst

def run(mode):
    start=time.time();initialize();protocol=json.loads((HERE/'protocol.json').read_text())
    for p,h in protocol['sources'].items():assert sha(p)==h
    tasks=list(itertools.product(['sppb','tug'],COHORTS,SEEDS))
    if mode=='pilot':tasks=[(t,'MS',SEEDS[0]) for t in ['sppb','tug']]
    result=[]
    for i,t in enumerate(tasks,1):
        tag=f'{t[0]}-{t[1]}-s{t[2]}'
        while not (STORE/'folds'/f'{tag}.json').exists():time.sleep(1)
        time.sleep(.05);result.append(check(t,t[1]=='MS' and t[2] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        error=verify_metrics()
    else:error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(tasks),'model_replays':sum(r['models'] for r in result),'refits':sum(r['refits'] for r in result),'max_prediction_error':max(r['error'] for r in result),'independent_metric_error':error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start});print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
