from study import *
spec=importlib.util.spec_from_file_location('scalar_verifier',ref.base.PREV/'verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def check(task,refit=False):
    target,fold,seed=task;tag=f'{target}-long-5-f{fold}-s{seed}'
    ledger=json.loads((STORE/'folds'/f'{tag}.json').read_text());assert ledger['code_sha256']==sha(HERE/'study.py') and sha(ledger['source'])==ledger['source_sha256']
    for suffix,key in [('.csv.gz','predictions_sha256'),('-inner.npz','inner_sha256')]:assert sha(STORE/'folds'/f'{tag}{suffix}')==ledger[key]
    d=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz');ins=np.load(STORE/'folds'/f'{tag}-inner.npz');tr,te,inner,sp=old.cell_data(target,'long',5,fold)
    x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';cl=old.classes(tr[col].to_numpy(),target);y=(cl==old.lowclass(target)).astype(int)
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12)
    assert set(ids).isdisjoint(te.participantid)
    for key,value in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'class4':cl,'inner':inner}.items():np.testing.assert_array_equal(ins[key],value)
    source_out=pd.read_csv(ref.STORE/'folds'/f'{tag}.csv.gz');source_ins=np.load(ref.STORE/'folds'/f'{tag}-inner.npz');low=old.lowclass(target)
    count=0;refits=0;error=0.
    for name in METHODS:
        info=ledger['models'][name]
        if name.startswith('Binary'):
            assert sha(info['path'])==info['sha256'];model=joblib.load(info['path']);v.verify_scaler(model,x,ids);raw=prob(model,xt);count+=1
            for i,mi in enumerate(info['inner_models']):
                assert sha(mi['path'])==mi['sha256'];m=joblib.load(mi['path']);a=inner!=i;b=~a;v.verify_scaler(m,x[a],ids[a])
                np.testing.assert_allclose(prob(m,x[b]),ins[name+'_raw_oof'][b],rtol=0,atol=1e-12);count+=1
            best=min(info['trials'],key=lambda t:t['inner_logloss']);assert best['leaves']==info['selected_leaves']
            z=ins[name+'_raw_oof'];loss=float(np.average(-(y*np.log(np.clip(z,1e-12,1))+(1-y)*np.log(np.clip(1-z,1e-12,1))),weights=weights(ids)));assert abs(loss-best['inner_logloss'])<1e-12
            if refit:
                for i in [0,9]:
                    a=inner!=i if i==0 else np.ones(len(tr),bool)
                    m=old.fit(make(name,seed+fold*10+i,info['selected_leaves']),x[a],y[a],ids[a],'RF')
                    np.testing.assert_array_equal(prob(m,x[inner==0] if i==0 else xt),ins[name+'_raw_oof'][inner==0] if i==0 else raw);refits+=1
        else:
            si=info['source_model'];assert sha(si['path'])==si['sha256'];m=joblib.load(si['path']);v.verify_scaler(m,x,ids)
            if name=='OriginalRF':raw=source_out[f'TurnRF_cal_p{low}'].to_numpy();np.testing.assert_array_equal(ins[name+'_raw_oof'],source_ins['TurnRF_crosscal'][:,low])
            else:
                raw=old.probabilities(m,xt,'RF')[:,low];count+=1
                np.testing.assert_array_equal(ins[name+'_raw_oof'],source_ins[info['source_name']+'_oof'][:,low])
        np.testing.assert_allclose(raw,d[name+'_raw_probability'],rtol=0,atol=1e-12)
        oof=ins[name+'_raw_oof'];cross=ins[name+'_crosscal']
        if name=='OriginalRF':final=raw;np.testing.assert_array_equal(cross,oof);np.testing.assert_array_equal(d[name+'_fpr10'],source_out.TurnRF_cal_fpr10)
        else:
            ci=info['calibrator'];assert sha(ci['path'])==ci['sha256'];m=joblib.load(ci['path']);re=mapper(oof,y,ids)
            np.testing.assert_array_equal(m.coef_,re.coef_);np.testing.assert_array_equal(m.intercept_,re.intercept_)
            final=calibrated(m,raw);count+=1
            for i,ci in enumerate(info['cross_calibrators']):
                a=inner!=i;b=~a;assert sha(ci['path'])==ci['sha256'];m=joblib.load(ci['path']);re=mapper(oof[a],y[a],ids[a])
                np.testing.assert_array_equal(m.coef_,re.coef_);np.testing.assert_array_equal(m.intercept_,re.intercept_)
                np.testing.assert_allclose(cross[b],calibrated(m,oof[b]),rtol=0,atol=1e-12);count+=1
        delta=float(np.max(abs(final-d[name+'_probability'].to_numpy())));assert delta<1e-12;error=max(error,delta)
        for policy,cut in info['thresholds'].items():
            v.independently_check_threshold(cross,y,ids,cut,policy);np.testing.assert_array_equal(final>=cut,d[name+'_'+policy].astype(bool))
    return {'models':count,'refits':refits,'error':error}

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv');assert len(m)==50 and len(mt)==100
    worst=0.
    mean=lambda values,ids:float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        d=pd.concat([pd.read_csv(STORE/'folds'/f'{target}-long-5-f{fold}-s{seed}.csv.gz') for fold in range(5)]).sort_values(['fold']+KEY).reset_index(drop=True)
        ids=d.participantid.to_numpy();y=d.is_low.to_numpy();assert d.participantid.nunique()==(2174 if target=='sppb' else 1939)
        for name in METHODS:
            row=m[m.target.eq(target)&m.seed.eq(seed)&m.method.eq(name)].iloc[0];p=d[name+'_probability'].to_numpy();warn=d[name+'_fpr10'].to_numpy()
            vectors={'brier':(p-y)**2,'logloss':-(y*np.log(np.maximum(p,1e-12))+(1-y)*np.log(np.maximum(1-p,1e-12))),
              'miss':np.where(y==1,warn==0,np.nan),'fpr':np.where(y==0,warn==1,np.nan),'warning_rate':warn.astype(float)}
            for key,value in vectors.items():delta=abs(mean(value,ids)-row[key]);assert delta<1e-10;worst=max(worst,delta)
            for matching in ['speed','speed_cohort']:
                pairs=pd.read_csv(STORE/f'matches-{target}-{matching}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy()
                assert (y[a]==1).all() and (y[b]==0).all() and (ids[a]!=ids[b]).all()
                np.testing.assert_array_equal(d.fold.to_numpy()[a],d.fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
                if matching=='speed_cohort':np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b])
                match=mt[mt.target.eq(target)&mt.seed.eq(seed)&mt.method.eq(name)&mt.matching.eq(matching)].iloc[0]
                delta=p[a]-p[b];order=mean((delta>0).astype(float)+.5*(delta==0),ids[a]);assert abs(order-match.pair_order_accuracy)<1e-10
    return worst

def run(mode):
    initialize();protocol=json.loads((HERE/'protocol.json').read_text());start=time.time()
    for p,h in protocol['sources'].items():assert sha(p)==h
    tasks=list(itertools.product(['sppb','tug'],range(5),SEEDS))
    if mode=='pilot':tasks=[(t,0,SEEDS[0]) for t in ['sppb','tug']]
    results=[]
    for i,task in enumerate(tasks,1):
        tag=f'{task[0]}-long-5-f{task[1]}-s{task[2]}'
        while not (STORE/'folds'/f'{tag}.json').exists():time.sleep(1)
        time.sleep(.05);results.append(check(task,task[1]==0 and task[2] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error=verify_metrics()
    else:metric_error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(tasks),'model_and_calibrator_replays':sum(r['models'] for r in results),'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),'independent_metric_error':metric_error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start})
    print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
