from study import *
spec=importlib.util.spec_from_file_location('reference_verifier',ref.base.PREV/'verify.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def mean_by_person(values,ids):
    return float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())

def check(task,refit=False):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    ledger=json.loads((STORE/'folds'/f'{tag}.json').read_text());assert ledger['code_sha256']==sha(HERE/'study.py')
    assert sha(ledger['source'])==ledger['source_sha256']
    for suffix,key in [('.csv.gz','predictions_sha256'),('-inner.npz','inner_sha256')]:assert sha(STORE/'folds'/f'{tag}{suffix}')==ledger[key]
    d=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz');ins=np.load(STORE/'folds'/f'{tag}-inner.npz')
    tr,te,inner,split_reference=old.cell_data(target,band,k,fold);x,xt=inputs(tr,k),inputs(te,k);ids=tr.participantid.to_numpy()
    col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12)
    assert set(ids).isdisjoint(te.participantid)
    for key,value in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}.items():np.testing.assert_array_equal(ins[key],value)
    count=0;error=0.;refits=0
    for name in METHODS:
        info=ledger['models'][name];assert sha(info['path'])==info['sha256'];model=joblib.load(info['path'])
        v.verify_scaler(model,x[name],ids)
        assert model['model'].n_estimators==128 and model['model'].min_samples_leaf==20 and model['model'].max_features==1.
        raw=probability(model,xt[name],name);count+=1
        delta=float(np.max(abs(raw-d[[f'{name}_raw_p{j}' for j in range(4)]].to_numpy())));assert delta<1e-12;error=max(error,delta)
        for i,mi in enumerate(info['inner_models']):
            assert sha(mi['path'])==mi['sha256'];m=joblib.load(mi['path']);a=inner!=i;b=~a;v.verify_scaler(m,x[name][a],ids[a])
            np.testing.assert_allclose(probability(m,x[name][b],name),ins[name+'_oof'][b],rtol=0,atol=1e-12);count+=1
        if name=='RF':
            src=np.load(ref.STORE/'folds'/f'{tag}-inner.npz');np.testing.assert_array_equal(ins[name+'_oof'],src[info['reused_name']+'_oof'])
        oof=ins[name+'_oof'];cross=ins[name+'_crosscal'];t=old.temperature(oof,y,ids)
        assert abs(t-info['temperature'])<1e-12
        np.testing.assert_allclose(old.temper(raw,t),d[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for i in range(3):
            a=inner!=i;b=~a;ct=old.temperature(oof[a],y[a],ids[a]);assert abs(ct-info['cross_temperatures'][i])<1e-12
            np.testing.assert_allclose(cross[b],old.temper(oof[b],ct),rtol=0,atol=1e-12)
        for variant,ip,op in [('raw',oof,raw),('cal',cross,old.temper(raw,t))]:
            low=old.lowclass(target)
            for policy,cut in info['thresholds'][variant].items():
                v.independently_check_threshold(ip[:,low],(y==low).astype(int),ids,cut,policy)
                np.testing.assert_array_equal(op[:,low]>=cut,d[f'{name}_{variant}_{policy}'].astype(bool))
        if name=='OrdinalRF' and refit:
            for i in [0,9]:
                a=inner!=i if i==0 else np.ones(len(tr),bool)
                model=old.fit(old.make(name,seed+fold*10+i),x[name][a],y[a],ids[a],name)
                test=x[name][inner==0] if i==0 else xt[name];expected=oof[inner==0] if i==0 else raw
                np.testing.assert_array_equal(probability(model,test,name),expected);refits+=1
    return {'tag':tag,'models':count,'error':error,'refits':refits}

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    assert len(m)==50 and len(mt)==40
    worst=0.;pair_checks=0
    for target,band,k,seed in itertools.product(['sppb','tug'],['long'],[5],SEEDS):
        frames=[pd.read_csv(STORE/'folds'/f'{target}-{band}-{k}-f{fold}-s{seed}.csv.gz') for fold in range(5)]
        d=pd.concat(frames,ignore_index=True).sort_values(['fold']+KEY).reset_index(drop=True)
        y=d['class'].to_numpy();ids=d.participantid.to_numpy();low=y==old.lowclass(target)
        assert d.participantid.nunique()==(2174 if target=='sppb' else 1939)
        assert d.groupby('participantid').fold.nunique().max()==1
        sub=m[m.target.eq(target)&m.band.eq(band)&m.k.eq(k)&m.seed.eq(seed)]
        for _,row in sub.iterrows():
            name=row.method;p=d[[f'{name}_p{j}' for j in range(4)]].to_numpy();pred=p.argmax(1)
            # Deliberately use per-person group means, not training weight helper.
            truth=np.column_stack([y==j for j in range(4)]).astype(float)
            arrays={'brier':np.sum((p-truth)**2,axis=1),
                    'rps':np.sum((np.cumsum(p,axis=1)[:,:3]-np.cumsum(truth,axis=1)[:,:3])**2,axis=1)/3,
                    'log_loss':-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),
                    'accuracy':(pred==y).astype(float),'extreme_error':(abs(pred-y)==3).astype(float),
                    'low_miss':np.where(low,pred!=old.lowclass(target),np.nan),
                    'fpr':np.where(~low,pred==old.lowclass(target),np.nan)}
            if name!='Prior':
                warning=d[f'{name}_fpr10'].to_numpy()
                arrays.update(fpr10_miss=np.where(low,warning==0,np.nan),fpr10_fpr=np.where(~low,warning==1,np.nan))
            for j in range(4):arrays[f'recall_{j}']=np.where(y==j,pred==j,np.nan)
            scores={key:mean_by_person(v,ids) for key,v in arrays.items()}
            scores['balanced_recall']=sum(scores[f'recall_{j}'] for j in range(4))/4
            for key,value in scores.items():
                err=abs(value-row[key]);assert err<=1e-10,(name,key,value,row[key]);worst=max(worst,err)
        for matching in ['speed','speed_cohort']:
            pairs=pd.read_csv(STORE/f'matches-{target}-{band}-{k}-{matching}.csv.gz')
            ai=pairs.positive_index.to_numpy();bi=pairs.negative_index.to_numpy()
            assert low[ai].all() and (~low[bi]).all()
            assert (d.fold.to_numpy()[ai]==d.fold.to_numpy()[bi]).all()
            assert (ids[ai]!=ids[bi]).all()
            gap=abs(d.speed.to_numpy()[ai]-d.speed.to_numpy()[bi]);assert (gap<=.05+1e-12).all()
            np.testing.assert_allclose(gap,pairs.speed_gap,rtol=0,atol=1e-12)
            if matching=='speed_cohort':np.testing.assert_array_equal(d.cohort.to_numpy()[ai],d.cohort.to_numpy()[bi])
            if seed==SEEDS[0]:
                # Independently verify all eligible lows are present and selected controls are nearest.
                found=[]
                for index in np.flatnonzero(low):
                    ok=(~low)&d.fold.eq(d.loc[index,'fold']).to_numpy()&(ids!=ids[index])
                    if matching=='speed_cohort':ok &= d.cohort.eq(d.loc[index,'cohort']).to_numpy()
                    candidates=np.flatnonzero(ok)
                    if not len(candidates):continue
                    diffs=abs(d.speed.to_numpy()[candidates]-d.speed.iloc[index])
                    if diffs.min()>.05+1e-12:continue
                    found.append(index)
                    pair=pairs[pairs.positive_index.eq(index)]
                    assert len(pair)==1
                    assert abs(float(pair.speed_gap.iloc[0])-float(diffs.min()))<=1e-12
                np.testing.assert_array_equal(np.sort(found),np.sort(ai))
            matched=mt[mt.target.eq(target)&mt.band.eq(band)&mt.k.eq(k)&mt.seed.eq(seed)&mt.matching.eq(matching)]
            for _,row in matched.iterrows():
                risk=d[f'{row.method}_p{old.lowclass(target)}'].to_numpy();delta=risk[ai]-risk[bi]
                rate=mean_by_person((delta>0).astype(float)+.5*(delta==0),ids[ai])
                assert abs(rate-row.pair_order_accuracy)<=1e-10
                assert row.matched_low_people==len(np.unique(ids[ai]))
                assert row.unique_control_people==len(np.unique(ids[bi]))
                assert row.low_people==len(np.unique(ids[low]))
                assert abs(row.low_visit_coverage-len(pairs)/low.sum())<=1e-12
                assert abs(row.mean_risk_gap-mean_by_person(delta,ids[ai]))<=1e-12
                warning=d[f'{row.method}_fpr10'].to_numpy()
                assert abs(row.matched_low_miss-mean_by_person((warning[ai]==0).astype(float),ids[ai]))<=1e-12
                assert abs(row.matched_control_warning-mean_by_person((warning[bi]==1).astype(float),ids[ai]))<=1e-12
                pair_checks+=1
    return worst,pair_checks


def run(mode):
    start=time.time();initialize();protocol=json.loads((HERE/'protocol.json').read_text())
    for f,h in protocol['sources'].items():assert sha(f)==h
    tasks=[(t,'long',5,f,s) for t,f,s in itertools.product(['sppb','tug'],range(5),SEEDS)]
    if mode=='pilot':tasks=[(t,'long',5,0,SEEDS[0]) for t in ['sppb','tug']]
    results=[]
    for i,task in enumerate(tasks,1):
        tag=f'{task[0]}-{task[1]}-{task[2]}-f{task[3]}-s{task[4]}'
        while not (STORE/'folds'/f'{tag}.json').exists():time.sleep(1)
        time.sleep(.05);results.append(check(task,task[3]==0 and task[4] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error,pairs=verify_metrics()
    else:metric_error,pairs=None,0
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(tasks),'model_replays':sum(r['models'] for r in results),'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),'independent_metric_error':metric_error,'matched_checks':pairs,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start})
    print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
