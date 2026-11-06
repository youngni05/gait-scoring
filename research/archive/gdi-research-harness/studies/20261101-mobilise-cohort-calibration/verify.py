from study import *
def mean(values,ids):
    return float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())

def verify_scaler(model,x,ids):
    center=np.average(x,axis=0,weights=weights(ids));variance=np.average((x-center)**2,axis=0,weights=weights(ids))
    np.testing.assert_allclose(model['scale'].mean_,center,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(model['scale'].var_,variance,rtol=1e-10,atol=1e-12)

def independently_check_threshold(p,y,ids,cut,policy):
    neg=y==0;pos=~neg
    if policy=='fpr10':
        assert mean((p[neg]>=cut).astype(float),ids[neg])<=.1+1e-12
        candidates=np.r_[0.,np.unique(p),1.00000001];ix=np.flatnonzero(candidates==cut);assert len(ix)==1
        if ix[0]>0:assert mean((p[neg]>=candidates[ix[0]-1]).astype(float),ids[neg])>.1-1e-12
    elif policy=='sens80':assert mean((p[pos]>=cut).astype(float),ids[pos])>=.8-1e-12
    else:assert cut==.5

def check(task,refit=False):
    target,held,seed=task;tag=f'{target}-{held}-s{seed}';folder=STORE/'folds'
    l=json.loads((folder/f'{tag}.json').read_text());assert l['code_sha256']==sha(HERE/'study.py')
    assert sha(folder/f'{tag}.csv.gz')==l['predictions_sha256'] and sha(folder/f'{tag}-inner.npz')==l['inner_sha256']
    assert sha(l['source_ledger'])==l['source_ledger_sha256'] and sha(l['source_predictions'])==l['source_predictions_sha256']
    tr,te,inner,sources=cell(target,held);d=pd.read_csv(folder/f'{tag}.csv.gz');ins=np.load(folder/f'{tag}-inner.npz');source=pd.read_csv(l['source_predictions'])
    col='total_sppb_score' if target=='sppb' else 'tugtime';ids=tr.participantid.to_numpy();y=old.classes(tr[col].to_numpy(),target)
    assert set(tr.cohort)==set(COHORTS)-{held} and te.cohort.eq(held).all() and set(ids).isdisjoint(te.participantid)
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d.original_fold,te.fold)
    for key,val in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner,'cohorts':tr.cohort.to_numpy(dtype='U')}.items():np.testing.assert_array_equal(ins[key],val)
    x=ref.inputs(tr)['Turn'];xt=ref.inputs(te)['Turn'];final=l['final_model'];assert sha(final['path'])==final['sha256']
    m=joblib.load(final['path']);verify_scaler(m,x,ids);raw=old.probabilities(m,xt,'RF');error=0.;refits=0
    np.testing.assert_allclose(raw,d[[f'RawRF_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    for variant,name in [('raw','RawRF'),('cal','PersonCal')]:
        for c in source:
            prefix=f'LocoTurnRF_{variant}_'
            if c.startswith(prefix):np.testing.assert_allclose(d[name+'_'+c[len(prefix):]],source[c],rtol=0,atol=1e-12)
    source_ledger=json.loads(Path(l['source_ledger']).read_text());baseline=source_ledger['models']['LocoTurnRF']
    np.testing.assert_allclose(old.temper(raw,baseline['temperature']),d[[f'PersonCal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    for i,mi in enumerate(l['inner_models']):
        a=inner!=i;b=~a;sp=l['splits'][i]
        assert sp['valid_cohort']==sources[i] and set(tr.loc[a,'cohort'])==set(sources)-{sources[i]}
        assert sorted(set(ids[a].tolist()))==sp['train_people'] and sorted(set(ids[b].tolist()))==sp['valid_people']
        assert set(ids[a]).isdisjoint(ids[b]) and set(ids[a]).isdisjoint(te.participantid) and set(ids[b]).isdisjoint(te.participantid)
        assert sha(mi['path'])==mi['sha256'];model=joblib.load(mi['path']);verify_scaler(model,x[a],ids[a])
        assert model['model'].n_estimators==128 and model['model'].min_samples_leaf==20 and model['model'].max_features==1.
        pred=old.probabilities(model,x[b],'RF');delta=float(np.max(abs(pred-ins['Domain_oof'][b])));error=max(error,delta);assert delta<1e-12
        if refit and i in [0,2]:
            rebuilt=old.fit(old.make('RF',seed+COHORTS.index(held)*10+i),x[a],y[a],ids[a],'RF')
            np.testing.assert_array_equal(old.probabilities(rebuilt,x[b],'RF'),ins['Domain_oof'][b]);refits+=1
    oof=ins['Domain_oof'];cross=ins['Domain_crosscal'];cal=l['calibration'];temperature=old.temperature(oof,y,ids)
    assert abs(temperature-cal['temperature'])<1e-12
    outcal=old.temper(raw,temperature);np.testing.assert_allclose(outcal,d[[f'CohortCal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    np.testing.assert_array_equal(d[[f'CohortCal_p{j}' for j in range(4)]],d[[f'CohortGuard_p{j}' for j in range(4)]])
    np.testing.assert_array_equal(raw.argmax(1),outcal.argmax(1))
    for i in range(3):
        a=inner!=i;b=~a;t=old.temperature(oof[a],y[a],ids[a]);assert abs(t-cal['cross_temperatures'][i])<1e-12
        np.testing.assert_allclose(cross[b],old.temper(oof[b],t),rtol=0,atol=1e-12)
    low=old.lowclass(target);binary=(y==low).astype(int)
    for policy,cut in cal['thresholds']['cal'].items():
        independently_check_threshold(cross[:,low],binary,ids,cut,policy)
        np.testing.assert_array_equal(outcal[:,low]>=cut,d[f'CohortCal_{policy}'].astype(bool))
    for i,c in enumerate(sources):
        b=inner==i;cut=l['guard_group_cuts'][c];independently_check_threshold(cross[b,low],binary[b],ids[b],cut,'fpr10')
        neg=b & (binary==0);assert mean((cross[neg,low]>=l['guard_cut']).astype(float),ids[neg])<=.1+1e-12
    assert l['guard_cut']==max(l['guard_group_cuts'].values())
    np.testing.assert_array_equal(outcal[:,low]>=l['guard_cut'],d.CohortGuard_fpr10.astype(bool))
    return {'models':4,'error':error,'refits':refits}

def verify_metrics():
    metrics=pd.read_csv(HERE/'metrics-by-seed.csv');cm=pd.read_csv(HERE/'cohort-metrics-by-seed.csv');matched=pd.read_csv(HERE/'matched-by-seed.csv')
    assert len(metrics)==50 and len(cm)==200 and len(matched)==40;worst=0.
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        d=pd.concat([pd.read_csv(STORE/'folds'/f'{target}-{c}-s{seed}.csv.gz') for c in COHORTS]).sort_values(['cohort']+KEY).reset_index(drop=True)
        assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939) and len(d)==(7356 if target=='sppb' else 5688)
        for cohort in [None]+COHORTS:
            sub=d if cohort is None else d[d.cohort.eq(cohort)];y=sub['class'].to_numpy();ids=sub.participantid.to_numpy();truth=np.eye(4)[y];low=y==old.lowclass(target)
            scores=metrics[metrics.target.eq(target)&metrics.seed.eq(seed)] if cohort is None else cm[cm.target.eq(target)&cm.seed.eq(seed)&cm.cohort.eq(cohort)]
            for _,r in scores.iterrows():
                name=r.method;p=sub[[f'{name}_p{j}' for j in range(4)]].to_numpy();pred=p.argmax(1)
                np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-12);assert np.isfinite(p).all() and p.min()>=0
                vectors={'brier':((p-truth)**2).sum(1),'rps':((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1),'log_loss':-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),'accuracy':(pred==y).astype(float),'extreme_error':(abs(pred-y)==3).astype(float),'low_miss':np.where(low,pred!=old.lowclass(target),np.nan),'fpr':np.where(~low,pred==old.lowclass(target),np.nan)}
                if name!='Prior':
                    warn=sub[f'{name}_fpr10'].to_numpy();vectors.update(fpr10_miss=np.where(low,warn==0,np.nan),fpr10_fpr=np.where(~low,warn==1,np.nan),warning_rate=warn)
                for key,val in vectors.items():
                    delta=abs(mean(val,ids)-r[key]);assert delta<1e-10;worst=max(worst,delta)
        pairs=pd.read_csv(STORE/f'matches-{target}.csv.gz');aa=pairs.positive_index.to_numpy();bb=pairs.negative_index.to_numpy();y=d['class'].to_numpy();ids=d.participantid.to_numpy()
        assert (y[aa]==old.lowclass(target)).all() and (y[bb]!=old.lowclass(target)).all() and (ids[aa]!=ids[bb]).all()
        np.testing.assert_array_equal(d.cohort.to_numpy()[aa],d.cohort.to_numpy()[bb]);np.testing.assert_array_equal(d.original_fold.to_numpy()[aa],d.original_fold.to_numpy()[bb]);assert (abs(d.speed.to_numpy()[aa]-d.speed.to_numpy()[bb])<=.05+1e-12).all()
        for name in METHODS:
            row=matched[matched.target.eq(target)&matched.seed.eq(seed)&matched.method.eq(name)].iloc[0];risk=d[f'{name}_p{old.lowclass(target)}'].to_numpy();diff=risk[aa]-risk[bb]
            assert abs(mean((diff>0).astype(float)+.5*(diff==0),ids[aa])-row.pair_order_accuracy)<1e-10
    return worst

def run(mode):
    initialize();start=time.time();protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['code_sha256']==sha(HERE/'study.py')
    for path,digest in protocol['sources'].items():assert sha(path)==digest
    results=[]
    for i,t in enumerate(tasks(mode),1):
        while not (STORE/'folds'/f'{t[0]}-{t[1]}-s{t[2]}.json').exists():time.sleep(1)
        results.append(check(t,t[1]=='MS' and t[2] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        error=verify_metrics()
    else:error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(results),'model_replays':sum(r['models'] for r in results),'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),'independent_metric_error':error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start})
    print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
