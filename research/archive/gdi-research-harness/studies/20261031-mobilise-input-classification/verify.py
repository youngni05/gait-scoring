"""Replay models and decisions; independently recompute person-equal metrics/matches."""
from study import *

def mean_by_person(values, ids):
    series=pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean()
    return float(series.mean()) if len(series) else None

def verify_scaler(model, x, ids):
    w=weights(ids)
    mean=np.average(x,axis=0,weights=w)
    var=np.average((x-mean)**2,axis=0,weights=w)
    np.testing.assert_allclose(model['scale'].mean_,mean,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(model['scale'].var_,var,rtol=1e-10,atol=1e-12)

def verify_ds_scaler(bundle, x, ids):
    w=weights(ids);w/=w.sum();flat=x.reshape(-1,6);ww=np.repeat(w/x.shape[1],x.shape[1])
    mean=np.average(flat,axis=0,weights=ww)
    std=np.maximum(np.sqrt(np.average((flat-mean)**2,axis=0,weights=ww)),1e-8)
    np.testing.assert_allclose(bundle['mean'],mean,rtol=0,atol=1e-12)
    np.testing.assert_allclose(bundle['std'],std,rtol=0,atol=1e-12)

def check_model(info, x, train_x, train_ids, name):
    path=Path(info['path']);assert path.drive.upper()=='D:'
    assert sha(path)==info['sha256']
    if name=='DeepSets':
        bundle=dict(np.load(path));verify_ds_scaler(bundle,train_x,train_ids)
        pred=predict_ds(bundle,x)
        np.testing.assert_allclose(pred,predict_ds(bundle,x[:,::-1]),rtol=0,atol=1e-12)
        return pred
    model=joblib.load(path);verify_scaler(model,train_x,train_ids)
    assert model['model'].n_estimators==128 and model['model'].min_samples_leaf==20
    assert model['model'].max_features==1.
    return old.probabilities(model,x,'RF')

def independently_check_threshold(p,y,ids,cut,policy):
    neg=y==0;pos=~neg
    fp=mean_by_person((p[neg]>=cut).astype(float),ids[neg])
    tp=mean_by_person((p[pos]>=cut).astype(float),ids[pos])
    if policy=='fpr10':
        assert fp<=.1+1e-12
        candidates=np.r_[0.,np.unique(p),1.00000001];ix=np.flatnonzero(candidates==cut)
        assert len(ix)==1
        if ix[0]>0:
            prev=candidates[ix[0]-1]
            assert mean_by_person((p[neg]>=prev).astype(float),ids[neg])>.1-1e-12
    elif policy=='sens80':assert tp>=.8-1e-12
    else:assert cut==.5

def verify_task(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
    path=STORE/'folds'/f'{tag}.json';ledger=json.loads(path.read_text())
    assert ledger['task']==list(task) and ledger['code_sha256']==sha(HERE/'study.py')
    assert ledger['split_sha256']==sha(old.SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json')
    pred_path=STORE/'folds'/f'{tag}.csv.gz';inner_path=STORE/'folds'/f'{tag}-inner.npz'
    assert sha(pred_path)==ledger['predictions_sha256'];assert sha(inner_path)==ledger['inner_sha256']
    out=pd.read_csv(pred_path);ins=np.load(inner_path)
    tr,te,inner,ref=old.cell_data(target,band,k,fold);x=inputs(tr,k);xt=inputs(te,k)
    np.testing.assert_array_equal(out[KEY].to_numpy(),te[KEY].to_numpy())
    col='total_sppb_score' if target=='sppb' else 'tugtime'
    y=old.classes(tr[col].to_numpy(),target);ids=tr.participantid.to_numpy()
    np.testing.assert_array_equal(out['class'],old.classes(te[col].to_numpy(),target))
    np.testing.assert_array_equal(out.actual,te[col]);np.testing.assert_array_equal(out.speed,te.averagestridespeed)
    np.testing.assert_array_equal(ins['ids'],ids);np.testing.assert_array_equal(ins['inner'],inner)
    assert set(ids).isdisjoint(te.participantid)
    for i,split in enumerate(ref['inner_people']):
        assert set(ids[inner==i])==set(split['valid'])
        assert set(ids[inner!=i])==set(split['train'])
        assert set(split['valid']).isdisjoint(split['train'])
    final_error=0.;inner_error=0.;replays=0
    for name in METHODS:
        info=ledger['models'][name]
        raw=check_model(info,xt[name],x[name],ids,name);replays+=1
        expected=out[[f'{name}_raw_p{j}' for j in range(4)]].to_numpy()
        error=float(np.max(abs(raw-expected)));assert error<=1e-12;final_error=max(final_error,error)
        if name=='MedianRF':
            source=Path(info['reused_reference']);assert sha(source)==info['reference_sha256']
            previous=pd.read_csv(old.STORE/'folds'/f'{tag}.csv.gz')
            for v in ['raw','cal']:
                np.testing.assert_allclose(out[[f'{name}_{v}_p{j}' for j in range(4)]],
                    previous[[f'RF_{v}_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
            oldins=np.load(old.STORE/'folds'/f'{tag}-inner.npz')
            np.testing.assert_array_equal(ins[name+'_oof'],oldins['RF_oof'])
        else:
            for i,model_info in enumerate(info['inner_models']):
                a=inner!=i;b=~a
                p=check_model(model_info,x[name][b],x[name][a],ids[a],name);replays+=1
                error=float(np.max(abs(p-ins[name+'_oof'][b])))
                assert error <= (1e-5 if name=='DeepSets' else 1e-12)
                inner_error=max(inner_error,error)
        if name=='DeepSets':
            combined=np.average(info['inner_curves'],axis=0,weights=info['inner_people'])
            np.testing.assert_allclose(combined,info['combined_curve'],rtol=0,atol=1e-12)
            assert EPOCHS[int(np.argmin(combined))]==info['epochs']
        oof=ins[name+'_oof'];cross=ins[name+'_crosscal']
        for i in range(3):
            a=inner!=i;b=~a;t=old.temperature(oof[a],y[a],ids[a])
            assert abs(t-info['cross_temperatures'][i])<=1e-12
            np.testing.assert_allclose(cross[b],old.temper(oof[b],t),rtol=0,atol=1e-12)
        t=old.temperature(oof,y,ids);assert abs(t-info['temperature'])<=1e-12
        cal=old.temper(raw,t)
        np.testing.assert_allclose(cal,out[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
        for variant,ip,op in [('raw',oof,raw),('cal',cross,cal)]:
            low=old.lowclass(target)
            for policy,cut in info['thresholds'][variant].items():
                independently_check_threshold(ip[:,low],(y==low).astype(int),ids,cut,policy)
                np.testing.assert_array_equal(op[:,low]>=cut,out[f'{name}_{variant}_{policy}'].to_numpy().astype(bool))
    return {'tag':tag,'replays':replays,'final_error':final_error,'inner_error':inner_error,
            'epochs':ledger['models']['DeepSets']['epochs']}

def refit_checks():
    rows=[]
    for target in ['sppb','tug']:
        for seed in [SEEDS[0],SEEDS[-1]]:
            fold=0;k=5;tag=f'{target}-long-5-f0-s{seed}'
            tr,te,inner,_=old.cell_data(target,'long',5,0);x=inputs(tr,k);xt=inputs(te,k)
            label='total_sppb_score' if target=='sppb' else 'tugtime'
            y=old.classes(tr[label].to_numpy(),target);ids=tr.participantid.to_numpy()
            ledger=json.loads((STORE/'folds'/f'{tag}.json').read_text())
            for name in ['SpeedRF','BoutRF']:
                for i in [0,9]:
                    a=inner!=i if i==0 else np.ones(len(tr),dtype=bool)
                    model=old.fit(old.make('RF',seed+i),x[name][a],y[a],ids[a],'RF')
                    info=ledger['models'][name]['inner_models'][0] if i==0 else ledger['models'][name]
                    saved=joblib.load(info['path'])
                    np.testing.assert_array_equal(model.predict_proba(xt[name]),saved.predict_proba(xt[name]))
                    rows.append(dict(tag=tag,model=name,stage=i))
            info=ledger['models']['DeepSets']
            for i in [0,9]:
                a=inner!=i if i==0 else np.ones(len(tr),dtype=bool)
                bundle,_,_,_=ds_fit(x['DeepSets'][a],y[a],ids[a],seed+i,info['epochs'])
                path=info['inner_models'][0]['path'] if i==0 else info['path']
                saved=dict(np.load(path))
                for key in bundle:np.testing.assert_array_equal(bundle[key],saved[key])
                rows.append(dict(tag=tag,model='DeepSets',stage=i))
    return rows

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    assert len(m)==2*6*5*9 and len(mt)==2*6*5*4*2
    worst=0.;pair_checks=0
    for target,band,k,seed in itertools.product(['sppb','tug'],['short','long'],[1,3,5],SEEDS):
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

def await_file(path, follow):
    if not follow:
        assert path.exists(),str(path)
        return
    # A separate verification process follows the producer, without occupying a CPU core.
    while not path.exists(): time.sleep(1)
    time.sleep(.1)

def run(follow=False):
    start=time.time();protocol=json.loads((HERE/'protocol.json').read_text())
    assert protocol['code_sha256']==sha(HERE/'study.py')
    for path,digest in protocol['source_hashes'].items():assert sha(path)==digest
    if not follow: assert json.loads((HERE/'main-completion.json').read_text())['tasks']==300
    initialize();results=[]
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    # Serial replay avoids competing with unrelated local applications; RFs are single-threaded.
    for i,task in enumerate(tasks,1):
        target,band,k,fold,seed=task
        await_file(STORE/'folds'/f'{target}-{band}-{k}-f{fold}-s{seed}.json',follow)
        results.append(verify_task(task))
        if i%25==0:
            print('REPLAY',i,'/300',round(time.time()-start),flush=True)
            save(HERE/'verification-progress.json',{'replayed_tasks':i,'total_tasks':300,'seconds':time.time()-start,'state':'replaying'})
    save(HERE/'model-replay.json',results)
    await_file(HERE/'main-completion.json',follow)
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==300
    refits=refit_checks();print('REFITS',len(refits),flush=True)
    await_file(HERE/'summary.json',follow)
    metric_error,pair_checks=verify_metrics()
    save(HERE/'verification.json',{'status':'passed','tasks':300,'replayed_models':sum(r['replays'] for r in results),
        'final_max_error':max(r['final_error'] for r in results),'inner_max_error':max(r['inner_error'] for r in results),
        'tolerance':{'float64':1e-12,'float32_numpy':1e-5,'independent_metrics':1e-10},
        'refits':refits,'metric_rows':540,'matched_metric_rows':pair_checks,'independent_metric_max_error':metric_error,
        'epoch600_count':sum(r['epochs']==600 for r in results),'seconds':time.time()-start,
        'analysis_sha256':sha(HERE/'analyze.py'),'verify_sha256':sha(__file__),
        'checks':['source/model hashes','exact people visits bouts and outer/inner folds','no person overlap',
                  'training-only normalization','all final plus saved inner model predictions','same reference median RF',
                  'inner epoch minimizer','cross-calibration and thresholds','selected exact refits',
                  'independent global metrics','evaluation-only nearest-speed and cohort matching'],
        'limits':'fixed development folds reused; matching descriptive with replacement, not external validity or independence of pairs; inner calibration uses epoch-selected OOF predictions'})
    print('VERIFIED',round(time.time()-start),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true')
    run(parser.parse_args().follow)
