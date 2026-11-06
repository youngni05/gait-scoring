"""Replay fitted models, audit splits/scalers, independently recalculate metrics."""
import json, time
import train as experiment
from train import HERE, FACTORIAL, SOURCE, SEEDS, LABELS, COMPONENTS, KEY, ref, old, save
import numpy as np
import pandas as pd
import torch
import joblib

def numpy_forward(bundle, x):
    z=((x-bundle['x_mean'])/bundle['x_std']).astype(np.float32)
    h=np.maximum(z@bundle['w0'].T+bundle['b0'],0).mean(axis=1)
    h=np.maximum(h@bundle['w1'].T+bundle['b1'],0)
    return (h@bundle['w2'].T+bundle['b2'])*bundle['y_std']+bundle['y_mean']

def restore(bundle, width):
    model=experiment.MultiDeepSet(width)
    with torch.no_grad():
        for j,index in enumerate([0,2,4]):
            model.layers[index].weight.copy_(torch.tensor(bundle[f'w{j}']))
            model.layers[index].bias.copy_(torch.tensor(bundle[f'b{j}']))
    return model

def check_scalers(train, bundle, arrays, k):
    x=arrays[train.bag_id.to_numpy(),:k]
    # Compute each person's moments first, unlike training's weighted flattened array.
    visits=pd.DataFrame(x.mean(axis=1));visits['person']=train.participantid.to_numpy()
    mean=visits.groupby('person').mean().mean().to_numpy()
    visits=pd.DataFrame(((x-mean)**2).mean(axis=1));visits['person']=train.participantid.to_numpy()
    std=np.sqrt(visits.groupby('person').mean().mean().to_numpy())
    np.testing.assert_allclose(mean,bundle['x_mean'],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(std,bundle['x_std'],atol=1e-10,rtol=1e-10)
    for j,label in enumerate(LABELS):
        d=train[['participantid',label]].dropna()
        mean=d.groupby('participantid')[label].mean().mean()
        var=(d[label]-mean)**2
        std=np.sqrt(var.groupby(d.participantid).mean().mean())
        np.testing.assert_allclose([mean,std],[bundle['y_mean'][j],bundle['y_std'][j]],atol=1e-10,rtol=1e-10)

def baseline_check():
    labels=pd.read_csv(HERE/'labels.csv'); baselines=pd.read_csv(HERE/'baselines.csv')
    source=pd.read_csv(SOURCE,usecols=['participantid','visit.number','total_sppb_score']+LABELS).rename(columns={'visit.number':'visit_number'})
    check=labels.merge(source,on=['participantid','visit_number'],validate='one_to_one',suffixes=('','_source'))
    for label in LABELS+['total_sppb_score']:
        np.testing.assert_array_equal(check[label],check[label+'_source'])
    folds=pd.read_csv(old.BASE/'person-folds.csv')
    assert labels.groupby('participantid').fold.nunique().eq(1).all()
    linked=labels.merge(folds,on='participantid',validate='many_to_one',suffixes=('','_reference'))
    np.testing.assert_array_equal(linked.fold,linked.fold_reference)
    np.testing.assert_array_equal(linked.cohort,linked.cohort_reference)
    def independent_median(frame,label):
        d=frame[frame[label].notna()].copy()
        d['weight']=1/d.groupby('participantid').participantid.transform('size')
        d=d.sort_values(label)
        return float(d.loc[(d.weight.cumsum()>=d.weight.sum()/2),label].iloc[0])
    median_ties=[]
    for fold in range(5):
        tr=labels[labels.fold.ne(fold)].copy();te=labels[labels.fold.eq(fold)].copy()
        tr['remaining']=tr[LABELS[1:]].sum(axis=1,min_count=2)
        actual=te.merge(baselines,on=['participantid','visit_number','fold'],validate='one_to_one')
        for name,label in list(zip(COMPONENTS,LABELS))+[('remaining','remaining'),('total','total_sppb_score')]:
            np.testing.assert_array_equal(actual['constant_'+name],np.repeat(independent_median(tr,label),len(actual)))
            if name in ['gait','total']:continue
            for score in range(5):
                sub=tr[tr[LABELS[0]].eq(score)]
                value=independent_median(sub,label) if sub[label].notna().any() else independent_median(tr,label)
                pred=actual.loc[actual[LABELS[0]].eq(score),'lookup_'+name]
                assert len(pred)==0 or pred.nunique()==1
                if len(pred) and pred.iloc[0]!=value:
                    # At exactly 50% cumulative mass the MAE-optimal median is
                    # an interval. Normalizing weights can select either end.
                    # Independently verify optimality instead of demanding an
                    # arbitrary tie endpoint; do not modify fitted baselines.
                    eligible=sub[sub[label].notna()].copy()
                    weights=1/eligible.groupby('participantid').participantid.transform('size')
                    observed=float(pred.iloc[0])
                    below=weights[eligible[label]<observed].sum()/weights.sum()
                    above=weights[eligible[label]>observed].sum()/weights.sum()
                    assert below<=.5+1e-12 and above<=.5+1e-12
                    assert observed in eligible[label].unique()
                    median_ties.append({'fold':fold,'component':name,'gait_score':score,'prediction':observed,'other_optimal_median':value})
        assert actual.loc[actual[LABELS[0]].isna(),'lookup_total_direct_remaining'].isna().all()
    return {'label_rows_exact':len(labels),'people':int(labels.participantid.nunique()),'lookup_folds_recomputed':5,'non_unique_weighted_medians':median_ties}

def replay(ledgers, early=False):
    counts={'DS_replays':0,'RF_forest_replays':0,'inner_selections_checked':0,'scalers_checked':0}
    maxdiff=0.
    for path in ledgers:
        ledger=json.loads(path.read_text());band,k,fold,seed,kind=[ledger[c] for c in ['band','k','fold','seed','kind']]
        tr,te,prior,splits=experiment.task_data(band,k,fold,seed)
        for field in ['train_people','test_people','inner_people']:assert ledger[field]==prior[field]
        assert set(tr.participantid).isdisjoint(te.participantid)
        observed=pd.read_csv(path.with_suffix('.csv.gz'))
        pd.testing.assert_frame_equal(observed[['participantid','visit_number','draw','fold']],te[['participantid','visit_number','draw','fold']].reset_index(drop=True),check_dtype=False)
        x=experiment.ARRAYS[te.bag_id.to_numpy(),:k]
        if kind=='DS':
            bundle=dict(np.load(HERE/'models'/f'{path.stem}.npz'))
            model=restore(bundle,ledger['width'])
            prediction=experiment.ds_predict(model,bundle,x)
            stored=observed[['DS_'+c for c in COMPONENTS]].to_numpy()
            np.testing.assert_allclose(prediction,stored,rtol=1e-12,atol=1e-12)
            replayed=numpy_forward(bundle,x)
            np.testing.assert_allclose(replayed,stored,rtol=2e-5,atol=2e-5)
            np.testing.assert_allclose(numpy_forward(bundle,x[:,::-1]),stored,rtol=2e-5,atol=2e-5)
            maxdiff=max(maxdiff,float(abs(replayed-stored).max()))
            np.testing.assert_allclose(stored.sum(axis=1),observed.DS_total,atol=1e-12,rtol=1e-12)
            check_scalers(tr,bundle,experiment.ARRAYS,k)
            combined=np.average(ledger['inner_curves'],axis=0,weights=ledger['inner_sizes'])
            np.testing.assert_array_equal(combined,ledger['combined_curve'])
            assert ledger['epochs']==experiment.EPOCHS[int(np.argmin(combined))]
            assert ledger['width']==prior['selected']['600']['width']
            counts['DS_replays']+=1;counts['scalers_checked']+=1;counts['inner_selections_checked']+=1
        else:
            models=joblib.load(HERE/'models'/f'{path.stem}.joblib')
            for c,model in models.items():
                assert model.random_state==seed+fold*10+9 and model.n_estimators==128
                prediction=model.predict(np.median(x,axis=1))
                np.testing.assert_allclose(prediction,observed['RF_'+c],rtol=1e-12,atol=1e-12)
                trials=ledger['trials'][c]
                for trial in trials:
                    np.testing.assert_allclose(np.average(trial['fold_scores'],weights=trial['sizes']),trial['inner_mae'],atol=0,rtol=0)
                assert ledger['selected'][c]==min(trials,key=lambda t:t['inner_mae'])['leaf']==model.min_samples_leaf
                counts['RF_forest_replays']+=1;counts['inner_selections_checked']+=1
            np.testing.assert_allclose(observed[['RF_'+c for c in COMPONENTS]].sum(axis=1),observed.RF_total,atol=1e-12,rtol=1e-12)
    counts['numpy_max_abs_difference']=maxdiff
    return counts

def refit_checks():
    counts={'DS_outer_refits':0,'RF_outer_refits':0,'DS_inner_curves':0,'RF_inner_curves':0}
    for i,(band,k) in enumerate([('short',1),('short',3),('short',5),('long',1),('long',3),('long',5)]):
        fold=i%5;seed=SEEDS[i%5];tag=f'{band}-{k}-f{fold}-s{seed}'
        ledger=json.loads((HERE/'folds'/f'{tag}-DS.json').read_text())
        tr,te,prior,splits=experiment.task_data(band,k,fold,seed)
        _,bundle,_=experiment.ds_fit(tr,ledger['width'],ledger['epochs'],seed+fold*10+9,experiment.ARRAYS,k)
        saved=dict(np.load(HERE/'models'/f'{tag}-DS.npz'))
        for key in saved:np.testing.assert_array_equal(saved[key],bundle[key])
        counts['DS_outer_refits']+=1
    band,k,fold,seed='long',5,0,SEEDS[0]
    tag=f'{band}-{k}-f{fold}-s{seed}';tr,te,prior,splits=experiment.task_data(band,k,fold,seed)
    ledger=json.loads((HERE/'folds'/f'{tag}-DS.json').read_text())
    for i,(a,b) in enumerate(splits):
        _,_,curve=experiment.ds_fit(a,ledger['width'],600,seed+fold*10+i,experiment.ARRAYS,k,b)
        np.testing.assert_array_equal(curve,ledger['inner_curves'][i]);counts['DS_inner_curves']+=1
    ledger=json.loads((HERE/'folds'/f'{tag}-RF.json').read_text())
    saved=joblib.load(HERE/'models'/f'{tag}-RF.joblib')
    for c,label in list(zip(COMPONENTS,LABELS))+[('total_direct','total_sppb_score')]:
        a=tr[tr[label].notna()];model=experiment.rf_model(ledger['selected'][c],seed+fold*10+9)
        model.fit(np.median(experiment.ARRAYS[a.bag_id.to_numpy(),:k],axis=1),a[label],sample_weight=ref.weights(a.participantid.to_numpy()))
        for fresh,stored in zip(model.estimators_,saved[c].estimators_):
            np.testing.assert_array_equal(fresh.tree_.threshold,stored.tree_.threshold)
            np.testing.assert_array_equal(fresh.tree_.value,stored.tree_.value)
        counts['RF_outer_refits']+=1
    # Replay every candidate and inner fold for balance on one long5 RF job.
    for trial in ledger['trials']['balance']:
        for i,(a,b) in enumerate(splits):
            model=experiment.rf_model(trial['leaf'],seed+fold*10+i)
            model.fit(np.median(experiment.ARRAYS[a.bag_id.to_numpy(),:k],axis=1),a.balance_score,sample_weight=ref.weights(a.participantid.to_numpy()))
            pred=model.predict(np.median(experiment.ARRAYS[b.bag_id.to_numpy(),:k],axis=1))
            mae=pd.Series(abs(pred-b.balance_score.to_numpy()),index=b.index).groupby(b.participantid).mean().mean()
            np.testing.assert_allclose(mae,trial['fold_scores'][i],atol=1e-12,rtol=1e-12)
            counts['RF_inner_curves']+=1
    return counts

def metric_checks():
    import analyze
    p=analyze.augment(pd.read_csv(HERE/'predictions.csv.gz'))
    assert len(p)==7356*6*3*5 and not p.duplicated(KEY).any()
    assert set(p.seed)==set(SEEDS)
    assert p.groupby(['band','k','draw','seed']).participantid.nunique().eq(2174).all()
    assert p.groupby(['band','k','draw','seed']).size().eq(7356).all()
    for kind,prefix in [('DS','DS_'),('RF','RF_')]:
        source=pd.concat([pd.read_csv(path) for path in sorted((HERE/'folds').glob('*-'+kind+'.csv.gz'))],ignore_index=True)
        columns=[c for c in source if c.startswith(prefix)]
        joined=p.merge(source[KEY+columns],on=KEY,validate='one_to_one',suffixes=('','_fold'))
        assert len(joined)==len(p)
        for c in columns:np.testing.assert_allclose(joined[c],joined[c+'_fold'],atol=1e-12,rtol=1e-12)
    labels=pd.read_csv(HERE/'labels.csv')
    joined=p.merge(labels[['participantid','visit_number','fold']+LABELS],on=['participantid','visit_number','fold'],validate='many_to_one',suffixes=('','_source'))
    for c in LABELS:np.testing.assert_array_equal(joined[c],joined[c+'_source'])
    baselines=pd.read_csv(HERE/'baselines.csv')
    columns=[c for c in baselines if c.startswith(('constant_','lookup_'))]
    joined=p.merge(baselines,on=['participantid','visit_number','fold'],validate='many_to_one',suffixes=('','_source'))
    for c in columns:np.testing.assert_array_equal(joined[c],joined[c+'_source'])
    previous=pd.read_csv(FACTORIAL/'predictions.csv.gz');previous=previous[previous.target.eq('sppb')]
    joined=p.merge(previous[KEY+['A','B']],on=KEY,validate='one_to_one',suffixes=('','_old'))
    for c in ['A','B']:np.testing.assert_array_equal(joined[c],joined[c+'_old'])
    m=pd.read_csv(HERE/'seed-metrics.csv');count=0
    for key,d in p.groupby(analyze.CELL):
        result=m[(m.band==key[0])&(m.k==key[1])&(m.draw==key[2])&(m.seed==key[3])]
        gs=dict(analyze.groups(d))
        for row in result.itertuples():
            actual,models=analyze.ENDPOINTS[row.endpoint];column=models[row.model]
            frame=gs[row.group];frame=frame[frame[actual].notna()&frame[column].notna()]
            error=(frame[column]-frame[actual]);ids=frame.participantid
            mae=abs(error).groupby(ids).mean().mean();rmse=np.sqrt((error**2).groupby(ids).mean().mean());bias=error.groupby(ids).mean().mean()
            z=pd.DataFrame({'error':abs(error),'person':ids})
            z['w']=1/z.groupby('person').person.transform('size')/z.person.nunique();z=z.sort_values('error')
            p90=z.loc[z.w.cumsum()>=.9,'error'].iloc[0]
            np.testing.assert_allclose([mae,rmse,bias,p90],[row.mae,row.rmse,row.bias,row.p90],atol=1e-10,rtol=1e-10)
            assert len(frame)==row.visits and ids.nunique()==row.people
            count+=1
    paired=pd.read_csv(HERE/'paired-seed-metrics.csv')
    # Primary total-score deltas must reproduce the independently checked metrics.
    primary=paired[paired.endpoint.eq('total')&paired.group.isin(['all','low_sppb'])]
    lookup=m.set_index(analyze.CELL+['endpoint','group','model'])
    for row in primary.itertuples():
        key=(row.band,row.k,row.draw,row.seed,row.endpoint,row.group)
        a=lookup.loc[key+(row.model,)];b=lookup.loc[key+(row.reference,)]
        for name in analyze.METRICS:np.testing.assert_allclose(a[name]-b[name],getattr(row,name+'_change'),atol=1e-12,rtol=1e-12)
    # Recompute the component improvements on exactly matched observations.
    component_checks=0
    component_rows=paired[paired.draw.eq(0)&paired.endpoint.ne('total')]
    for key,d in p[p.draw.eq(0)].groupby(analyze.CELL):
        selected=component_rows[(component_rows.band==key[0])&(component_rows.k==key[1])&(component_rows.seed==key[3])]
        gs=dict(analyze.groups(d))
        for row in selected.itertuples():
            actual,models=analyze.ENDPOINTS[row.endpoint]
            a,b=models[row.model],models[row.reference]
            frame=gs[row.group].dropna(subset=[actual,a,b])
            delta=abs(frame[a]-frame[actual])-abs(frame[b]-frame[actual])
            expected=delta.groupby(frame.participantid).mean().mean()
            np.testing.assert_allclose(expected,row.mae_change,atol=1e-12,rtol=1e-12)
            assert len(frame)==row.visits and frame.participantid.nunique()==row.people
            component_checks+=1
    # Verify every joint pass count directly from the seed-level differences.
    joint=pd.read_csv(HERE/'joint-criteria.csv')
    for row in joint.itertuples():
        subset=paired[(paired.band==row.band)&(paired.k==row.k)&(paired.draw==row.draw)&(paired.endpoint=='total')&(paired.model==row.model)&(paired.reference==row.reference)]
        a=subset[subset.group.eq('all')].set_index('seed').sort_index()
        b=subset[subset.group.eq('low_sppb')].set_index('seed').sort_index()
        assert len(a)==len(b)==5
        good=a.mae_change.le(0)&b.mae_change.lt(0)
        assert int(good.sum())==row.joint_seeds
        assert int((good&b.p90_change.lt(0)).sum())==row.joint_p90_seeds
        np.testing.assert_allclose([a.mae_change.mean(),b.mae_change.mean(),b.p90_change.mean()],[row.overall_change,row.low_change,row.low_p90_change],atol=1e-12,rtol=1e-12)
    # Bootstrap centers must match the independently checked paired estimands.
    aggregate=pd.read_csv(HERE/'paired-aggregate.csv').set_index(['band','k','draw','group','endpoint','model','reference'])
    intervals=json.loads((HERE/'paired-bootstrap.json').read_text())
    for ci in intervals:
        key=(ci['band'],ci['k'],0,ci['group'],ci['endpoint'],ci['model'],ci['reference'])
        np.testing.assert_allclose(ci['mae_change'],aggregate.loc[key,'mae_change_mean'],atol=1e-12,rtol=1e-12)
        assert ci['ci_low']<=ci['ci_high'] and ci['people']==aggregate.loc[key,'people']
    return {'metric_rows_independently_recomputed':count,'primary_total_contrasts_checked':len(primary),'primary_component_matched_deltas_checked':component_checks,'joint_criteria_rows_checked':len(joint),'bootstrap_centers_checked':len(intervals),'original_predictions_unchanged':True,'prediction_rows':len(p)}

def run():
    import sys
    early='--early' in sys.argv;started=time.time()
    models_only='--models' in sys.argv;metrics_only='--metrics' in sys.argv
    experiment.initialize()
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert ref.sha(path)==digest,path
    if metrics_only:
        checks=json.loads((HERE/'model-verification.json').read_text())
        assert checks['status']=='passed'
        manifest=json.loads((HERE/'model-manifest.json').read_text())
        for name,digest in manifest.items():assert ref.sha(HERE/'models'/name)==digest
        checks.update(metric_checks());print('METRICS_PASSED',flush=True)
        completion=json.loads((HERE/'completion.json').read_text())
        assert ref.sha(HERE/'predictions.csv.gz')==completion['predictions_sha256']
        checks.update(scope='all requested fitted models and metrics',metrics_elapsed_seconds=time.time()-started)
        save('verification.json',checks);print(json.dumps(checks,indent=2),flush=True);return
    checks=baseline_check()
    if early:
        paths=[sorted((HERE/'folds').glob('*-'+kind+'.json'))[0] for kind in ['DS','RF']]
    else:
        paths=sorted((HERE/'folds').glob('*.json'));assert len(paths)==300
    checks.update(replay(paths,early))
    print('REPLAYS_PASSED',json.dumps(checks),flush=True)
    if not early:
        checks.update(refit_checks());print('REFITS_PASSED',flush=True)
        if not models_only:
            checks.update(metric_checks());print('METRICS_PASSED',flush=True)
        for path,digest in protocol['source_hashes'].items():assert ref.sha(path)==digest,path
        model_paths=sorted((HERE/'models').glob('*'))
        assert len(model_paths)==300
        save('model-manifest.json',{path.name:ref.sha(path) for path in model_paths})
        checks['model_files_hashed']=len(model_paths)
    checks.update(status='passed',scope='early two-model checks only' if early else ('all fitted models; metric checks pending' if models_only else 'all requested fitted models and metrics'),elapsed_seconds=time.time()-started)
    save('early-verification.json' if early else ('model-verification.json' if models_only else 'verification.json'),checks)
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':run()
