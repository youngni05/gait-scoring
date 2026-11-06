"""Replay all models, audit paired searches, and independently recalculate errors."""
from train import HERE,REFERENCE,BASE,MLP,SEEDS,BUDGETS,FEATURES,ref,save
import numpy as np
import pandas as pd
import torch,json
from threadpoolctl import threadpool_limits
KEY=['participantid','visit_number','draw','fold','target','band','k']
def run():
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert ref.sha(path)==digest
    prior_verification=json.loads((REFERENCE/'verification.json').read_text());assert prior_verification['passed']
    for name,digest in prior_verification['artifacts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    p=pd.read_csv(HERE/'predictions.csv');prior=pd.read_csv(REFERENCE/'paired-predictions.csv')
    assert len(p)==len(prior)*10 and not p.duplicated(KEY+['seed','budget']).any()
    assert p.groupby('participantid').fold.nunique().eq(1).all()
    expected=prior.set_index(KEY).sort_index()
    for _,g in p.groupby(['seed','budget']):
        actual=g.set_index(KEY).sort_index();assert actual.index.equals(expected.index)
        np.testing.assert_array_equal(actual.actual,expected.actual)
    f=pd.read_csv(BASE/'features.csv');s=pd.read_csv(BASE/'selected-bouts.csv')
    bagkey=ref.BAGKEY
    tables={r:s[s['rank'].eq(r)].set_index(bagkey)[FEATURES] for r in range(1,6)}
    def bags(frame,k):
        ix=pd.MultiIndex.from_frame(frame[bagkey])
        return np.stack([tables[r].loc[ix].to_numpy() for r in range(1,k+1)],axis=1)
    replayed=0;reference_matches=0;paired_searches=0;refitted=0;max_math=0.;max_order=0.
    for (target,band,k,fold),g in p.groupby(['target','band','k','fold']):
        label='total_sppb_score' if target=='sppb' else 'tugtime'
        train=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)&f.fold.ne(fold)].sort_values(['draw','participantid','visit_number']).reset_index(drop=True)
        test=g[g.seed.eq(SEEDS[0])&g.budget.eq(120)].sort_values(['draw','participantid','visit_number'])
        tx=bags(train,int(k));xx=bags(test,int(k))
        counts=train.groupby('participantid').participantid.transform('size').to_numpy();w=1/counts;w=w/w.sum()
        flat=tx.reshape(-1,4);fw=np.repeat(w/int(k),int(k));y=train[label].to_numpy()
        xm=np.average(flat,axis=0,weights=fw);xs=np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw))
        ym=np.average(y,weights=w);ys=np.sqrt(np.average((y-ym)**2,weights=w))
        tag=f'{target}-{band}-{k}-f{fold}';old=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text())
        assert set(old['train_people'])==set(train.participantid)
        assert set(old['test_people'])==set(test.participantid)
        assert not set(train.participantid)&set(test.participantid)
        for seed in SEEDS:
            prefix=f'{tag}-s{seed}';ledger=json.loads((HERE/'folds'/f'{prefix}.json').read_text())
            for name in ['inner_people','train_people','test_people']:assert ledger[name]==old[name]
            assert len(ledger['trials'])==120
            for budget in BUDGETS:
                selected=ledger['selected'][str(budget)]
                assert selected==min((t for t in ledger['trials'] if t['epochs']<=budget),key=lambda t:t['inner_person_mae'])
            assert ledger['selected']['600']['inner_person_mae']<=ledger['selected']['120']['inner_person_mae']
            paired_searches+=1
            if seed==SEEDS[0]:
                assert [t for t in ledger['trials'] if t['epochs']<=120]==old['trials']
                assert ledger['selected']['120']==old['selected']
            for budget in BUDGETS:
                d=g[g.seed.eq(seed)&g.budget.eq(budget)].sort_values(['draw','participantid','visit_number'])
                pd.testing.assert_frame_equal(d[KEY].reset_index(drop=True),test[KEY].reset_index(drop=True))
                b=dict(np.load(HERE/'models'/f'{prefix}-b{budget}.npz'))
                for name,expected_value in [('x_mean',xm),('x_std',xs),('y_mean',ym),('y_std',ys)]:np.testing.assert_allclose(b[name],expected_value,rtol=1e-12,atol=1e-12)
                model=ref.make(ledger['selected'][str(budget)]['width'],seed)
                with torch.no_grad():
                    for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
                        layer.weight.copy_(torch.tensor(b[f'w{i}']));layer.bias.copy_(torch.tensor(b[f'b{i}']))
                prediction=ref.predict(model,b,xx)
                np.testing.assert_array_equal(prediction,d.prediction.to_numpy(dtype=np.float32))
                permutation=ref.predict(model,b,xx[:,::-1]);max_order=max(max_order,float(abs(prediction-permutation).max()))
                np.testing.assert_allclose(prediction,permutation,rtol=1e-5,atol=2e-5)
                z=((xx-b['x_mean'])/b['x_std']).astype('float32')
                h=np.maximum(z@b['w0'].T+b['b0'],0).mean(axis=1);h=np.maximum(h@b['w1'].T+b['b1'],0)
                independent=(h@b['w2'].T+b['b2'])[:,0]*float(b['y_std'])+float(b['y_mean'])
                max_math=max(max_math,float(abs(independent-prediction).max()))
                np.testing.assert_allclose(independent,prediction,rtol=1e-5,atol=2e-5)
                if seed==SEEDS[0] and budget==120:
                    oldb=dict(np.load(REFERENCE/'models'/f'{tag}.npz'))
                    for name in ['w0','b0','w1','b1','w2','b2']:np.testing.assert_array_equal(b[name],oldb[name])
                    oldp=prior[prior.target.eq(target)&prior.band.eq(band)&prior.k.eq(k)&prior.fold.eq(fold)].sort_values(['draw','participantid','visit_number'])
                    np.testing.assert_array_equal(d.prediction.to_numpy(),oldp.DeepSets.to_numpy());reference_matches+=1
                if seed==SEEDS[-1] and budget==600 and k==5 and band=='long' and fold==0:
                    frame=train.copy();frame['bag_id']=np.arange(len(frame));choice=ledger['selected']['600']
                    _,again,_=ref.fit(frame,label,choice['width'],choice['epochs'],seed+int(fold)*10+9,tx,int(k))
                    for name in ['w0','b0','w1','b1','w2','b2']:np.testing.assert_array_equal(b[name],again[name])
                    refitted+=1
                replayed+=1
        if replayed%100==0:print('Replayed',replayed,'models',flush=True)
    # A separate per-person pandas calculation checks the NumPy weighted metrics.
    p=p.merge(prior[KEY+['cohort']],on=KEY,validate='many_to_one')
    m=pd.read_csv(HERE/'seed-metrics.csv').set_index(['target','band','k','draw','seed','budget','group'])
    checked=0
    for key,d in p.groupby(['target','band','k','draw','seed','budget']):
        target=key[0];grouped={'all':d}
        if target=='sppb':grouped.update(low=d[d.actual.le(3)],middle=d[d.actual.gt(3)&d.actual.le(9)],high=d[d.actual.gt(9)])
        else:grouped.update(long=d[d.actual.gt(20)],other=d[d.actual.le(20)])
        grouped.update({'cohort_'+c:g for c,g in d.groupby('cohort')})
        for group,g in grouped.items():
            row=m.loc[key+(group,)];err=g.prediction-g.actual
            person=pd.DataFrame({'id':g.participantid,'ae':abs(err),'e':err,'se':err**2}).groupby('id').mean()
            assert row.people==len(person) and row.visits==len(g)
            for metric,value in [('mae',person.ae.mean()),('bias',person.e.mean()),('rmse',np.sqrt(person.se.mean()))]:np.testing.assert_allclose(row[metric],value,rtol=1e-12,atol=1e-12)
            # Quantile must enclose the .9 weighted empirical CDF, including ties.
            visit_weight=1/g.groupby('participantid').participantid.transform('size').to_numpy();visit_weight/=visit_weight.sum()
            assert visit_weight[abs(err).to_numpy()<row.error_p90-1e-12].sum()<=.9+1e-12
            assert visit_weight[abs(err).to_numpy()<=row.error_p90+1e-12].sum()>=.9-1e-12
            checked+=1
    assert checked==len(m)
    aggregate=pd.read_csv(HERE/'aggregate-metrics.csv');source=m.reset_index()
    for row in aggregate.itertuples(index=False):
        d=source
        for name in ['target','band','k','draw','budget','group']:d=d[d[name].eq(getattr(row,name))]
        assert set(d.seed)==set(SEEDS)
        for name in ['mae','bias','rmse','error_p90']:
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:np.testing.assert_allclose(getattr(row,name+'_'+suffix),value,rtol=1e-12,atol=1e-12)
    paired=pd.read_csv(HERE/'paired-budgets.csv')
    for row in paired.itertuples(index=False):
        k=(row.target,row.band,row.k,row.draw,row.seed)
        short=m.loc[k+(120,row.group)];long=m.loc[k+(600,row.group)]
        np.testing.assert_allclose(row.mae_reduction,short.mae-long.mae,rtol=1e-12,atol=1e-12)
    intervals=json.loads((HERE/'paired-bootstrap.json').read_text())
    lookup=aggregate.set_index(['target','band','k','draw','group','budget'])
    for interval in intervals:
        key=(interval['target'],interval['band'],interval['k'],0,interval['group'])
        current=lookup.loc[key+(600,)]
        comparison=(lookup.loc[key+(120,)].mae_mean if interval['reference']=='paired_120'
                    else current[interval['reference']+'_mae'])
        np.testing.assert_allclose(interval['mae_reduction'],comparison-current.mae_mean,rtol=1e-12,atol=1e-12)
        assert interval['people']==current.people
        assert interval['ci_low']<=interval['mae_reduction']<=interval['ci_high']
    assert len(intervals)==96
    assert replayed==600 and paired_searches==300 and reference_matches==60 and refitted==2
    result={'passed':True,'models_replayed_exact_float32':replayed,'independent_numpy_and_permutation_checks':replayed,'max_numpy_forward_difference':max_math,'max_permutation_difference':max_order,'paired_inner_searches_checked':paired_searches,'original_seed_120_models_weights_and_predictions_exact':reference_matches,'new_seed_600_parameter_refits_exact':refitted,'seed_subgroup_metrics_checked':checked,'aggregate_rows_checked':len(aggregate),'paired_budget_rows_checked':len(paired),'source_hashes_unchanged':True,'same_people_visits_bouts_folds_and_draws':True,'training_only_scalers_and_inner_selection':True,'no_best_seed_selection_or_prediction_ensemble':True,'artifacts_sha256':{name:ref.sha(HERE/name) for name in ['protocol.json','predictions.csv','completion.json','seed-metrics.csv','aggregate-metrics.csv','reference-metrics.csv','paired-budgets.csv','paired-bootstrap.json','selections.csv','summary.json']},'scripts_sha256':{name:ref.sha(HERE/name) for name in ['train.py','analyze.py','verify.py']}}
    result['bootstrap_point_estimates_checked']=len(intervals)
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
