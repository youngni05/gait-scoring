"""Audit unchanged data, loss weighting, model replay, and paired evaluation."""
from train import HERE,REFERENCE,BASE,SEEDS,FACTORS,KEY,FEATURES,ref,fit,save
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

def restore(bundle,width,seed):
    model=ref.make(width,seed)
    with torch.no_grad():
        for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
            layer.weight.copy_(torch.tensor(bundle[f'w{i}']));layer.bias.copy_(torch.tensor(bundle[f'b{i}']))
    return model

def independent_refit(x,y,ids,scaler,width,epochs,seed,factor,target):
    # Separate construction of person and tail weights, independent of train.fit.
    counts=pd.Series(ids).map(pd.Series(ids).value_counts()).to_numpy()
    weights=1/counts;weights/=weights.sum()
    mask=y<=3 if target=='sppb' else y>20
    weights[mask]*=factor;weights/=weights.sum()
    xt=torch.tensor((x-scaler['x_mean'])/scaler['x_std'],dtype=torch.float32)
    yt=torch.tensor((y-float(scaler['y_mean']))/float(scaler['y_std']),dtype=torch.float32)
    wt=torch.tensor(weights,dtype=torch.float32)
    model=ref.make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss=torch.nn.functional.smooth_l1_loss(model(xt).flatten(),yt,reduction='none',beta=.5)
        (loss*wt).sum().backward();optimizer.step()
    return model

def run():
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert ref.sha(path)==digest
    verification=json.loads((REFERENCE/'verification.json').read_text());assert verification['passed']
    for name,digest in verification['artifacts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    for name,digest in verification['scripts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    p=pd.read_csv(HERE/'predictions.csv');old=pd.read_csv(REFERENCE/'predictions.csv');old=old[old.budget.eq(600)].drop(columns='budget')
    assert len(p)==2*len(old) and not p.duplicated(KEY+['factor']).any()
    assert p.groupby('participantid').fold.nunique().eq(1).all()
    expected=old.set_index(KEY).sort_index()
    for factor,g in p.groupby('factor'):
        actual=g.set_index(KEY).sort_index();assert actual.index.equals(expected.index)
        np.testing.assert_array_equal(actual.actual,expected.actual)
    f=pd.read_csv(BASE/'features.csv');s=pd.read_csv(BASE/'selected-bouts.csv')
    tables={rank:s[s['rank'].eq(rank)].set_index(ref.BAGKEY)[FEATURES] for rank in range(1,6)}
    def bags(frame,k):
        ix=pd.MultiIndex.from_frame(frame[ref.BAGKEY])
        return np.ascontiguousarray(np.stack([tables[rank].loc[ix].to_numpy() for rank in range(1,k+1)],axis=1))
    replayed=0;baseline_replays=0;controls=0;refits=0;audits=0;max_numpy=0.;max_order=0.
    for (target,band,k,fold),g in p.groupby(['target','band','k','fold']):
        label='total_sppb_score' if target=='sppb' else 'tugtime'
        tr=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)&f.fold.ne(fold)].sort_values(['draw','participantid','visit_number']).reset_index(drop=True)
        test=g[g.seed.eq(SEEDS[0])&g.factor.eq(2)].sort_values(['draw','participantid','visit_number'])
        tx=bags(tr,int(k));xx=bags(test,int(k));y=tr[label].to_numpy()
        w=1/tr.groupby('participantid').participantid.transform('size').to_numpy();w/=w.sum()
        flat=tx.reshape(-1,4);fw=np.repeat(w/k,k)
        xm=np.average(flat,axis=0,weights=fw);xs=np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw))
        ym=np.average(y,weights=w);ys=np.sqrt(np.average((y-ym)**2,weights=w))
        tail=y<=3 if target=='sppb' else y>20
        for seed in SEEDS:
            tag=f'{target}-{band}-{k}-f{fold}-s{seed}'
            ledger=json.loads((HERE/'folds'/f'{tag}.json').read_text());prior=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text())
            assert ledger['reference_ledger_sha256']==ref.sha(REFERENCE/'folds'/f'{tag}.json')
            assert ledger['selected']==prior['selected']['600']
            assert set(ledger['train_people'])==set(tr.participantid)==set(prior['train_people'])
            assert set(ledger['test_people'])==set(test.participantid)==set(prior['test_people'])
            assert not set(tr.participantid)&set(test.participantid)
            choice=ledger['selected'];baseline=dict(np.load(REFERENCE/'models'/f'{tag}-b600.npz'))
            bp=old[old.target.eq(target)&old.band.eq(band)&old.k.eq(k)&old.fold.eq(fold)&old.seed.eq(seed)].sort_values(['draw','participantid','visit_number'])
            bm=restore(baseline,choice['width'],seed)
            np.testing.assert_array_equal(ref.predict(bm,baseline,xx),bp.prediction.to_numpy(dtype=np.float32));baseline_replays+=1
            if ledger['factor1_exact_control']:
                frame=tr.copy();frame['bag_id']=np.arange(len(frame))
                _,control,_=fit(frame,label,choice['width'],choice['epochs'],seed+fold*10+9,tx,k,1)
                for name in baseline:np.testing.assert_allclose(control[name],baseline[name],rtol=0,atol=0)
                controls+=1
            for factor in FACTORS:
                audit=ledger['weight_audits'][str(factor)];mass=w[tail].sum()
                np.testing.assert_allclose(audit['base_tail_mass'],mass,rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(audit['weighted_tail_mass'],factor*mass/(1+(factor-1)*mass),rtol=1e-12,atol=1e-12)
                np.testing.assert_allclose(audit['loss_weight_sum'],1,atol=1e-12,rtol=0)
                assert audit['tail_visits']==int(tail.sum()) and audit['tail_people']==tr.loc[tail,'participantid'].nunique();audits+=1
                d=g[g.seed.eq(seed)&g.factor.eq(factor)].sort_values(['draw','participantid','visit_number'])
                pd.testing.assert_frame_equal(d[KEY[:-1]].reset_index(drop=True),test[KEY[:-1]].reset_index(drop=True))
                b=dict(np.load(HERE/'models'/f'{tag}-w{factor}.npz'))
                for name,value in [('x_mean',xm),('x_std',xs),('y_mean',ym),('y_std',ys)]:
                    np.testing.assert_array_equal(b[name],baseline[name])
                    np.testing.assert_allclose(b[name],value,rtol=1e-12,atol=1e-12)
                model=restore(b,choice['width'],seed);prediction=ref.predict(model,b,xx)
                np.testing.assert_array_equal(prediction,d.prediction.to_numpy(dtype=np.float32))
                permuted=ref.predict(model,b,xx[:,::-1]);max_order=max(max_order,float(abs(prediction-permuted).max()))
                np.testing.assert_allclose(prediction,permuted,rtol=1e-5,atol=2e-5)
                z=((xx-b['x_mean'])/b['x_std']).astype('float32');h=np.maximum(z@b['w0'].T+b['b0'],0).mean(axis=1)
                h=np.maximum(h@b['w1'].T+b['b1'],0);independent=(h@b['w2'].T+b['b2'])[:,0]*float(b['y_std'])+float(b['y_mean'])
                max_numpy=max(max_numpy,float(abs(prediction-independent).max()))
                np.testing.assert_allclose(prediction,independent,rtol=1e-5,atol=2e-5)
                if seed==SEEDS[-1] and band=='long' and k==5 and fold==0:
                    again=independent_refit(tx,y,tr.participantid.to_numpy(),baseline,choice['width'],choice['epochs'],seed+fold*10+9,factor,target)
                    for i,layer in enumerate([again.layers[0],again.layers[2],again.layers[4]]):
                        np.testing.assert_array_equal(layer.weight.detach().numpy(),b[f'w{i}'])
                        np.testing.assert_array_equal(layer.bias.detach().numpy(),b[f'b{i}'])
                    refits+=1
                replayed+=1
        if replayed%100==0:print('Replayed',replayed,'weighted models',flush=True)
    old['factor']=1;p=pd.concat([p,old],ignore_index=True)
    metadata=pd.read_csv(ref.HERE/'paired-predictions.csv')[KEY[:-1]+['cohort']]
    p=p.merge(metadata,on=KEY[:-1],validate='many_to_one')
    m=pd.read_csv(HERE/'seed-metrics.csv').set_index(['target','band','k','draw','seed','factor','group']);checked=0
    for key,d in p.groupby(['target','band','k','draw','seed','factor']):
        tail=d.actual.le(3) if key[0]=='sppb' else d.actual.gt(20)
        grouped={'all':d,'tail':d[tail],'non_tail':d[~tail]}
        if key[0]=='sppb':grouped.update(middle=d[d.actual.gt(3)&d.actual.le(9)],high=d[d.actual.gt(9)])
        grouped.update({'cohort_'+cohort:g for cohort,g in d.groupby('cohort')})
        for group,g in grouped.items():
            row=m.loc[key+(group,)];err=g.prediction-g.actual
            person=pd.DataFrame({'id':g.participantid,'ae':abs(err),'e':err,'se':err**2}).groupby('id').mean()
            assert row.people==len(person) and row.visits==len(g)
            for name,value in [('mae',person.ae.mean()),('bias',person.e.mean()),('rmse',np.sqrt(person.se.mean()))]:np.testing.assert_allclose(row[name],value,rtol=1e-12,atol=1e-12)
            w=1/g.groupby('participantid').participantid.transform('size').to_numpy();w/=w.sum()
            assert w[abs(err).to_numpy()<row.error_p90-1e-12].sum()<=.9+1e-12
            assert w[abs(err).to_numpy()<=row.error_p90+1e-12].sum()>=.9-1e-12
            checked+=1
    assert checked==len(m)
    paired=pd.read_csv(HERE/'paired-metrics.csv')
    for row in paired.itertuples(index=False):
        key=(row.target,row.band,row.k,row.draw,row.seed)
        a=m.loc[key+(1,row.group)];b=m.loc[key+(row.factor,row.group)]
        for name in ['mae','rmse','error_p90']:np.testing.assert_allclose(getattr(row,name+'_reduction'),a[name]-b[name],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(row.absolute_bias_reduction,abs(a.bias)-abs(b.bias),atol=1e-12,rtol=1e-12)
    agg=pd.read_csv(HERE/'aggregate-metrics.csv');source=m.reset_index()
    for row in agg.itertuples(index=False):
        d=source
        for name in ['target','band','k','draw','factor','group']:d=d[d[name].eq(getattr(row,name))]
        assert set(d.seed)==set(SEEDS)
        for name in ['mae','bias','rmse','error_p90']:
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:np.testing.assert_allclose(getattr(row,name+'_'+suffix),value,atol=1e-12,rtol=1e-12)
    joint=pd.read_csv(HERE/'joint-tradeoffs.csv')
    for row in joint.itertuples(index=False):
        key=(row.target,row.band,row.k,row.draw,row.seed)
        da=m.loc[key+(1,'all')].mae-m.loc[key+(row.factor,'all')].mae
        dt=m.loc[key+(1,'tail')].mae-m.loc[key+(row.factor,'tail')].mae
        np.testing.assert_allclose([row.mae_reduction_all,row.mae_reduction_tail],[da,dt],atol=1e-12,rtol=1e-12)
        assert row.tail_better_global_not_worse==bool(da>=0 and dt>0)
    intervals=json.loads((HERE/'paired-bootstrap.json').read_text());lookup=agg.set_index(['target','band','k','draw','factor','group'])
    for interval in intervals:
        key=(interval['target'],interval['band'],interval['k'],0)
        a=lookup.loc[key+(1,interval['group'])];b=lookup.loc[key+(interval['factor'],interval['group'])]
        np.testing.assert_allclose(interval['mae_reduction'],a.mae_mean-b.mae_mean,atol=1e-12,rtol=1e-12)
        assert interval['people']==a.people and interval['ci_low']<=interval['mae_reduction']<=interval['ci_high']
    assert replayed==600 and baseline_replays==300 and controls==2 and refits==4 and audits==600 and len(intervals)==72
    result={'passed':True,'weighted_model_replays':replayed,'reference_model_replays':baseline_replays,'independent_numpy_and_permutation_checks':replayed,'max_numpy_difference':max_numpy,'max_order_difference':max_order,'independent_weighted_training_refits_exact':refits,'factor1_reference_controls_exact':controls,'loss_weight_audits':audits,'seed_subgroup_metrics_checked':checked,'paired_metrics_checked':len(paired),'aggregate_rows_checked':len(agg),'joint_tradeoffs_checked':len(joint),'bootstrap_point_estimates_checked':len(intervals),'same_people_bouts_folds_seeds_scalers_widths_epochs':True,'source_hashes_unchanged':True,'evaluation_weights_unboosted':True,'artifacts_sha256':{name:ref.sha(HERE/name) for name in ['protocol.json','completion.json','predictions.csv','seed-metrics.csv','paired-metrics.csv','aggregate-metrics.csv','joint-tradeoffs.csv','condition-tradeoffs.csv','paired-bootstrap.json','summary.json']},'scripts_sha256':{name:ref.sha(HERE/name) for name in ['train.py','analyze.py','verify.py']}}
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
