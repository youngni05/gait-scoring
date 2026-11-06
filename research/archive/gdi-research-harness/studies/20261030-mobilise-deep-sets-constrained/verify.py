"""Check every constrained choice, replay predictions, and recalculate evaluation."""
from train import HERE,WEIGHTED,REFERENCE,BASE,SEEDS,FACTORS,EPOCHS,KEY,FEATURES,ref,weighted,curve_fit,save
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
    counts=pd.Series(ids).map(pd.Series(ids).value_counts()).to_numpy();w=1/counts;w/=w.sum()
    mask=y<=3 if target=='sppb' else y>20;w[mask]*=factor;w/=w.sum()
    xt=torch.tensor((x-scaler['x_mean'])/scaler['x_std'],dtype=torch.float32)
    yt=torch.tensor((y-float(scaler['y_mean']))/float(scaler['y_std']),dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    model=ref.make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    for _ in range(epochs):
        optimizer.zero_grad();loss=torch.nn.functional.smooth_l1_loss(model(xt).flatten(),yt,reduction='none',beta=.5)
        (loss*wt).sum().backward();optimizer.step()
    return model

def run():
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items():assert ref.sha(path)==digest
    previous=json.loads((REFERENCE/'verification.json').read_text());assert previous['passed']
    for name,digest in previous['artifacts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    for name,digest in previous['scripts_sha256'].items():assert ref.sha(REFERENCE/name)==digest
    p=pd.read_csv(HERE/'predictions.csv');old=pd.read_csv(REFERENCE/'predictions.csv');old=old[old.budget.eq(600)].drop(columns='budget')
    expected=old.set_index(KEY).sort_index();actual=p.set_index(KEY).sort_index()
    assert actual.index.equals(expected.index) and not p.duplicated(KEY).any()
    np.testing.assert_array_equal(actual.actual,expected.actual)
    assert p.groupby('participantid').fold.nunique().eq(1).all()
    f=pd.read_csv(BASE/'features.csv');s=pd.read_csv(BASE/'selected-bouts.csv')
    tables={rank:s[s['rank'].eq(rank)].set_index(ref.BAGKEY)[FEATURES] for rank in range(1,6)}
    def bags(frame,k):
        ix=pd.MultiIndex.from_frame(frame[ref.BAGKEY])
        return np.ascontiguousarray(np.stack([tables[rank].loc[ix].to_numpy() for rank in range(1,k+1)],axis=1))
    replayed=0;fallbacks=0;curves_replayed=0;refitted=set();max_math=0.;max_order=0.;selected_rows=[]
    for (target,band,k,fold),g in p.groupby(['target','band','k','fold']):
        label='total_sppb_score' if target=='sppb' else 'tugtime'
        tr=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)&f.fold.ne(fold)].sort_values(['draw','participantid','visit_number']).reset_index(drop=True)
        te=g[g.seed.eq(SEEDS[0])].sort_values(['draw','participantid','visit_number']);tx=bags(tr,int(k));xx=bags(te,int(k));y=tr[label].to_numpy()
        w=1/tr.groupby('participantid').participantid.transform('size').to_numpy();w/=w.sum()
        flat=tx.reshape(-1,4);fw=np.repeat(w/k,k)
        xm=np.average(flat,axis=0,weights=fw);xs=np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw));ym=np.average(y,weights=w);ys=np.sqrt(np.average((y-ym)**2,weights=w))
        for seed in SEEDS:
            tag=f'{target}-{band}-{k}-f{fold}-s{seed}';ledger=json.loads((HERE/'folds'/f'{tag}.json').read_text());prior=json.loads((REFERENCE/'folds'/f'{tag}.json').read_text())
            assert ledger['reference_ledger_sha256']==ref.sha(REFERENCE/'folds'/f'{tag}.json')
            assert ledger['inner_people']==prior['inner_people']
            assert set(ledger['train_people'])==set(tr.participantid)==set(prior['train_people'])
            assert set(ledger['test_people'])==set(te.participantid)==set(prior['test_people'])
            assert not set(tr.participantid)&set(te.participantid)
            sizes=[];tails=[];validation_people=[]
            for ids in ledger['inner_people']:
                assert not set(ids['train'])&set(ids['valid'])
                assert set(ids['train'])|set(ids['valid'])==set(tr.participantid)
                va=tr[tr.participantid.isin(ids['valid'])];mask=va[label].le(3) if target=='sppb' else va[label].gt(20)
                sizes.append(va.participantid.nunique());tails.append(va.loc[mask,'participantid'].nunique());validation_people.extend(ids['valid'])
            assert len(validation_people)==len(set(validation_people)) and set(validation_people)==set(tr.participantid)
            assert sizes==ledger['inner_sizes'] and tails==ledger['inner_tail_sizes']
            base=ledger['baseline'];control=np.array(ledger['baseline_inner_metrics']);oldchoice=prior['selected']['600']
            assert ledger['width']==oldchoice['width'] and base['epochs']==oldchoice['epochs']
            assert base['inner_mae']==oldchoice['inner_person_mae']==float(np.average(control[:,0],weights=sizes))
            assert base['inner_tail_mae']==float(np.average(control[:,1],weights=tails))
            reconstructed=[base]
            for factor in FACTORS:
                c=np.array(ledger['inner_curves'][str(factor)]);assert c.shape==(3,60,2) and np.isfinite(c).all() and (c>=0).all()
                all_mae=np.average(c[:,:,0],axis=0,weights=sizes);tail_mae=np.average(c[:,:,1],axis=0,weights=tails)
                reconstructed.extend({'factor':factor,'epochs':epoch,'inner_mae':float(a),'inner_tail_mae':float(b)} for epoch,a,b in zip(EPOCHS,all_mae,tail_mae))
            assert reconstructed==ledger['trials'] and len(reconstructed)==181
            feasible=[x for x in reconstructed if x['inner_mae']<=base['inner_mae']]
            chosen=sorted(feasible,key=lambda x:(x['inner_tail_mae'],x['inner_mae'],x['factor'],x['epochs']))[0]
            assert chosen==ledger['selected'] and len(feasible)==ledger['feasible_candidates']
            assert chosen['inner_mae']<=base['inner_mae'] and chosen['inner_tail_mae']<=base['inner_tail_mae']
            selected_rows.append((target,band,k,fold,seed,ledger['width'],chosen['factor'],chosen['epochs'],chosen['inner_mae'],chosen['inner_tail_mae']))
            d=g[g.seed.eq(seed)].sort_values(['draw','participantid','visit_number'])
            pd.testing.assert_frame_equal(d[KEY[:-1]].reset_index(drop=True),te[KEY[:-1]].reset_index(drop=True))
            b=dict(np.load(HERE/'models'/f'{tag}.npz'));baseline=dict(np.load(REFERENCE/'models'/f'{tag}-b600.npz'))
            for name,value in [('x_mean',xm),('x_std',xs),('y_mean',ym),('y_std',ys)]:
                np.testing.assert_array_equal(b[name],baseline[name]);np.testing.assert_allclose(b[name],value,atol=1e-12,rtol=1e-12)
            model=restore(b,ledger['width'],seed);prediction=ref.predict(model,b,xx)
            np.testing.assert_array_equal(prediction,d.prediction.to_numpy(dtype=np.float32))
            bp=old[old.target.eq(target)&old.band.eq(band)&old.k.eq(k)&old.fold.eq(fold)&old.seed.eq(seed)].sort_values(['draw','participantid','visit_number'])
            bm=restore(baseline,ledger['width'],seed)
            np.testing.assert_array_equal(ref.predict(bm,baseline,xx),bp.prediction.to_numpy(dtype=np.float32))
            if chosen['factor']==1:
                assert ref.sha(HERE/'models'/f'{tag}.npz')==ref.sha(REFERENCE/'models'/f'{tag}-b600.npz')
                np.testing.assert_array_equal(d.prediction.to_numpy(),bp.prediction.to_numpy());fallbacks+=1
            elif target not in refitted:
                again=independent_refit(tx,y,tr.participantid.to_numpy(),baseline,ledger['width'],chosen['epochs'],seed+fold*10+9,chosen['factor'],target)
                for i,layer in enumerate([again.layers[0],again.layers[2],again.layers[4]]):
                    np.testing.assert_array_equal(layer.weight.detach().numpy(),b[f'w{i}']);np.testing.assert_array_equal(layer.bias.detach().numpy(),b[f'b{i}'])
                refitted.add(target)
            perm=ref.predict(model,b,xx[:,::-1]);max_order=max(max_order,float(abs(prediction-perm).max()));np.testing.assert_allclose(prediction,perm,atol=2e-5,rtol=1e-5)
            z=((xx-b['x_mean'])/b['x_std']).astype('float32');h=np.maximum(z@b['w0'].T+b['b0'],0).mean(axis=1);h=np.maximum(h@b['w1'].T+b['b1'],0)
            independent=(h@b['w2'].T+b['b2'])[:,0]*float(b['y_std'])+float(b['y_mean']);max_math=max(max_math,float(abs(prediction-independent).max()));np.testing.assert_allclose(prediction,independent,atol=2e-5,rtol=1e-5)
            if band=='long' and k==5 and fold==0 and seed==SEEDS[0]:
                frame=tr.copy();frame['bag_id']=np.arange(len(frame));ids=ledger['inner_people'][0]
                a=frame[frame.participantid.isin(ids['train'])];v=frame[frame.participantid.isin(ids['valid'])]
                for factor in [1.25,2.0]:
                    replay=curve_fit(a,v,label,ledger['width'],600,seed,tx,k,factor,EPOCHS)
                    np.testing.assert_array_equal(replay,ledger['inner_curves'][str(factor)][0]);curves_replayed+=1
            replayed+=1
        if replayed%50==0:print('Replayed and audited',replayed,'selected models',flush=True)
    # Independent pandas per-person evaluation, including unchanged fallback predictions.
    paired_predictions=pd.read_csv(HERE/'paired-predictions.csv')
    expected_pair=p.merge(old[KEY+['prediction']],on=KEY,validate='one_to_one',suffixes=('_selected','_baseline'))
    check=paired_predictions.merge(expected_pair,on=KEY,validate='one_to_one',suffixes=('','_check'))
    np.testing.assert_array_equal(check.constrained,check.prediction_selected);np.testing.assert_array_equal(check.baseline,check.prediction_baseline)
    m=pd.read_csv(HERE/'seed-metrics.csv').set_index(['target','band','k','draw','seed','group','model']);checked=0
    for key,d in paired_predictions.groupby(['target','band','k','draw','seed']):
        mask=d.actual.le(3) if key[0]=='sppb' else d.actual.gt(20)
        groups={'all':d,'tail':d[mask],'non_tail':d[~mask]}
        if key[0]=='sppb':groups.update(middle=d[d.actual.gt(3)&d.actual.le(9)],high=d[d.actual.gt(9)])
        groups.update({'cohort_'+c:g for c,g in d.groupby('cohort')})
        for group,g in groups.items():
            w=1/g.groupby('participantid').participantid.transform('size').to_numpy();w/=w.sum()
            for model in ['baseline','constrained']:
                row=m.loc[key+(group,model)];err=g[model]-g.actual
                person=pd.DataFrame({'id':g.participantid,'ae':abs(err),'e':err,'se':err**2}).groupby('id').mean()
                assert row.people==len(person) and row.visits==len(g)
                for name,value in [('mae',person.ae.mean()),('bias',person.e.mean()),('rmse',np.sqrt(person.se.mean()))]:np.testing.assert_allclose(row[name],value,atol=1e-12,rtol=1e-12)
                assert w[abs(err).to_numpy()<row.error_p90-1e-12].sum()<=.9+1e-12
                assert w[abs(err).to_numpy()<=row.error_p90+1e-12].sum()>=.9-1e-12
                checked+=1
    assert checked==len(m)
    paired=pd.read_csv(HERE/'paired-metrics.csv')
    for row in paired.itertuples(index=False):
        key=(row.target,row.band,row.k,row.draw,row.seed,row.group)
        a=m.loc[key+('baseline',)];b=m.loc[key+('constrained',)]
        for name in ['mae','rmse','error_p90']:np.testing.assert_allclose(getattr(row,name+'_reduction'),a[name]-b[name],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(row.absolute_bias_reduction,abs(a.bias)-abs(b.bias),atol=1e-12,rtol=1e-12)
    agg=pd.read_csv(HERE/'aggregate-metrics.csv')
    for row in agg.itertuples(index=False):
        d=paired
        for name in ['target','band','k','draw','group']:d=d[d[name].eq(getattr(row,name))]
        assert set(d.seed)==set(SEEDS)
        for name in ['mae','bias','rmse','error_p90']:
            np.testing.assert_allclose(getattr(row,name+'_reference_mean'),d[name+'_reference'].mean(),atol=1e-12,rtol=1e-12)
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:np.testing.assert_allclose(getattr(row,name+'_'+suffix),value,atol=1e-12,rtol=1e-12)
        for name in ['mae','rmse','error_p90','absolute_bias']:
            delta=d[name+'_reduction'];np.testing.assert_allclose(getattr(row,name+'_reduction_mean'),delta.mean(),atol=1e-12,rtol=1e-12)
            for suffix,value in [('better',delta.gt(0).sum()),('same',delta.eq(0).sum()),('worse',delta.lt(0).sum())]:assert getattr(row,name+'_'+suffix+'_seeds')==value
    joint=pd.read_csv(HERE/'joint-tradeoffs.csv')
    for row in joint.itertuples(index=False):
        key=(row.target,row.band,row.k,row.draw,row.seed)
        da=m.loc[key+('all','baseline')].mae-m.loc[key+('all','constrained')].mae
        dt=m.loc[key+('tail','baseline')].mae-m.loc[key+('tail','constrained')].mae
        np.testing.assert_allclose([row.mae_reduction_all,row.mae_reduction_tail],[da,dt],atol=1e-12,rtol=1e-12)
        assert row.tail_better_global_not_worse==bool(da>=0 and dt>0)
    intervals=json.loads((HERE/'paired-bootstrap.json').read_text());lookup=agg.set_index(['target','band','k','draw','group'])
    for interval in intervals:
        row=lookup.loc[(interval['target'],interval['band'],interval['k'],0,interval['group'])]
        np.testing.assert_allclose(interval['mae_reduction'],row.mae_reduction_mean,atol=1e-12,rtol=1e-12)
        assert interval['people']==row.people and interval['ci_low']<=interval['mae_reduction']<=interval['ci_high']
    assert replayed==300 and curves_replayed==4 and len(intervals)==36
    selection=pd.read_csv(HERE/'selections.csv');assert len(selection)==300 and selection.fallback.sum()==fallbacks and selection.inner_global_reduction.ge(0).all() and selection.inner_tail_reduction.ge(0).all()
    indexed=selection.set_index(['target','band','k','fold','seed'])
    for target,band,k,fold,seed,width,factor,epochs,mae,tail in selected_rows:
        row=indexed.loc[(target,band,k,fold,seed)]
        assert (row.width,row.factor,row.epochs)==(width,factor,epochs)
        np.testing.assert_allclose([row.inner_mae,row.inner_tail_mae],[mae,tail],atol=1e-12,rtol=1e-12)
    condition=pd.read_csv(HERE/'condition-tradeoffs.csv')
    for row in condition.itertuples(index=False):
        d=joint[joint.draw.eq(0)&joint.target.eq(row.target)&joint.band.eq(row.band)&joint.k.eq(row.k)]
        np.testing.assert_allclose([row.global_mae_change,row.tail_mae_change],[-d.mae_reduction_all.mean(),-d.mae_reduction_tail.mean()],atol=1e-12,rtol=1e-12)
        assert row.both_objectives_met_seeds==d.tail_better_global_not_worse.sum()
        assert row.global_worse_seeds==d.mae_reduction_all.lt(0).sum() and row.tail_better_seeds==d.mae_reduction_tail.gt(0).sum()
    result={'passed':True,'selected_and_reference_models_replayed':replayed,'constrained_searches_independently_reaggregated':replayed,'all_inner_overall_and_tail_constraints_pass':True,'all_person_splits_and_source_hashes_unchanged':True,'fallback_models_and_predictions_exact':fallbacks,'independent_selected_training_refits_exact':sorted(refitted),'full_inner_curves_replayed_exact':curves_replayed,'metric_rows_checked':checked,'aggregate_rows_checked':len(agg),'joint_tradeoffs_checked':len(joint),'bootstrap_point_estimates_checked':len(intervals),'max_numpy_forward_difference':max_math,'max_permutation_difference':max_order,'artifacts_sha256':{name:ref.sha(HERE/name) for name in ['protocol.json','completion.json','predictions.csv','paired-predictions.csv','seed-metrics.csv','paired-metrics.csv','aggregate-metrics.csv','joint-tradeoffs.csv','condition-tradeoffs.csv','selections.csv','paired-bootstrap.json','summary.json']},'scripts_sha256':{name:ref.sha(HERE/name) for name in ['train.py','analyze.py','verify.py']},'metric_helper_sha256':ref.sha(WEIGHTED/'analyze.py')}
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
