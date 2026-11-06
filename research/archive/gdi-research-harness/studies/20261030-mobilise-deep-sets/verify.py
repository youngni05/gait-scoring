"""Independent input assembly, forward math, invariance and person-split audit."""
from train import HERE,BASE,MLP,FEATURES,BAGKEY,SEED,sha,save,make,fit,predict
import numpy as np
import pandas as pd
import torch,json
from threadpoolctl import threadpool_limits

def run():
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in protocol['source_hashes'].items():assert sha(name)==digest
    for root in [BASE,MLP]:
        verified=json.loads((root/'verification.json').read_text())
        assert verified['passed']
        for name,digest in verified['artifacts_sha256'].items():assert sha(root/name)==digest
    prepared=json.loads((BASE/'preparation.json').read_text())
    assert sha(BASE/'selected-bouts.csv')==prepared['selected_bouts_sha256']
    f=pd.read_csv(BASE/'features.csv');s=pd.read_csv(BASE/'selected-bouts.csv');p=pd.read_csv(HERE/'predictions.csv')
    joined=pd.read_csv(HERE/'paired-predictions.csv');ref=pd.read_csv(BASE/'predictions.csv');mp=pd.read_csv(MLP/'predictions.csv')
    keys=['participantid','visit_number','draw','fold','target','band','k']
    assert not p.duplicated(keys).any() and p.groupby('participantid').fold.nunique().eq(1).all()
    a=p.set_index(keys).sort_index();b=ref.set_index(keys).sort_index();c=mp.set_index(keys).sort_index();j=joined.set_index(keys).sort_index()
    assert a.index.equals(b.index) and a.index.equals(c.index) and a.index.equals(j.index)
    np.testing.assert_array_equal(a.actual,b.actual);np.testing.assert_array_equal(a.actual,c.actual)
    for name,frame in [('DeepSets',a),('SVR',b),('baseline',b),('MLP',c)]:np.testing.assert_array_equal(j[name],frame[name])
    for target,n,visits in [('sppb',2174,7356),('tug',1939,5688)]:
        for _,g in p[p.target.eq(target)].groupby(['band','k','draw']):
            assert g.participantid.nunique()==n and len(g)==visits
    rank_tables={rank:s[s['rank'].eq(rank)].set_index(BAGKEY)[FEATURES] for rank in range(1,6)}
    def independent_bags(frame,k):
        requested=pd.MultiIndex.from_frame(frame[BAGKEY])
        return np.stack([rank_tables[r].loc[requested].to_numpy() for r in range(1,k+1)],axis=1)
    models=0;inner=0;retrained=0;one_bout=0;max_math_error=0.;max_order_error=0.;at_limit=0;parameters=[]
    for (target,band,k,fold),d in p.groupby(['target','band','k','fold']):
        tag=f'{target}-{band}-{k}-f{fold}';label='total_sppb_score' if target=='sppb' else 'tugtime'
        tr=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)&f.fold.ne(fold)].sort_values(['draw','participantid','visit_number']).reset_index(drop=True)
        tx=independent_bags(tr,int(k));xx=independent_bags(d,int(k))
        np.testing.assert_allclose(np.median(tx,axis=1),tr[FEATURES],rtol=1e-12,atol=1e-12)
        ledger=json.loads((HERE/'folds'/f'{tag}.json').read_text())
        prior=json.loads((MLP/'folds'/f'{tag}.json').read_text())
        assert ledger['inner_people']==prior['inner_people']
        assert ledger['train_people']==prior['train_people']==sorted(map(int,tr.participantid.unique()))
        assert ledger['test_people']==prior['test_people']==sorted(map(int,d.participantid.unique()))
        assert not set(tr.participantid)&set(d.participantid)
        for split in ledger['inner_people']:
            assert not set(split['train'])&set(split['valid'])
            assert set(split['train'])|set(split['valid'])==set(tr.participantid)
            inner+=1
        selected=ledger['selected'];assert selected['inner_person_mae']==min(t['inner_person_mae'] for t in ledger['trials'])
        at_limit+=int(selected['epochs']==120);parameters.append(ledger['parameters'])
        bundle=dict(np.load(HERE/'models'/f'{tag}.npz'))
        w=1/tr.groupby('participantid').participantid.transform('size').to_numpy();w=w/w.sum()
        flat=tx.reshape(-1,4);fw=np.repeat(w/int(k),int(k));y=tr[label].to_numpy()
        np.testing.assert_allclose(bundle['x_mean'],np.average(flat,axis=0,weights=fw),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['x_std'],np.sqrt(np.average((flat-bundle['x_mean'])**2,axis=0,weights=fw)),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['y_mean'],np.average(y,weights=w),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['y_std'],np.sqrt(np.average((y-bundle['y_mean'])**2,weights=w)),rtol=1e-12,atol=1e-12)
        z=((xx-bundle['x_mean'])/bundle['x_std']).astype('float32')
        h=np.maximum(z@bundle['w0'].T+bundle['b0'],0).mean(axis=1)
        h=np.maximum(h@bundle['w1'].T+bundle['b1'],0)
        independent=(h@bundle['w2'].T+bundle['b2'])[:,0]*float(bundle['y_std'])+float(bundle['y_mean'])
        max_math_error=max(max_math_error,float(abs(independent-d.DeepSets.to_numpy()).max()))
        np.testing.assert_allclose(independent,d.DeepSets,rtol=1e-5,atol=2e-5)
        model=make(selected['width'],SEED)
        with torch.no_grad():
            for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
                layer.weight.copy_(torch.tensor(bundle[f'w{i}']));layer.bias.copy_(torch.tensor(bundle[f'b{i}']))
        original=predict(model,bundle,xx);permuted=predict(model,bundle,xx[:,::-1])
        max_order_error=max(max_order_error,float(abs(original-permuted).max()))
        # CSV stores the shortest round-trippable float32 decimal; compare in the
        # model's original dtype, rather than treating that decimal as float64.
        np.testing.assert_array_equal(original,d.DeepSets.to_numpy(dtype=np.float32))
        np.testing.assert_allclose(original,permuted,rtol=1e-5,atol=2e-5)
        if k==1:
            old=dict(np.load(MLP/'models'/f'{tag}.npz'))
            for name in bundle:
                if name.startswith(('w','b')):np.testing.assert_array_equal(bundle[name],old[name])
                # C/F array layouts change float64 summation order (~4e-13).
                # Learned float32 weights and all saved predictions stay exact.
                else:np.testing.assert_allclose(bundle[name],old[name],rtol=1e-12,atol=1e-12)
            assert ledger['selected']==prior['selected']
            assert ledger['trials']==prior['trials']
            left=d.set_index(keys).sort_index();right=mp[(mp.target==target)&(mp.band==band)&(mp.k==k)&(mp.fold==fold)].set_index(keys).sort_index()
            np.testing.assert_array_equal(left.DeepSets,right.MLP);one_bout+=1
        if fold==0 and k==5 and band=='long':
            refit_frame=tr.copy();refit_frame['bag_id']=np.arange(len(tr))
            _,again,_=fit(refit_frame,label,selected['width'],selected['epochs'],SEED+int(fold)*10+9,tx,int(k))
            for name in bundle:
                if name.startswith(('w','b')):np.testing.assert_array_equal(bundle[name],again[name])
                else:np.testing.assert_allclose(bundle[name],again[name],rtol=1e-12,atol=1e-12)
            retrained+=1
        models+=1
    metrics=pd.read_csv(HERE/'metrics.csv');count=0
    for row in metrics.itertuples(index=False):
        d=joined[joined.target.eq(row.target)&joined.band.eq(row.band)&joined.k.eq(row.k)&joined.draw.eq(row.draw)]
        means={}
        for name in ['DeepSets','MLP','SVR','baseline']:
            # Explicit per-person means provide an independent metric implementation.
            by_person={}
            for person,actual,pred in zip(d.participantid,d.actual,d[name]):by_person.setdefault(int(person),[]).append(abs(actual-pred))
            means[name]=sum(sum(v)/len(v) for v in by_person.values())/len(by_person)
            np.testing.assert_allclose(means[name],getattr(row,name+'_mae'),rtol=1e-12,atol=1e-12)
            assert len(by_person)==row.people
        for reference in ['MLP','SVR','baseline']:
            delta=means[reference]-means['DeepSets']
            np.testing.assert_allclose(delta,getattr(row,reference+'_improvement'),rtol=1e-10,atol=1e-12)
        count+=1
    assert models==60 and inner==180 and count==36 and retrained==2 and one_bout==20
    result={'passed':True,'independent_bout_assembly':'join each rank separately by participant,visit,draw,band; no positional reshape','numpy_forward_models':models,'torch_replays_and_permutation_checks':models,'max_numpy_vs_torch_error':max_math_error,'max_bout_permutation_error':max_order_error,'inner_person_splits_match_mlp':inner,'metric_rows_recomputed':count,'seeded_outer_refits_with_exact_learned_weights':retrained,'one_bout_models_with_exact_mlp_weights_and_predictions':one_bout,'numeric_checks':'stored model replay exact in original float32 dtype; float64 scaler agreement rtol/atol 1e-12; independent NumPy/permuted forward rtol1e-5 atol2e-5','reference_hashes_unchanged':True,'same_people_visits_selected_bouts_and_draws':True,'train_only_person_weighted_scalers':True,'selected_maximum_epoch_count':at_limit,'parameter_counts':sorted(set(parameters)),'limitations':['single initialization seed schedule','bounded width/epoch search','fixed OOF bootstrap excludes retraining and selection uncertainty','sensor summaries, not video or raw within-bout time series'],'artifacts_sha256':{n:sha(HERE/n) for n in ['protocol.json','predictions.csv','paired-predictions.csv','metrics.csv','summary.json','subgroup-metrics.csv','condition-contrasts.json','completion.json']},'scripts_sha256':{n:sha(HERE/n) for n in ['train.py','summarize.py','verify.py']}}
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
