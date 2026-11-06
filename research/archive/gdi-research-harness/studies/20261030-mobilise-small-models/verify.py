"""Independent NumPy forward passes, split/metric audits, and seeded fit replay."""
from train import HERE,BASE,FEATURES,SEED,fit,sha,save
import numpy as np
import pandas as pd
import torch,json
from threadpoolctl import threadpool_limits

def run():
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert sha(BASE/'features.csv')==protocol['features_sha256']
    assert sha(BASE/'person-folds.csv')==protocol['folds_sha256']
    source_verified=json.loads((BASE/'verification.json').read_text())
    for name,digest in source_verified['artifacts_sha256'].items():assert sha(BASE/name)==digest
    f=pd.read_csv(BASE/'features.csv');p=pd.read_csv(HERE/'predictions.csv');ref=pd.read_csv(BASE/'predictions.csv')
    keys=['participantid','visit_number','draw','fold','target','band','k']
    assert not p.duplicated(keys).any()
    assert p.groupby('participantid').fold.nunique().eq(1).all()
    joined=p.merge(ref[keys+['actual','baseline','SVR']],on=keys,validate='one_to_one',suffixes=('','_ref'))
    assert len(joined)==len(p)==len(ref)
    np.testing.assert_allclose(joined.actual,joined.actual_ref,rtol=0,atol=1e-12)
    models=0;inner=0;retrained=0;maxerror=0.;at_limit=0;parameters=[]
    for (target,band,k,fold),d in p.groupby(['target','band','k','fold']):
        tag=f'{target}-{band}-{k}-f{fold}';label='total_sppb_score' if target=='sppb' else 'tugtime'
        tr=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)&f.fold.ne(fold)]
        te=d.merge(f[['participantid','visit_number','draw','band','k']+FEATURES],on=['participantid','visit_number','draw','band','k'],validate='one_to_one')
        ledger=json.loads((HERE/'folds'/f'{tag}.json').read_text())
        assert set(ledger['train_people'])==set(tr.participantid)
        assert set(ledger['test_people'])==set(d.participantid)
        assert not set(tr.participantid)&set(d.participantid)
        inner_valid=[]
        for split in ledger['inner_people']:
            a=set(split['train']);b=set(split['valid'])
            assert not a&b and a|b==set(tr.participantid)
            inner_valid.extend(b);inner+=1
        assert len(inner_valid)==len(set(inner_valid))==tr.participantid.nunique()
        selected=ledger['selected'];assert selected['inner_person_mae']==min(t['inner_person_mae'] for t in ledger['trials'])
        at_limit+=int(selected['epochs']==120);parameters.append(ledger['parameters'])
        bundle=dict(np.load(HERE/'models'/f'{tag}.npz'))
        w=1/tr.groupby('participantid').participantid.transform('size').to_numpy();w=w/w.sum()
        x=tr[FEATURES].to_numpy();y=tr[label].to_numpy()
        np.testing.assert_allclose(bundle['x_mean'],np.average(x,axis=0,weights=w),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['x_std'],np.sqrt(np.average((x-bundle['x_mean'])**2,axis=0,weights=w)),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['y_mean'],np.average(y,weights=w),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(bundle['y_std'],np.sqrt(np.average((y-bundle['y_mean'])**2,weights=w)),rtol=1e-12,atol=1e-12)
        z=((te[FEATURES].to_numpy()-bundle['x_mean'])/bundle['x_std']).astype('float32')
        for i in range(3):
            z=z@bundle[f'w{i}'].T+bundle[f'b{i}']
            if i<2:z=np.maximum(z,0)
        predicted=z[:,0]*float(bundle['y_std'])+float(bundle['y_mean'])
        maxerror=max(maxerror,float(abs(predicted-te.MLP.to_numpy()).max()))
        np.testing.assert_allclose(predicted,te.MLP,rtol=1e-5,atol=2e-5)
        if fold==0 and k==5 and band=='long':
            tr=tr.sort_values(['draw','participantid','visit_number']).reset_index(drop=True)
            _,again,_=fit(tr,label,selected['width'],selected['epochs'],SEED+int(fold)*10+9)
            for name in bundle:np.testing.assert_array_equal(bundle[name],again[name])
            retrained+=1
        models+=1
    metrics=pd.read_csv(HERE/'metrics.csv');count=0
    for row in metrics.itertuples(index=False):
        d=joined[joined.target.eq(row.target)&joined.band.eq(row.band)&joined.k.eq(row.k)&joined.draw.eq(row.draw)]
        for name,value in [('MLP',row.MLP_mae),('SVR',row.SVR_mae),('baseline',row.baseline_mae)]:
            err=pd.DataFrame({'person':d.participantid,'error':abs(d.actual-d[name])}).groupby('person').error.mean()
            np.testing.assert_allclose(err.mean(),value,rtol=1e-12,atol=1e-12)
            assert len(err)==row.people
        count+=1
    assert models==60 and inner==180 and count==36 and retrained==2
    result={'passed':True,'numpy_forward_models':models,'max_numpy_vs_torch_error':maxerror,'inner_person_splits_checked':inner,'independent_metric_rows':count,'exact_seeded_outer_refits':retrained,'unchanged_reference_artifacts':True,'same_people_visits_and_draws':True,'train_only_x_and_y_scalers':True,'selected_maximum_epoch_count':at_limit,'parameter_counts':sorted(set(parameters)),'limitations':['one initialization seed schedule','bounded width/epoch search','no new video inputs or GPU use'],'artifacts_sha256':{n:sha(HERE/n) for n in ['protocol.json','predictions.csv','metrics.csv','completion.json']},'scripts_sha256':{n:sha(HERE/n) for n in ['train.py','verify.py']}}
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
