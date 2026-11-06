"""Independent predictions, normalization, nested-selection and metric audit."""
import json
import joblib,numpy as np,torch
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from features import OUT,GdiCNN,channels,sha
from compare import load,splits,save

def metrics(y,p):
    e=p-y
    result={'people':len(y),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt((e*e).mean())),'bias':float(e.mean()),'max_error':float(np.abs(e).max())}
    for name,mask in [('TUG_ge20',y>=20),('TUG_lt20',y<20)]:
        result[name]={'people':int(mask.sum()),'MAE':float(np.abs(e[mask]).mean()) if mask.any() else None,'bias':float(e[mask].mean()) if mask.any() else None}
    return result

def main():
    d=load();scalar=dict(np.load(OUT/'scalar/predictions.npz'));scratch=dict(np.load(OUT/'scratch/predictions.npz'))
    for obj in (scalar,scratch):
        for key in ('ids','y','folds'):assert np.array_equal(obj[key],d[key])
    selection=json.loads((OUT/'scalar/selection.json').read_text());assert len(selection)==25
    base=d['base'];inputs={'A_basic_Ridge':base,'A_basic_RF':base,'B_GDI_score_Ridge':np.c_[base,d['gdi_score']],
         'C_GDI_features_Ridge':np.c_[base,d['gdi_features']],'E_random_features_Ridge':np.c_[base,d['random_features']]}
    maximum=0.;files={}
    for fold,tr,te,inner in splits(d):
        assert not set(d['ids'][tr])&set(d['ids'][te])
        np.testing.assert_allclose(scalar['Median'][te],np.median(d['y'][tr]),rtol=0,atol=0)
        for name,x in inputs.items():
            row=next(r for r in selection if r['fold']==fold and r['name']==name)
            assert row['train_people']==d['ids'][tr].tolist() and row['test_people']==d['ids'][te].tolist()
            assert row['setting']==row['grid'][int(np.argmin(row['inner_MAE']))]
            p=OUT/'scalar'/('%s-fold%d.joblib'%(name,fold));assert sha(p)==row['model_sha256'];m=joblib.load(p)
            recomputed=[]
            for setting in row['grid']:
                values=np.full(len(d['y']),np.nan)
                for a,b in inner:
                    candidate=clone(m)
                    candidate[-1].set_params(**({'min_samples_leaf':setting} if name.endswith('_RF') else {'alpha':setting}))
                    candidate.fit(x[a],d['y'][a]);values[b]=candidate.predict(x[b])
                recomputed.append(float(np.abs(values[tr]-d['y'][tr]).mean()))
            np.testing.assert_allclose(recomputed,row['inner_MAE'],rtol=0,atol=1e-5)
            np.testing.assert_allclose(m[0].mean_,x[tr].astype(np.float64).mean(0),atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(m[0].var_,x[tr].astype(np.float64).var(0),atol=1e-10,rtol=1e-10)
            replay=m.predict(x[te]);delta=float(np.abs(replay-scalar[name][te]).max());assert delta<1e-5;maximum=max(maximum,delta);files[str(p.relative_to(OUT))]=sha(p)
        for seed in range(20261031,20261036):
            folder=OUT/'scratch'/('s%d-fold%d'%(seed,fold));epochs=[]
            for i,(a,b) in enumerate(inner):
                ck=torch.load(folder/('inner%d/model.pt'%i),map_location='cpu',weights_only=True)
                assert ck['train_people']==d['ids'][a].tolist() and ck['validation_people']==d['ids'][b].tolist()
                assert not set(ck['train_people'])&set(ck['validation_people']) and not set(ck['train_people'])&set(d['ids'][te])
                h=json.loads((folder/('inner%d/history.json'%i)).read_text());chosen=min(h,key=lambda r:r['validation_MAE'])['epoch'];assert chosen==ck['epoch'];epochs.append(chosen)
            p=folder/'refit/model.pt';ck=torch.load(p,map_location='cpu',weights_only=True)
            assert ck['epoch']==max(1,int(np.median(epochs))) and ck['train_people']==d['ids'][tr].tolist()
            assert ck['input_sha256']==sha(OUT/'prepared.npz') and not ck['validation_people']
            np.testing.assert_allclose(ck['mu'].numpy(),d['x'][tr].mean((0,1,2),keepdims=True),rtol=0,atol=1e-6)
            np.testing.assert_allclose(ck['sd'].numpy(),np.maximum(d['x'][tr].std((0,1,2),keepdims=True),.001),rtol=0,atol=1e-6)
            assert ck['ymean']==float(d['y'][tr].mean()) and ck['ystd']==max(float(d['y'][tr].std()),.1)
            m=GdiCNN(.5).eval();m.load_state_dict(ck['state']);x=(torch.tensor(d['x'][te])-ck['mu'])/ck['sd']
            with torch.inference_mode():replay=m(channels(x).reshape(-1,12,124))[:,0].reshape(-1,2).mean(1).numpy()*ck['ystd']+ck['ymean']
            delta=float(np.abs(replay-scratch['seed%d'%seed][te]).max());assert delta<1e-5;maximum=max(maximum,delta);files[str(p.relative_to(OUT))]=sha(p)
    ensemble=np.stack([scratch['seed%d'%seed] for seed in range(20261031,20261036)]).mean(0)
    np.testing.assert_allclose(ensemble,scratch['D_scratch_ensemble'],rtol=0,atol=1e-10)
    predictions={k:v for k,v in scalar.items() if k not in ('ids','y','folds')}
    predictions.update({k:v for k,v in scratch.items() if k not in ('ids','y','folds')})
    rng=np.random.default_rng(20261031);ix=rng.integers(0,len(d['y']),size=(10000,len(d['y'])))
    differences={}
    for name in predictions:
        if name=='A_basic_Ridge':continue
        delta=np.abs(predictions[name]-d['y'])-np.abs(predictions['A_basic_Ridge']-d['y'])
        differences[name]={'MAE_difference':float(delta.mean()),'conditional_bootstrap95':np.quantile(delta[ix].mean(1),[.025,.975]).tolist()}
    result={'people':len(d['ids']),'metrics':{k:metrics(d['y'],v) for k,v in predictions.items()},'versus_A_basic_Ridge':differences,
            'per_person':[{'person':str(person),'actual_TUG':float(d['y'][i]),**{k:float(v[i]) for k,v in predictions.items()}} for i,person in enumerate(d['ids'])],
            'scope':'Small development feasibility, one manually selected frontal interval per person; one slow participant in original cohort; no independent clinical validation'}
    save(OUT/'comparison.json',result)
    save(OUT/'verification.json',{'passed':True,'scalar_models':25,'scalar_inner_refits':150,'scratch_refits':25,'scratch_inner_selections':75,'replay_tolerance_seconds':1e-5,'max_replay_difference':maximum,
                                'model_hashes':files,'input_sha256':sha(OUT/'prepared.npz'),'result_sha256':sha(OUT/'comparison.json')})
    print(json.dumps(result['metrics'],indent=2),flush=True)
if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
