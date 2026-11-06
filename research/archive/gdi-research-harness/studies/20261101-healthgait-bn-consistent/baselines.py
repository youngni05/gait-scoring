"""Same coordinates, same person splits: nested Ridge and RF, saved-model replay."""
import csv
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import manage
from train import partition,groups


def features(x):
    a=x.reshape(len(x),12,124).astype(np.float64)
    return np.concatenate([a.mean(2),a.std(2),np.abs(np.diff(a,axis=2)).mean(2)],axis=1)


def weights(d,mask):
    ids=d['ids'][mask]
    return np.array([1./np.count_nonzero(ids==p) for p in ids])


def person_rows(d,pred,mask):
    return [dict(id=p,y=float(d['y'][d['ids']==p].mean()),prediction=float(pred[d['ids'][mask]==p].mean())) for p in groups(d,mask)]


def fit_replay(d,x,tr,held,test,kind,param,seed,path):
    identity=dict(input_sha256=manage.sha(manage.BUNDLE/'target.npz'),feature_schema='mean_std_mean_abs_diff_36_v1',
                  train_people=groups(d,tr),heldout_people=groups(d,held),outer_test_people=groups(d,test),model=kind,parameter=param,seed=seed)
    assert not set(identity['train_people'])&set(identity['heldout_people'])
    # At final refit heldout==outer test; inner heldout is disjoint from outer test.
    assert not set(identity['train_people'])&set(identity['outer_test_people'])
    modelpath=path.with_suffix('.joblib');metapath=path.with_suffix('.json')
    if metapath.exists():
        meta=manage.load(metapath);assert meta['identity']==identity
        assert meta['model_sha256']==manage.sha(modelpath)
    else:
        estimator=Ridge(alpha=float(param)) if kind=='Ridge' else RandomForestRegressor(n_estimators=128,min_samples_leaf=int(param),random_state=seed,n_jobs=2)
        pipe=Pipeline([('scale',StandardScaler()),('model',estimator)])
        w=weights(d,tr)
        pipe.fit(x[tr],d['y'][tr,0],scale__sample_weight=w,model__sample_weight=w)
        rows=person_rows(d,pipe.predict(x[held]),held)
        modelpath.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(pipe,modelpath)
        meta=dict(identity=identity,model_sha256=manage.sha(modelpath),predictions=rows,person_MAE=float(np.mean([abs(r['prediction']-r['y']) for r in rows])))
        manage.save(metapath,meta)
    loaded=joblib.load(modelpath)
    # Independently check the scaler uses only train-person-weighted input values.
    np.testing.assert_allclose(loaded.named_steps['scale'].mean_,np.average(x[tr],axis=0,weights=weights(d,tr)),atol=1e-12)
    replay=person_rows(d,loaded.predict(x[held]),held)
    assert [r['id'] for r in replay]==[r['id'] for r in meta['predictions']]
    np.testing.assert_allclose([r['prediction'] for r in replay],[r['prediction'] for r in meta['predictions']],atol=1e-12,rtol=0)
    score=float(np.mean([abs(r['prediction']-r['y']) for r in replay]))
    assert abs(score-meta['person_MAE'])<1e-12
    return score,replay,len(replay),dict(path=str(modelpath),sha256=manage.sha(modelpath))


def main():
    with np.load(manage.BUNDLE/'target.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    assert d['x'].shape==(220,2,6,124)
    np.testing.assert_array_equal(d['owner'],np.arange(220))
    x=features(d['x']);assert x.shape==(220,36) and np.isfinite(x).all()
    protocol=manage.load(manage.HERE/'protocol.json');seeds=protocol['target']['seeds']
    root=manage.DATA/'received/baselines/models';rows=[];choices=[];files=[];fits=0
    for fold in range(5):
        train,_,test=partition(d,fold)
        for kind,params,modelseeds in [('Ridge',[1,100],[None]),('RF',[2,5],seeds)]:
            for seed in modelseeds:
                prefix=root/('fold%d/%s-%s'%(fold,kind,seed))
                candidates=[]
                for parameter in params:
                    scores=[];sizes=[]
                    for inner in range(3):
                        tr,va,_=partition(d,fold,inner)
                        score,_,n,file=fit_replay(d,x,tr,va,test,kind,parameter,seed,prefix/('p%s-inner%d'%(parameter,inner)))
                        scores.append(score);sizes.append(n);files.append(file);fits+=1
                    candidates.append(dict(parameter=parameter,inner_MAE=float(np.average(scores,weights=sizes))))
                best=min(candidates,key=lambda v:(v['inner_MAE'],params.index(v['parameter'])))
                _,predictions,_,file=fit_replay(d,x,train,test,test,kind,best['parameter'],seed,prefix/'refit')
                files.append(file);fits+=1
                choices.append(dict(fold=fold,model=kind,seed=seed,candidates=candidates,selected=best))
                rows += [dict(r,fold=fold,model=kind,seed=seed) for r in predictions]
        print(json.dumps(dict(fold=fold,baseline_fits_replayed=fits)),flush=True)
    people=groups(d,np.ones(len(d['ids']),bool));truth=np.array([d['y'][d['ids']==p].mean() for p in people])
    out=manage.HERE/'analysis/baselines';out.mkdir(parents=True,exist_ok=True)
    arrays={};metrics={}
    def metric(pred):
        err=pred-truth
        return dict(MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),low_score_5_8_MAE=float(np.abs(err[truth<=8]).mean()),prediction_range=[float(pred.min()),float(pred.max())])
    for kind,modelseeds in [('Ridge',[None]),('RF',seeds)]:
        values=[];perseed={}
        for seed in modelseeds:
            pred=np.array([next(r['prediction'] for r in rows if r['id']==p and r['model']==kind and r['seed']==seed) for p in people])
            values.append(pred);perseed[str(seed)]=metric(pred)
        arrays[kind]=np.mean(values,axis=0)
        metrics[kind]=dict(ensemble=metric(arrays[kind]),seeds=perseed,seed_prediction_std_mean=float(np.std(values,axis=0).mean()))
    for kind in ('mean','median'):
        pred=np.empty(len(people))
        for fold in range(5):
            tr,_,te=partition(d,fold)
            yy=[d['y'][d['ids']==p].mean() for p in groups(d,tr)]
            for i,p in enumerate(people):
                if p in groups(d,te):pred[i]=float(np.mean(yy) if kind=='mean' else np.median(yy))
        arrays[kind]=pred;metrics[kind]=metric(pred)
    with (out/'person-predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['id','SPPB','Ridge','RF','mean','median'])
        for i,p in enumerate(people):w.writerow([p,truth[i],*[arrays[k][i] for k in ('Ridge','RF','mean','median')]])
    assert fits==140 and len(rows)==176
    manage.save(out/'selected-configs.json',choices)
    result=dict(passed=True,people=44,records=220,features=36,input_sha256=manage.sha(manage.BUNDLE/'target.npz'),saved_and_replayed_fits=fits,final_models=20,metrics=metrics,model_files=files,scope='Same coordinates, temporal summaries; no extra speed/3D/age. Exploratory nested person CV.')
    manage.save(out/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='model_files'}))


if __name__=='__main__':main()
