"""CPU RF on exactly the same nested participant plans; durable, resumable files."""
import os,sys,csv,time
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from common import load,save,sha,data,mask,check_plan,person_rows
import manage

def features(x):
    a=x.reshape(len(x),12,124).astype(np.float64)
    return np.concatenate([a.mean(2),a.std(2),np.abs(np.diff(a,axis=2)).mean(2)],axis=1)
def fit_replay(d,x,tr,held,path,identity):
    meta=path/'fit_complete.json';modelpath=path/'model.joblib';t=mask(d,tr);v=mask(d,held)
    w=np.array([1./np.count_nonzero(d['ids'][t]==p) for p in d['ids'][t]])
    if meta.exists():
        result=load(meta);assert result['identity']==identity and sha(modelpath)==result['model_sha256']
    else:
        model=Pipeline([('scale',StandardScaler()),('model',RandomForestRegressor(n_estimators=128,min_samples_leaf=identity['leaf'],random_state=identity['seed'],n_jobs=2))])
        model.fit(x[t],d['y'][t,0],scale__sample_weight=w,model__sample_weight=w)
        pred=np.zeros(len(x));pred[v]=model.predict(x[v]);rows=person_rows(d,pred,held)
        path.mkdir(parents=True,exist_ok=True);joblib.dump(model,modelpath)
        result=dict(identity=identity,model_sha256=sha(modelpath),rows=rows,MAE=float(np.mean([abs(r['prediction']-r['y']) for r in rows])))
        save(meta,result)
    model=joblib.load(modelpath)
    np.testing.assert_allclose(model.named_steps['scale'].mean_,np.average(x[t],axis=0,weights=w),atol=1e-12)
    pred=np.zeros(len(x));pred[v]=model.predict(x[v]);rows=person_rows(d,pred,held)
    np.testing.assert_allclose([r['prediction'] for r in rows],[r['prediction'] for r in result['rows']],atol=1e-12,rtol=0)
    assert abs(float(np.mean([abs(r['prediction']-r['y']) for r in rows]))-result['MAE'])<1e-12
    return result
def main():
    root=manage.DATA/'rf';d=data(manage.BUNDLE/'target.npz');x=features(d['x']);plan=load(manage.HERE/'plan.json');check_plan(plan,d)
    root.mkdir(exist_ok=True);lock=root/'running.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    rows=[];selections=[];fits=0;files=[]
    try:
        for case in plan['cases']:
            for seed in (20261101,20261102,20261103):
                path=root/('fold%d'%case['fold'])/case['key']/('rf-s%d'%seed);candidates=[]
                for leaf in (2,5):
                    inner=[]
                    for j,split in enumerate(case['inner']):
                        ident=dict(plan_sha256=sha(manage.HERE/'plan.json'),input_sha256=sha(manage.BUNDLE/'target.npz'),train=split['train'],validation=split['validation'],test=case['test'],leaf=leaf,seed=seed)
                        p=path/('leaf%d/inner%d'%(leaf,j));r=fit_replay(d,x,split['train'],split['validation'],p,ident);inner.append(r);fits+=1;files.append(dict(path=str(p/'model.joblib'),sha256=r['model_sha256']))
                    candidates.append(dict(leaf=leaf,score=float(np.average([r['MAE'] for r in inner],weights=[len(s['validation']) for s in case['inner']]))))
                best=min(candidates,key=lambda v:(v['score'],v['leaf']))
                ident=dict(plan_sha256=sha(manage.HERE/'plan.json'),input_sha256=sha(manage.BUNDLE/'target.npz'),train=case['people'],validation=[],test=case['test'],leaf=best['leaf'],seed=seed)
                r=fit_replay(d,x,case['people'],case['test'],path/'refit',ident);fits+=1;files.append(dict(path=str(path/'refit/model.joblib'),sha256=r['model_sha256']))
                selections.append(dict(fold=case['fold'],case=case['key'],seed=seed,candidates=candidates,selected=best))
                rows += [dict(q,fold=case['fold'],case=case['key'],size=case['size'],draw=case['draw'],model='RF',seed=seed) for q in r['rows']]
                save(root/'progress.json',dict(state='running',pid=os.getpid(),fits=fits,expected_fits=1050,case=case['key'],fold=case['fold'],updated_at=manage.now()))
                print('RF verified fits %d/1050'%fits,flush=True)
        assert fits==1050 and len(rows)==1320
        out=manage.HERE/'analysis/rf';out.mkdir(parents=True,exist_ok=True)
        with (out/'person-predictions.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(out/'selection.json',selections)
        save(out/'verification.json',dict(passed=True,input_sha256=sha(manage.BUNDLE/'target.npz'),plan_sha256=sha(manage.HERE/'plan.json'),saved_replayed_fits=1050,final_models=150,people=44,files=files))
        save(root/'progress.json',dict(state='complete',pid=os.getpid(),fits=fits,updated_at=manage.now(),verification_sha256=sha(out/'verification.json')))
    except BaseException as e:
        save(root/'progress.json',dict(state='failed',pid=os.getpid(),fits=fits,error=repr(e),updated_at=manage.now()));raise
    finally:lock.unlink()
if __name__=='__main__':main()
