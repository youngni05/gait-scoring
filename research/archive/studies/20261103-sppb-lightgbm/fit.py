"""Nested person-held-out LightGBM comparisons; outer labels never select models."""
from common import *
import itertools,time,platform
import lightgbm as lgb
import scipy,sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
FITS=[]

def fit_once(X,y,tr,te,params,kind,meta,ids,final=False):
    if kind=='ridge':
        scaler=StandardScaler().fit(X[tr]);model=Ridge(alpha=params['alpha']).fit(scaler.transform(X[tr]),y[tr]);pred=model.predict(scaler.transform(X[te]))
        artifact=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=model.coef_.tolist(),intercept=float(model.intercept_))
    else:
        cfg={**PROTOCOL['LGBM']['fixed'],**params};rounds=cfg.pop('num_boost_round')
        model=lgb.train(cfg,lgb.Dataset(X[tr],label=y[tr],free_raw_data=True),num_boost_round=rounds)
        pred=model.predict(X[te],num_threads=2);artifact=None
    rec=dict(**meta,kind=kind,params=params,train=[ids[i] for i in tr],evaluation=[ids[i] for i in te],prediction=pred.tolist())
    if final:
        stem=f"{meta['cohort']}-{meta['condition']}-fold{meta['outer']}"
        path=BIG/'models'/(stem+('.json' if kind=='ridge' else '.txt'))
        if kind=='ridge':save(path,artifact)
        else:
            model.save_model(str(path));save(path.with_suffix('.trees.json'),model.dump_model())
            rec['gain_importance']=model.feature_importance('gain').tolist()
        rec.update(model=str(path),model_sha256=sha(path))
    FITS.append(rec)
    return pred

def run(name,pp):
    ids=[p['id'] for p in pp];assert len(ids)==len(set(ids))
    y=np.array([p['sppb'] for p in pp],float);folds=np.array([p['fold'] for p in pp]);features=list(pp[0]['features'])
    full=np.array([[p['features'][f] for f in features] for p in pp],float)
    assert not np.isinf(full).any()
    matrices={'speed_LGBM':full[:,[features.index('speed')]],'features_LGBM':full}
    names={'speed_LGBM':['speed'],'features_LGBM':features}
    if name in ['older37','older24']:
        matrices['features_GDI_LGBM']=np.column_stack([full,[p['gdi'] for p in pp]])
        names['features_GDI_LGBM']=features+['predicted_GDI']
    matrices['speed_Ridge']=matrices['speed_LGBM'];names['speed_Ridge']=['speed']
    preds={k:np.full(len(pp),np.nan) for k in [*matrices,'Mean','Median']};selections=[]
    grid=PROTOCOL['LGBM']['grid'];params=[dict(zip(grid,v)) for v in itertools.product(*grid.values())]
    for outer in sorted(set(folds)):
        tr=np.flatnonzero(folds!=outer);te=np.flatnonzero(folds==outer)
        preds['Mean'][te]=y[tr].mean();preds['Median'][te]=np.median(y[tr])
        for cond,X in matrices.items():
            kind='ridge' if cond=='speed_Ridge' else 'lgbm';candidates=[dict(alpha=a) for a in [.1,1,10,100,1000]] if kind=='ridge' else params
            losses=[]
            for ci,cfg in enumerate(candidates):
                errors=[]
                for inner,(tt,vv) in enumerate(KFold(3,shuffle=True,random_state=20261031).split(tr)):
                    pred=fit_once(X,y,tr[tt],tr[vv],cfg,kind,dict(cohort=name,condition=cond,outer=int(outer),inner=inner,candidate=ci),ids)
                    errors.extend(abs(pred-y[tr[vv]]))
                losses.append(float(np.mean(errors)))
            chosen=int(np.argmin(losses));cfg=candidates[chosen]
            preds[cond][te]=fit_once(X,y,tr,te,cfg,kind,dict(cohort=name,condition=cond,outer=int(outer),inner=None,candidate=chosen),ids,True)
            selections.append(dict(condition=cond,outer=int(outer),candidate=chosen,params=cfg,inner_MAE=losses))
        print(name,'fold',outer,'finished',len(FITS),'fits',flush=True)
        save(HERE/'progress.json',dict(cohort=name,outer=int(outer),fits=len(FITS),status='running'))
    assert all(np.isfinite(v).all() for v in preds.values())
    out=dict(ids=ids,y=y.tolist(),folds=folds.tolist(),feature_names=names,predictions={k:v.tolist() for k,v in preds.items()},metrics={k:metrics(y,v) for k,v in preds.items()},contrasts=contrasts(y,preds),selection=selections)
    mask=np.array([p['age']>=65 for p in pp]);out['age65plus']={k:metrics(y[mask],v[mask]) for k,v in preds.items()}
    print(name,{k:round(v['MAE'],4) for k,v in out['metrics'].items()},flush=True)
    return out

def main():
    global PROTOCOL
    PROTOCOL=read(HERE/'protocol.json');receipt=read(HERE/'preparation-receipt.json')
    assert receipt['protocol_sha256']==sha(HERE/'protocol.json') and receipt['cohorts_sha256']==sha(BIG/'cohorts.json')
    (BIG/'models').mkdir(exist_ok=True);cohorts=read(BIG/'cohorts.json');results={};start=time.monotonic()
    for n,pp in cohorts.items():
        results[n]=run(n,pp);save(HERE/'comparison.json',results);save(BIG/'fits.json',FITS)
    save(HERE/'fit-receipt.json',dict(protocol_sha256=sha(HERE/'protocol.json'),cohorts_sha256=sha(BIG/'cohorts.json'),comparison_sha256=sha(HERE/'comparison.json'),fits_sha256=sha(BIG/'fits.json'),fits=len(FITS),final_models=sum('model'in f for f in FITS),elapsed_seconds=time.monotonic()-start,code_sha256=sha(Path(__file__)),runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,lightgbm=lgb.__version__)))
    save(HERE/'progress.json',dict(status='fits_complete_pending_verification',fits=len(FITS)))
if __name__=='__main__':main()
