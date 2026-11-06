from models import *
import itertools,time,gc,warnings
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore',message='auto_scale_n_estimators is deprecated')
def main():
    protocol=read(HERE/'protocol.json');receipt=read(HERE/'preparation-receipt.json')
    assert sha(HERE/'protocol.json')==receipt['protocol_sha256'];assert sha(BIG/'cohorts.json')==receipt['cohorts_sha256'];assert sha(HERE/'weights-manifest.json')==receipt['weights_manifest_sha256']
    for w in read(HERE/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    for d in ['contexts','predictions','models','inner']:(BIG/d).mkdir(exist_ok=True)
    rows=read(BIG/'cohorts.json');complete=0;start=time.monotonic()
    for target in ['tug','frailty','non_slowness']:
        rr=[r for r in rows if r['status']=='ok' and r[target] is not None];ids=np.array([r['id'] for r in rr]);y=np.array([r[target] for r in rr]);fold=np.array([r['fold'] for r in rr]);classification=target=='frailty'
        names=list(rr[0]['features']);Xall=np.array([[r['features'][c] for c in names] for r in rr]);assert names[0]=='speed' and np.isfinite(Xall).all()
        for condition in ['speed','features']:
            X=Xall[:,:1] if condition=='speed' else Xall;cols=names[:1] if condition=='speed' else names
            for outer in range(5):
                tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);stem=f'{target}-{condition}-{outer}';context=BIG/'contexts'/(stem+'.npz')
                if not context.exists():np.savez_compressed(context,X_train=X[tr],y_train=y[tr],X_test=X[te],train_ids=ids[tr],test_ids=ids[te],feature_names=np.array(cols))
                for kind in protocol['models']:
                    path=BIG/'predictions'/(stem+'-'+kind+'.json')
                    if path.exists():
                        r=read(path);assert r['context_sha256']==sha(context) and r['protocol_sha256']==sha(HERE/'protocol.json');complete+=1;continue
                    t=time.monotonic();selection=None;inner=[]
                    if kind=='LGBM':
                        grid=protocol['LGBM']['grid'];cfgs=[dict(zip(grid,v)) for v in itertools.product(*grid.values())];losses=[]
                        splitter=StratifiedKFold(3,shuffle=True,random_state=20261031) if classification else KFold(3,shuffle=True,random_state=20261031)
                        splits=list(splitter.split(tr,y[tr]))
                        fixed=protocol['LGBM']['fixed'].copy()
                        if classification:fixed.update(objective='multiclass',num_class=3)
                        for ci,cfg in enumerate(cfgs):
                            errors=[]
                            for ii,(it,iv) in enumerate(splits):
                                cf={**fixed,**cfg};rounds=cf.pop('num_boost_round');m=lgb.train(cf,lgb.Dataset(X[tr[it]],label=transform(y[tr[it]],target)),num_boost_round=rounds)
                                ip=undo(m.predict(X[tr[iv]],num_threads=2),target)
                                errors.extend(-np.log(np.clip(ip[np.arange(len(iv)),y[tr[iv]].astype(int)],1e-15,1)) if classification else abs(ip-y[tr[iv]]))
                                inner.append(dict(candidate=ci,inner=ii,train=ids[tr[it]].tolist(),evaluation=ids[tr[iv]].tolist(),prediction=ip.tolist()))
                            losses.append(float(np.mean(errors)))
                        ci=int(np.argmin(losses));selection=dict(candidate=ci,params=cfgs[ci],losses=losses)
                        cf={**fixed,**cfgs[ci]};rounds=cf.pop('num_boost_round');model=lgb.train(cf,lgb.Dataset(X[tr],label=transform(y[tr],target)),num_boost_round=rounds)
                        pred=undo(model.predict(X[te],num_threads=2),target);mp=BIG/'models'/(stem+'.txt');model.save_model(str(mp));save(BIG/'inner'/(stem+'.json'),inner)
                    else:
                        model=factory(kind,classification);model.fit(X[tr],transform(y[tr],target));pred=model.predict_proba(X[te]) if classification else undo(model.predict(X[te]),target)
                        if classification:assert np.array_equal(model.classes_,[0,1,2])
                    assert np.isfinite(pred).all() and len(pred)==len(te)
                    r=dict(target=target,condition=condition,fold=outer,kind=kind,prediction=pred.tolist(),seconds=time.monotonic()-t,context=str(context),context_sha256=sha(context),protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(Path(__file__)),factory_sha256=sha(HERE/'models.py'),selection=selection)
                    if kind=='LGBM':r.update(model=str(mp),model_sha256=sha(mp))
                    save(path,r);complete+=1;print(stem,kind,round(r['seconds'],2),'s',flush=True);save(HERE/'progress.json',dict(status='running',complete=complete,expected=90,last=path.name,elapsed_seconds=time.monotonic()-start));del model;gc.collect()
    save(HERE/'run-receipt.json',dict(status='predictions_complete_pending_verification',models=complete,protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(Path(__file__)),factory_sha256=sha(HERE/'models.py'),predictions={p.name:sha(p) for p in (BIG/'predictions').glob('*.json')}))
if __name__=='__main__':main()
