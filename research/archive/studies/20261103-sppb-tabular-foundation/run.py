"""Fixed pretrained comparisons plus a nested LightGBM large-cohort baseline."""
from foundation import *
import argparse, itertools, time, gc, warnings
import lightgbm as lgb
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore',message='auto_scale_n_estimators is deprecated')
def main():
    arg=argparse.ArgumentParser();arg.add_argument('scope',choices=['small','large']);args=arg.parse_args()
    protocol=read(HERE/'protocol.json');receipt=read(HERE/'preparation-receipt.json')
    assert sha(HERE/'protocol.json')==receipt['protocol_sha256']
    assert sha(BIG/'cohorts.json')==receipt['cohorts_sha256']
    for p,h in protocol['sources'].items():assert sha(p)==h,p
    for w in read(HERE/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    for d in ['contexts','predictions','models','inner']: (BIG/d).mkdir(exist_ok=True)
    cohorts=read(BIG/'cohorts.json');records=[];start=time.monotonic()
    for name,pp in cohorts.items():
        if (name=='mobilised') != (args.scope=='large'):continue
        y=np.array([p['sppb'] for p in pp],float);ids=np.array([p['id'] for p in pp]);fold=np.array([p['fold'] for p in pp]);assert len(set(ids))==len(ids)
        for condition,(X,names) in matrices(name,pp).items():
            assert not np.isinf(X).any()
            for outer in sorted(set(fold)):
                tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);stem=f'{name}-{condition}-{outer}'
                context=BIG/'contexts'/(stem+'.npz')
                if not context.exists():np.savez_compressed(context,X_train=X[tr],y_train=y[tr],X_test=X[te],train_ids=ids[tr],test_ids=ids[te],feature_names=np.array(names))
                for kind in ['TabICLv2','TabPFNv2']+(['LGBM'] if name=='mobilised' else []):
                    path=BIG/'predictions'/(stem+'-'+kind+'.json')
                    if path.exists():
                        old=read(path);assert old['context_sha256']==sha(context) and old['protocol_sha256']==sha(HERE/'protocol.json');records.append(old);continue
                    t=time.monotonic();selection=None;inner=[]
                    if kind=='LGBM':
                        grid=protocol['LGBM']['grid'];cfgs=[dict(zip(grid,v)) for v in itertools.product(*grid.values())];losses=[]
                        for ci,cfg in enumerate(cfgs):
                            errs=[]
                            for ii,(it,iv) in enumerate(KFold(3,shuffle=True,random_state=20261031).split(tr)):
                                cf={**protocol['LGBM']['fixed'],**cfg};rounds=cf.pop('num_boost_round')
                                m=lgb.train(cf,lgb.Dataset(X[tr[it]],label=y[tr[it]]),num_boost_round=rounds);ip=m.predict(X[tr[iv]],num_threads=2);errs.extend(abs(ip-y[tr[iv]]))
                                inner.append(dict(candidate=ci,inner=ii,train=ids[tr[it]].tolist(),evaluation=ids[tr[iv]].tolist(),prediction=ip.tolist()))
                            losses.append(float(np.mean(errs)))
                        ci=int(np.argmin(losses));selection=dict(candidate=ci,params=cfgs[ci],losses=losses)
                        cf={**protocol['LGBM']['fixed'],**cfgs[ci]};rounds=cf.pop('num_boost_round');model=lgb.train(cf,lgb.Dataset(X[tr],label=y[tr]),num_boost_round=rounds)
                        pred=model.predict(X[te],num_threads=2);modelpath=BIG/'models'/(stem+'.txt');model.save_model(str(modelpath));save(modelpath.with_suffix('.trees.json'),model.dump_model());save(BIG/'inner'/(stem+'.json'),inner)
                    else:
                        model=factory(kind);model.fit(X[tr],y[tr]);pred=model.predict(X[te])
                    assert pred.shape==(len(te),) and np.isfinite(pred).all()
                    rec=dict(cohort=name,condition=condition,fold=int(outer),kind=kind,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),feature_names=names,prediction=pred.tolist(),seconds=time.monotonic()-t,context=str(context),context_sha256=sha(context),protocol_sha256=sha(HERE/'protocol.json'),selection=selection,code_sha256=sha(Path(__file__)),factory_sha256=sha(HERE/'foundation.py'))
                    if kind=='LGBM':rec.update(model=str(modelpath),model_sha256=sha(modelpath))
                    save(path,rec);records.append(rec);print(stem,kind,round(rec['seconds'],2),'seconds',flush=True)
                    save(HERE/f'progress-{args.scope}.json',dict(status='running',complete=len(records),last=stem+'-'+kind,elapsed_seconds=time.monotonic()-start))
                    del model;gc.collect()
    save(HERE/f'run-{args.scope}-receipt.json',dict(status='predictions_complete_pending_verification',models=len(records),elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(HERE/'protocol.json'),factory_sha256=sha(HERE/'foundation.py'),code_sha256=sha(Path(__file__)),predictions={p.name:sha(p) for p in (BIG/'predictions').glob('*.json') if (p.name.startswith('mobilised'))==(args.scope=='large')}))
    save(HERE/f'progress-{args.scope}.json',dict(status='predictions_complete_pending_verification',complete=len(records)))
if __name__=='__main__':main()
