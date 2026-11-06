"""Replay all final states and independently reconstruct split/selection integrity."""
from models import *
import lightgbm as lgb
import itertools,time,gc,warnings
warnings.filterwarnings('ignore',message='auto_scale_n_estimators is deprecated')
def main():
    start=time.monotonic();protocol=read(HERE/'protocol.json');receipt=read(HERE/'run-receipt.json');rows=read(BIG/'cohorts.json');byid={r['id']:r for r in rows};checks=[]
    assert receipt['models']==90 and len(receipt['predictions'])==90
    for p,h in protocol['source_hashes'].items():assert sha(p)==h,p
    for w in read(HERE/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    assert sha(HERE/'run.py')==receipt['code_sha256'];assert sha(HERE/'models.py')==receipt['factory_sha256']
    grid=protocol['LGBM']['grid'];cfgs=[dict(zip(grid,v)) for v in itertools.product(*grid.values())]
    for filename,h in sorted(receipt['predictions'].items()):
        path=BIG/'predictions'/filename;assert sha(path)==h;rec=read(path);assert sha(rec['context'])==rec['context_sha256'];assert sha(HERE/'protocol.json')==rec['protocol_sha256']
        c=np.load(rec['context']);tr=c['train_ids'].tolist();te=c['test_ids'].tolist();target=rec['target'];cl=target=='frailty'
        eligible={r['id'] for r in rows if r['status']=='ok' and r[target] is not None}
        assert len(set(tr))==len(tr) and len(set(te))==len(te) and not set(tr)&set(te) and set(tr)|set(te)==eligible
        assert all(byid[i]['fold']==rec['fold'] for i in te) and all(byid[i]['fold']!=rec['fold'] for i in tr)
        assert np.array_equal(c['y_train'],[byid[i][target] for i in tr]);cols=c['feature_names'].tolist();assert cols==(['speed'] if rec['condition']=='speed' else list(byid[tr[0]]['features']))
        for part,ids in [('train',tr),('test',te)]:assert np.array_equal(c['X_'+part],[[byid[i]['features'][k] for k in cols] for i in ids])
        selection_checked=False
        if rec['kind']=='LGBM':
            assert sha(rec['model'])==rec['model_sha256'];model=lgb.Booster(model_file=rec['model']);pred=undo(model.predict(c['X_test'],num_threads=2),target)
            inner=read(BIG/'inner'/(f"{target}-{rec['condition']}-{rec['fold']}.json"));assert len(inner)==48;losses=[]
            for candidate in range(16):
                errs=[];seen=[]
                for v in [v for v in inner if v['candidate']==candidate]:
                    it,iv=set(v['train']),set(v['evaluation']);assert not it&iv and it|iv==set(tr) and not iv&set(te);seen.extend(v['evaluation'])
                    yy=np.array([byid[i][target] for i in v['evaluation']]);pp=np.array(v['prediction'])
                    errs.extend(-np.log(np.maximum(pp[np.arange(len(yy)),yy.astype(int)],1e-15)) if cl else np.abs(pp-yy))
                assert len(seen)==len(tr) and set(seen)==set(tr);losses.append(float(np.mean(errs)))
            assert np.allclose(losses,rec['selection']['losses'],rtol=1e-12,atol=1e-12);best=int(np.argmin(losses));assert best==rec['selection']['candidate'] and cfgs[best]==rec['selection']['params'];selection_checked=True
        else:
            model=factory(rec['kind'],cl);model.fit(c['X_train'],transform(c['y_train'],target));pred=model.predict_proba(c['X_test']) if cl else undo(model.predict(c['X_test']),target)
        expected=np.array(rec['prediction']);delta=float(np.max(abs(pred-expected)));assert np.allclose(pred,expected,rtol=1e-6,atol=1e-6),(filename,delta)
        if cl:assert pred.shape==(len(te),3) and np.allclose(pred.sum(axis=1),1,atol=1e-5) and (pred>=0).all()
        checks.append(dict(file=filename,people=len(te),max_abs_replay_difference=delta,inner_selection_checked=selection_checked));save(HERE/'review-progress.json',dict(complete=len(checks),expected=90,last=filename));print('verified',len(checks),filename,delta,flush=True);del model;gc.collect()
    save(HERE/'verification.json',dict(status='passed',models=90,contexts=30,inner_selections=30,max_abs_replay_difference=max(c['max_abs_replay_difference'] for c in checks),seconds=time.monotonic()-start,protocol_sha256=sha(HERE/'protocol.json'),cohorts_sha256=sha(BIG/'cohorts.json'),checks=checks,review_code_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
