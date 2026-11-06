from pathlib import Path
import sys,os,time,datetime,json,hashlib,itertools,gc
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
from round1 import H,B,SRC,read,save,sha,metrics,np,joblib,fit,predict,StratifiedKFold
sys.path.insert(0,'D:/Projects/D_P/studies/20261103-frailpol-short-walk')
from models import factory
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
def fit2(X,y,cfg):
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(X);a=imputer.transform(X);k=min(cfg.get('k',32),a.shape[1]);scores=f_regression(a,y)[0];scores=np.nan_to_num(scores,nan=-np.inf);rest=np.argsort(-scores[1:],kind='stable')[:max(k-1,0)]+1;cols=np.r_[0,rest];a=a[:,cols]
    model=None if cfg['kind'] in ['TabICLv2','TabPFNv2'] else fit(a,y,cfg)
    return dict(imputer=imputer,columns=cols,train_X=a,train_y=y,cfg=cfg,model=model)
def predict2(m,X):
    a=m['imputer'].transform(X)[:,m['columns']]
    if m['model'] is not None:return predict(m['model'],a)
    model=factory(m['cfg']['kind'],False);model.fit(m['train_X'],m['train_y']);p=model.predict(a);del model;gc.collect();return p
def main():
    h=H/'round03';b=B/'round03';assert read(h/'feature-verification.json')['status']=='passed';source=read(SRC/'cohorts.json');rr=[r for r in source if r['status']=='ok' and r['tug'] is not None];expanded={r['id']:r['features'] for r in read(b/'features.json')};ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);fold=np.array([r['fold'] for r in rr]);names=list(expanded[int(ids[0])]);xx=np.array([[expanded[int(i)][k] for k in names] for i in ids],float)
    configs=[dict(kind='ridge',alpha=10,k=k) for k in [16,32,187]]+[dict(kind='svr',C=100,gamma=.03,k=k) for k in [16,32]]+[dict(kind='extra',leaf=2,k=k) for k in [32,187]]+[dict(kind='linear_tree',leaf=15,rounds=150,linear_lambda=10,k=k) for k in [32,187]]+[dict(kind='residual',alpha=10,leaf=15,k=k) for k in [32,187]]+[dict(kind=k,k=32) for k in ['TabICLv2','TabPFNv2']]
    protocol=dict(round=3,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),round1_library_sha256=sha(H/'round1.py'),factory_sha256=sha(Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/models.py')),features_sha256=sha(b/'features.json'),source_sha256=sha(SRC/'cohorts.json'),candidates=configs,design='Same594 people/folds; outcome-blind contiguous same-foot straight cycles up to4seconds with original-window fallback,187features. Train-only median imputation and f_regression top-k with speed always retained. Fixed pre-trained models8ensembles32selectedfeatures, same verified weights. Preserve native units/no target clipping.',selection='Inner3 stratified by<12/12..20/>=20 seed20261103; minimize max(MAE/.9,p95/3,slowMAE/3), first tie. Evaluate selected and equal top3 average; no outer tuning.',conditions=['speed','expanded'],verification='Every candidate inner state/context saved; all final and inner contexts replayed independently. For foundation store training context plus fixed global weights, no duplicate checkpoint.',expected_additional_GB=1)
    if not(h/'protocol.json').exists():save(h/'protocol.json',protocol)
    p=read(h/'protocol.json');assert p['code_sha256']==sha(Path(__file__)) and p['features_sha256']==sha(b/'features.json');done=0
    for cond in ['speed','expanded']:
      X=xx[:,:1] if cond=='speed' else xx
      for outer in range(5):
        stem=f'{cond}-{outer}';out=b/(stem+'.json')
        if out.exists():done+=1;continue
        tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);np.savez_compressed(b/(stem+'-context.npz'),Xtrain=X[tr],ytrain=y[tr],Xtest=X[te],train_ids=ids[tr],test_ids=ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[];inner=[]
        for ci,cfg in enumerate(configs):
          ip=np.zeros(len(tr));t=time.monotonic()
          for k,(it,iv) in enumerate(splits):
            m=fit2(X[tr[it]],y[tr[it]],cfg);pr=predict2(m,X[tr[iv]]);mp=b/f'{stem}-candidate{ci}-inner{k}.joblib';joblib.dump(m,mp);ip[iv]=pr;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pr.tolist(),model=str(mp),sha256=sha(mp)))
          assert np.isfinite(ip).all();scores.append(metrics(y[tr],ip));print(stem,ci,cfg['kind'],round(time.monotonic()-t,1),'s','inner score',round(scores[-1]['score'],3),flush=True)
        top=np.argsort([s['score'] for s in scores],kind='stable')[:3];ps=[];models=[]
        for ci in top:
          m=fit2(X[tr],y[tr],configs[ci]);mp=b/f'{stem}-final{ci}.joblib';joblib.dump(m,mp);pr=predict2(m,X[te]);assert np.allclose(pr,predict2(joblib.load(mp),X[te]),atol=1e-6,rtol=1e-6);ps.append(pr);models.append(dict(candidate=int(ci),path=str(mp),sha256=sha(mp)))
        save(b/(stem+'-inner.json'),inner);save(out,dict(condition=cond,fold=outer,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner_scores=scores,top3=top.tolist(),models=models,selected_prediction=ps[0].tolist(),ensemble_prediction=np.mean(ps,axis=0).tolist(),context=str(b/(stem+'-context.npz')),protocol_sha256=sha(h/'protocol.json')));done+=1;save(h/'progress.json',dict(status='running',complete=done,expected=10,last=stem))
    save(h/'run-receipt.json',dict(status='complete_pending_independent_review',cases=done,hashes={str(f):sha(f) for f in b.glob('*') if f.is_file()},protocol_sha256=sha(h/'protocol.json')))
if __name__=='__main__':main()

