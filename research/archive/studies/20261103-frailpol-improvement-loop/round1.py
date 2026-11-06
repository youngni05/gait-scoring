from pathlib import Path
import os,sys,json,hashlib,itertools,time,datetime
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
sys.stdout.reconfigure(encoding='utf8')
import numpy as np,joblib,lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');SRC=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def augment(X):return np.column_stack([X,1/X[:,0]])
def metrics(y,p):
    e=abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()),score=float(max(e.mean()/.9,np.quantile(e,.95)/3,e[y>=20].mean()/3)))
def fit(X,y,cfg):
    a=augment(X);sc=StandardScaler().fit(a);a=sc.transform(a);kind=cfg['kind'];alpha=cfg.get('alpha',10);base=None
    if kind=='ridge':model=Ridge(alpha=alpha)
    elif kind=='huber':model=HuberRegressor(alpha=alpha,max_iter=1000)
    elif kind=='svr':model=SVR(C=cfg['C'],gamma=cfg['gamma'],epsilon=.2)
    elif kind=='kernel':model=KernelRidge(alpha=alpha,kernel='rbf',gamma=cfg['gamma'])
    elif kind=='extra':model=ExtraTreesRegressor(n_estimators=200,min_samples_leaf=cfg['leaf'],max_features=1.,random_state=20261103,n_jobs=2)
    else:
        if kind=='residual':
            base=Ridge(alpha=alpha).fit(a,y);y=y-base.predict(a)
        params=dict(objective='regression',verbosity=-1,num_threads=2,seed=20261103,deterministic=True,force_col_wise=True,learning_rate=.03,num_leaves=7,max_depth=3,min_data_in_leaf=cfg['leaf'],lambda_l2=10,linear_tree=kind=='linear_tree',linear_lambda=cfg.get('linear_lambda',10))
        model=lgb.train(params,lgb.Dataset(a,label=y),num_boost_round=cfg.get('rounds',150));return dict(scaler=sc,model=model,base=base,cfg=cfg)
    model.fit(a,y);return dict(scaler=sc,model=model,base=base,cfg=cfg)
def predict(m,X):
    a=m['scaler'].transform(augment(X));p=m['model'].predict(a,num_threads=2) if isinstance(m['model'],lgb.Booster) else m['model'].predict(a)
    if m['base'] is not None:p=p+m['base'].predict(a)
    return p
def main():
    H.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True);h=H/'round01';b=B/'round01';h.mkdir(exist_ok=True);b.mkdir(exist_ok=True)
    cfgs=[dict(kind='ridge',alpha=x) for x in [.1,1,10,100]]+[dict(kind='huber',alpha=x) for x in [.01,1]]+[dict(kind='svr',C=c,gamma=g) for c,g in itertools.product([10,100],[.03,.3])]+[dict(kind='kernel',alpha=a,gamma=g) for a,g in itertools.product([1,10],[.03,.3])]+[dict(kind='extra',leaf=l) for l in [2,5]]+[dict(kind='linear_tree',leaf=l,rounds=r,linear_lambda=a) for l,r,a in itertools.product([8,20],[150,300],[1,10])]+[dict(kind='residual',leaf=l,alpha=a) for l,a in itertools.product([8,20],[1,10,100])]
    protocol=dict(round=1,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),goal=dict(mae=.90,p95=3.,slow_mae=3.),source_sha256=sha(SRC/'cohorts.json'),code_sha256=sha(Path(__file__)),candidates=cfgs,inputs='Same19 features or speed only; add reciprocal speed to each condition, standardize using training only. No demographics/fullTUG/record length.',selection='Inner3 StratifiedKFold by TUG<12/12..20/>=20 seed20261103 on outertrain only. Choose min max(MAE/0.9,p95/3,slowMAE/3),first tie. Frozen outer folds. Selected model and equal average of top3 inner-ranked candidates evaluated separately; no outer-outcome weight adjustment.',limitations='Repeated development cohort; no clinical validation. Slow>=20 has9people. Coverage fixed594. Never exclude hard people to meet goals.',verification='Save all inner models+predictions; reload and replay independently. Save final selected/top3 models and contexts. Independent IDs,selection,metrics,bootstrap; report all candidates and unsuccessful rounds.')
    if not (h/'protocol.json').exists():save(h/'protocol.json',protocol)
    p=read(h/'protocol.json');assert p['code_sha256']==sha(Path(__file__)) and p['source_sha256']==sha(SRC/'cohorts.json')
    rr=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);fold=np.array([r['fold'] for r in rr]);names=list(rr[0]['features']);xx=np.array([[r['features'][k] for k in names] for r in rr]);results=[]
    for cond in ['speed','features']:
      X=xx[:,:1] if cond=='speed' else xx
      for outer in range(5):
        stem=f'{cond}-{outer}';out=b/(stem+'.json')
        if out.exists():results.append(read(out));continue
        tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);np.savez_compressed(b/(stem+'-context.npz'),Xtrain=X[tr],ytrain=y[tr],Xtest=X[te],train_ids=ids[tr],test_ids=ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[];inner=[]
        for ci,cfg in enumerate(cfgs):
          ip=np.zeros(len(tr))
          for k,(it,iv) in enumerate(splits):
            m=fit(X[tr[it]],y[tr[it]],cfg);pred=predict(m,X[tr[iv]]);mp=b/f'{stem}-candidate{ci}-inner{k}.joblib';joblib.dump(m,mp);replay=predict(joblib.load(mp),X[tr[iv]]);assert np.allclose(pred,replay,atol=1e-9,rtol=1e-9);ip[iv]=pred;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pred.tolist(),model=str(mp),sha256=sha(mp)))
          assert np.isfinite(ip).all();scores.append(metrics(y[tr],ip));print(stem,'candidate',ci,cfg['kind'],'inner score',round(scores[-1]['score'],3),flush=True)
        top=np.argsort([s['score'] for s in scores],kind='stable')[:3];preds=[];models=[]
        for ci in top:
          m=fit(X[tr],y[tr],cfgs[ci]);mp=b/f'{stem}-final{ci}.joblib';joblib.dump(m,mp);pred=predict(m,X[te]);assert np.allclose(pred,predict(joblib.load(mp),X[te]),atol=1e-9,rtol=1e-9);preds.append(pred);models.append(dict(candidate=int(ci),path=str(mp),sha256=sha(mp)))
        rec=dict(condition=cond,fold=outer,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner_scores=scores,top3=top.tolist(),models=models,selected_prediction=preds[0].tolist(),ensemble_prediction=np.mean(preds,axis=0).tolist(),context=str(b/(stem+'-context.npz')),protocol_sha256=sha(h/'protocol.json'));save(b/(stem+'-inner.json'),inner);save(out,rec);results.append(rec);save(h/'progress.json',dict(status='running',complete=len(results),expected=10,last=stem))
    save(h/'run-receipt.json',dict(status='complete_pending_independent_review',cases=len(results),hashes={str(f):sha(f) for f in b.glob('*') if f.is_file()},protocol_sha256=sha(h/'protocol.json')))
if __name__=='__main__':main()
