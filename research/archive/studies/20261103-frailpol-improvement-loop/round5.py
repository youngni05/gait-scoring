"""Bilateral short-walk comparison with nested selection and fixed train-only stacking."""
from pathlib import Path
import sys,time,datetime,gc
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round3 import B,SRC,read,save,sha,metrics,np,joblib,fit2,predict2,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
def configs_for(n):
    configs=[dict(kind='ridge',alpha=10,k=k) for k in [16,32,n]]+[dict(kind='svr',C=100,gamma=.03,k=k) for k in [16,32]]+[dict(kind='extra',leaf=2,k=k) for k in [32,n]]+[dict(kind='linear_tree',leaf=15,rounds=150,linear_lambda=10,k=k) for k in [32,n]]+[dict(kind='residual',alpha=10,leaf=15,k=k) for k in [32,n]]+[dict(kind=k,k=32) for k in ['TabICLv2','TabPFNv2']]
    unique=[]
    for c in configs:
        c={**c,'k':min(c['k'],n)}
        if c not in unique:unique.append(c)
    return unique
def meta_weights(y,tail):return np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.))
def fit_meta(X,y,tail):
    sc=StandardScaler().fit(X);a=sc.transform(X);w=meta_weights(y,tail);am=np.average(a,axis=0,weights=w);ym=np.average(y,weights=w);ac=(a-am)*np.sqrt(w[:,None]);yc=(y-ym)*np.sqrt(w);coef,_=nnls(np.vstack([ac,np.sqrt(10)*np.eye(a.shape[1])]),np.r_[yc,np.zeros(a.shape[1])],maxiter=10000)
    return dict(scaler=sc,coef=coef,intercept=float(ym-am@coef),tail=tail)
def predict_meta(m,X):return m['scaler'].transform(X)@m['coef']+m['intercept']
def main():
    h=H/'round05';b=B/'round05';assert read(h/'feature-verification.json')['status']=='passed';assert read(H/'resume-audit.json')['status']=='passed'
    source=read(SRC/'cohorts.json');rr=[r for r in source if r['status']=='ok' and r['tug'] is not None];expanded={r['id']:r['features'] for r in read(b/'features.json')};ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);fold=np.array([r['fold'] for r in rr]);names=list(expanded[int(ids[0])]);xx=np.array([[expanded[int(i)][k] for k in names] for i in ids],float);assert names[0]=='speed' and len(ids)==594
    cfgs={cond:configs_for(1 if cond=='speed' else xx.shape[1]) for cond in ['speed','bilateral']}
    protocol=dict(round=5,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),round3_library_sha256=sha(H/'round3.py'),round1_library_sha256=sha(H/'round1.py'),review2_library_sha256=sha(H/'review2.py'),factory_sha256=sha(Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/models.py')),features_sha256=sha(b/'features.json'),source_sha256=sha(SRC/'cohorts.json'),candidates_by_condition=cfgs,conditions=['speed','bilateral'],design='Same594people/folds. Bilateral complete-cycle mean and213short-window features, max4s core, original anchor fallback. Train-only median imputation/f_regression, speed always retained. Same classical/fixed-foundation families asround03. Deduplicate effective configs after clipping feature cap to actual input width, avoiding duplicated speed-only forests. No target clipping or new clinical inputs.',selection='Inner3 stratified TUG<12/12..20/>=20 seed20261103; exact outer-training person order; minimize max(MAE/.9,p95/3,slowMAE/3),first tie. Report selected and equal top3 average. Also two predeclared learned combinations without choosing among them from test results.',stacking='Fit positive Ridge alpha10 on training-only inner out-of-fold predictions of all effective candidates; standardize using that training matrix only. Unweighted and fixed tailweight4 (12..20weight2,>=20weight4). No meta hyperparameter search or performance claim on its training fit. Predict outer test using all base candidates refitted on outer training people. Save all base and meta models and contexts. Outer people never train preprocessing/base/meta or select candidates.',verification='300inner+100final base models and20meta models saved/replayed independently. Exact fixed inner partitions, imputer/ranking, original selection, training-only meta context/weights/coefficient optimality and independent metrics. Float32 foundation replay point tolerance1e-6 abs+rel retained; metric reconstruction uses original stored predictions at1e-12. No replay candidate reselection.',expected_additional_GB=1,limitations='Repeated development folds; not external clinical/camera validation. Short-window symmetry/variation proxies are not validated long-term variability. Slowgroup9people.')
    if not(h/'protocol.json').exists():save(h/'protocol.json',protocol)
    p=read(h/'protocol.json');assert p['code_sha256']==sha(Path(__file__)) and p['features_sha256']==sha(b/'features.json');done=0
    for cond in ['speed','bilateral']:
      X=xx[:,:1] if cond=='speed' else xx;configs=cfgs[cond]
      for outer in range(5):
        stem=f'{cond}-{outer}';out=b/(stem+'.json')
        if out.exists():done+=1;continue
        tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);np.savez_compressed(b/(stem+'-context.npz'),Xtrain=X[tr],ytrain=y[tr],Xtest=X[te],train_ids=ids[tr],test_ids=ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[];inner=[];oof_matrix=np.zeros((len(tr),len(configs)))
        for ci,cfg in enumerate(configs):
          t=time.monotonic()
          for k,(it,iv) in enumerate(splits):
            m=fit2(X[tr[it]],y[tr[it]],cfg);pr=predict2(m,X[tr[iv]]);mp=b/f'{stem}-candidate{ci}-inner{k}.joblib';joblib.dump(m,mp);oof_matrix[iv,ci]=pr;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pr.tolist(),model=str(mp),sha256=sha(mp)))
          scores.append(metrics(y[tr],oof_matrix[:,ci]));print(stem,ci,cfg['kind'],round(time.monotonic()-t,1),'s',round(scores[-1]['score'],3),flush=True)
        top=np.argsort([s['score'] for s in scores],kind='stable')[:3];test_matrix=np.zeros((len(te),len(configs)));models=[]
        for ci,cfg in enumerate(configs):
          m=fit2(X[tr],y[tr],cfg);mp=b/f'{stem}-final{ci}.joblib';joblib.dump(m,mp);pr=predict2(m,X[te]);assert np.allclose(pr,predict2(joblib.load(mp),X[te]),atol=1e-6,rtol=1e-6);test_matrix[:,ci]=pr;models.append(dict(candidate=ci,path=str(mp),sha256=sha(mp)))
        predictions=dict(selected=test_matrix[:,top[0]].tolist(),ensemble=np.mean(test_matrix[:,top],axis=1).tolist());metas=[]
        for variant,tail in [('stacked',1),('tail_stacked',4)]:
          meta=fit_meta(oof_matrix,y[tr],tail);mp=b/f'{stem}-{variant}.joblib';joblib.dump(meta,mp);pr=predict_meta(meta,test_matrix);assert np.array_equal(pr,predict_meta(joblib.load(mp),test_matrix));predictions[variant]=pr.tolist();metas.append(dict(kind=variant,tail=tail,path=str(mp),sha256=sha(mp)))
        np.savez_compressed(b/(stem+'-meta-context.npz'),oof_matrix=oof_matrix,test_matrix=test_matrix,train_ids=ids[tr],test_ids=ids[te]);save(b/(stem+'-inner.json'),inner);save(out,dict(condition=cond,fold=outer,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner_scores=scores,top3=top.tolist(),models=models,meta_models=metas,predictions=predictions,context=str(b/(stem+'-context.npz')),meta_context=str(b/(stem+'-meta-context.npz')),protocol_sha256=sha(h/'protocol.json')));done+=1;save(h/'progress.json',dict(status='running',complete=done,expected=10,last=stem))
    save(h/'run-receipt.json',dict(status='complete_pending_independent_review',cases=done,hashes={str(f):sha(f) for f in b.glob('*') if f.is_file()},protocol_sha256=sha(h/'protocol.json')))
if __name__=='__main__':main()
