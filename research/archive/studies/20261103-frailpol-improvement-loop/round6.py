"""Frozen CatBoost comparison, raw seconds versus velocity-product target."""
from pathlib import Path
import sys,os,json,itertools,time,datetime,argparse,hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');SRC=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');LIB=B/'runtime-catboost';sys.path[:0]=[str(LIB),str(H)]
from round1 import read,save,sha,metrics,np,StratifiedKFold
import catboost
from catboost import CatBoostRegressor,Pool
CONFIGS=[dict(depth=d,loss=l,target=t,tail_weight=w) for d,l,t,w in itertools.product([3,5],['RMSE','Huber:delta=1.0'],['raw','velocity_product'],[1,4])]
def load():
    rr=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ff={r['id']:r['features'] for r in read(B/'round05/features.json')};ids=np.array([r['id'] for r in rr]);y=np.array([r['tug'] for r in rr]);fold=np.array([r['fold'] for r in rr]);names=list(ff[int(ids[0])]);X=np.array([[ff[int(i)][k] for k in names] for i in ids],float);assert len(ids)==594 and sum(y>=20)==9 and names[0]=='speed' and np.all(X[:,0]>0);return X,y,ids,fold
def targets(X,y,cfg):return y*X[:,0] if cfg['target']=='velocity_product' else y.copy()
def weights(y,cfg):return np.where(y>=20,cfg['tail_weight'],np.where(y>=12,np.sqrt(cfg['tail_weight']),1.))
def parameters(cfg,iterations=400):return dict(iterations=iterations,depth=cfg['depth'],loss_function=cfg['loss'],learning_rate=.03,l2_leaf_reg=10,boosting_type='Ordered',bootstrap_type='No',random_strength=0,random_seed=20261104,thread_count=2,allow_writing_files=False,verbose=False,task_type='CPU',nan_mode='Min',border_count=64,feature_border_type='Median')
def quantized(X,y,cfg):
    p=Pool(X,label=targets(X,y,cfg),weight=weights(y,cfg),thread_count=2);p.quantize(border_count=64,feature_border_type='Median',nan_mode='Min',random_seed=20261104);return p
def fit_model(X,y,cfg,path,iterations=400):
    pool=quantized(X,y,cfg);borders=Path(str(path)+'.borders.txt');pool.save_quantization_borders(str(borders));m=CatBoostRegressor(**parameters(cfg,iterations));m.fit(pool);m.save_model(str(path));return m,dict(path=str(path),sha256=sha(path),borders=str(borders),borders_sha256=sha(borders),cfg=cfg,target_sha256=hashlib.sha256(targets(X,y,cfg).tobytes()).hexdigest(),weights_sha256=hashlib.sha256(weights(y,cfg).tobytes()).hexdigest(),training_X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),parameters=parameters(cfg,iterations))
def predict(m,X,cfg):
    p=np.array(m.predict(X,thread_count=2),float);return p/X[:,0] if cfg['target']=='velocity_product' else p
def prepare():
    h=H/'round06';b=B/'round06';h.mkdir(exist_ok=True);b.mkdir(exist_ok=True);binary=LIB/'catboost/_catboost.pyd';assert binary.exists() and catboost.__version__=='1.2.10'
    proto=dict(round=6,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',code_sha256=sha(Path(__file__)),round1_library_sha256=sha(H/'round1.py'),source_sha256=sha(SRC/'cohorts.json'),features_sha256=sha(B/'round05/features.json'),catboost_version=catboost.__version__,catboost_binary_sha256=sha(binary),candidates=CONFIGS,conditions=['speed','bilateral'],design='Same594people/folds and frozen bilateral213features/core<=4s. CatBoost1.2.10Ordered,symmetric depth3/5,400trees,learningrate.03,L2=10,no bootstrap/random-strength,seed20261104. Native missing-value handling. Each training split fits its own64Medianquantization borders; no evaluation data is quantized with fit or used for early stopping.',targets='RawTUGseconds or targetTUG*observed short-window speed; invert product prediction by that same observed speed at inference, without clipping. No actual testTUG used in transform inversion. Tailweights1/4, intermediate12..20sqrt(tailweight). RMSE versusHuber(delta1 in each target units). This is a predictive parameterization, not a measured physical travel distance.',selection='Same3innerStratifiedKFold by<12/12..20/>=20 seed20261103 on outertraining only. Exact original person order for metrics, minimize max(MAE/.9,p95/3,slowMAE/3),first tie. Report selected and equal top3 average. No outer selection/calibration.',verification='480inner+30final native model files, training borders, contexts and predictions saved. Independent borders rebuild from training people only, original targets/weights hashes, all saved models reload/replay, exactIDs/inner partitions/selection and independent metrics. Paired5000personbootstrap. No new GPU or external data upload.',expected_additional_GB=.7,limitations='Repeated outer development folds; no external/camera clinical validation; severe subgroup9people.')
    if(h/'protocol.json').exists():
        old=read(h/'protocol.json');assert old['code_sha256']==proto['code_sha256'] and old['features_sha256']==proto['features_sha256'] and old['catboost_binary_sha256']==proto['catboost_binary_sha256']
    else:save(h/'protocol.json',proto)
    return h,b
def pilot():
    h,b=prepare();p=b/'synthetic-pilot';p.mkdir(exist_ok=True);rng=np.random.default_rng(6);checks=[];start=time.monotonic()
    for nf in [1,5]:
      X=rng.normal(size=(60,nf));X[:,0]=rng.uniform(.2,1.8,60);X[rng.random(X.shape)<.08]=np.nan;X[:,0]=rng.uniform(.2,1.8,60);y=rng.uniform(6,32,60)
      for ci,cfg in enumerate(CONFIGS):
        path=p/f'n{nf}-config{ci}.cbm';m,rec=fit_model(X[:45],y[:45],cfg,path,25);pr=predict(m,X[45:],cfg);reload=CatBoostRegressor();reload.load_model(str(path));assert np.array_equal(pr,predict(reload,X[45:],cfg));assert np.isfinite(pr).all();checks.append(dict(nfeatures=nf,candidate=ci,sha256=sha(path)))
    save(h/'pilot-verification.json',dict(status='passed',data='Synthetic only,no cohort fitting or performance selection',models_replayed=len(checks),seconds=time.monotonic()-start,checks=checks));print('synthetic CatBoost pilot passed',len(checks),time.monotonic()-start,flush=True)
def worker(cond):
    h,b=prepare();assert read(h/'pilot-verification.json')['status']=='passed';xx,y,ids,fold=load();X=xx[:,:1] if cond=='speed' else xx;outdir=b/cond;outdir.mkdir(exist_ok=True);done=0
    for f in range(5):
      recpath=outdir/f'fold{f}.json'
      if recpath.exists():done+=1;continue
      tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);context=outdir/f'fold{f}-context.npz';np.savez_compressed(context,Xtrain=X[tr],Xtest=X[te],ytrain=y[tr],train_ids=ids[tr],test_ids=ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));inner=[];scores=[]
      for ci,cfg in enumerate(CONFIGS):
        ip=np.zeros(len(tr));start=time.monotonic()
        for k,(it,iv) in enumerate(splits):
            path=outdir/f'fold{f}-candidate{ci}-inner{k}.cbm';m,fitrec=fit_model(X[tr[it]],y[tr[it]],cfg,path);pr=predict(m,X[tr[iv]],cfg);ip[iv]=pr;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pr.tolist(),model=fitrec))
        scores.append(metrics(y[tr],ip));print(cond,f,ci,round(time.monotonic()-start,2),'s',round(scores[-1]['score'],3),flush=True)
      top=np.argsort([s['score'] for s in scores],kind='stable')[:3];models=[];preds=[]
      for ci in top:
        cfg=CONFIGS[int(ci)];path=outdir/f'fold{f}-final{ci}.cbm';m,fitrec=fit_model(X[tr],y[tr],cfg,path);pr=predict(m,X[te],cfg);reload=CatBoostRegressor();reload.load_model(str(path));assert np.array_equal(pr,predict(reload,X[te],cfg));models.append(dict(candidate=int(ci),model=fitrec));preds.append(pr)
      save(outdir/f'fold{f}-inner.json',inner);save(recpath,dict(condition=cond,fold=f,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),context=str(context),inner_scores=scores,top3=top.tolist(),models=models,selected_prediction=preds[0].tolist(),ensemble_prediction=np.mean(preds,axis=0).tolist(),protocol_sha256=sha(h/'protocol.json')));done+=1;save(h/f'progress-{cond}.json',dict(status='running',completed=done,expected=5,last_fold=f))
    save(h/f'receipt-{cond}.json',dict(status='complete_pending_independent_review',cases=done,hashes={str(p):sha(p) for p in outdir.glob('*') if p.is_file()},protocol_sha256=sha(h/'protocol.json')))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','pilot','speed','bilateral']);a=ap.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='pilot':pilot()
    else:worker(a.mode)
