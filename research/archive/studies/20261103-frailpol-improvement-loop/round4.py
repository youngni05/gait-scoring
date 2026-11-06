"""Frozen regression over fixed random-convolution features and gait descriptors."""
from pathlib import Path
import sys,time,datetime,itertools
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round1 import B,SRC,read,save,sha,metrics,np,joblib,StratifiedKFold,StandardScaler,Ridge
def weight(y,strength):return np.where(y>=20,strength,np.where(y>=12,np.sqrt(strength),1.))
def fit4(X,y,cfg):
    a=np.column_stack([X,1/X[:,0]]);sc=StandardScaler().fit(a);a=sc.transform(a)
    target=y-(6/X[:,0] if cfg['target']=='overhead' else 0)
    model=Ridge(alpha=cfg['alpha'],solver='cholesky').fit(a,target,sample_weight=weight(y,cfg['tail_weight']))
    return dict(scaler=sc,model=model,cfg=cfg)
def predict4(m,X):
    a=m['scaler'].transform(np.column_stack([X,1/X[:,0]]));p=m['model'].predict(a)
    return p+(6/X[:,0] if m['cfg']['target']=='overhead' else 0)
def main():
    h=H/'round04';b=B/'round04';assert read(h/'feature-verification.json')['status']=='passed'
    assert read(H/'round03/verification.json')['status']=='passed'
    assert not any(v['goal_met'] for k,v in read(H/'round03/comparison.json').items() if k.startswith('expanded/')),'Prior round reached goal; do not start unnecessary fitting'
    rows=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rows]);y=np.array([r['tug'] for r in rows]);fold=np.array([r['fold'] for r in rows]);data=np.load(b/'features.npz');assert np.array_equal(ids,data['ids']) and len(ids)==594
    configs=[dict(alpha=a,tail_weight=w,target=t) for a,w,t in itertools.product([1.,10.,100.,1000.],[1.,4.],['raw','overhead'])]
    conditions=['single-speed','single-features','multi-speed','multi-features']
    protocol=dict(round=4,created=datetime.datetime.now(datetime.timezone.utc).isoformat(),code_sha256=sha(Path(__file__)),round1_library_sha256=sha(H/'round1.py'),source_sha256=sha(SRC/'cohorts.json'),features_sha256=sha(b/'features.npz'),kernel_bank_sha256=sha(b/'kernel-bank.json'),feature_code_sha256=sha(H/'features4.py'),candidates=configs,conditions=conditions,design='594samepeople/folds. Single original or multi-cycle frozen windows. Fixed random-convolution2000features+19gait descriptors, versus same-window speed only. Append reciprocal speed; standardize training only. Ridge raw TUG or residual TUG-6/speed;6meters is nominal3m out-and-back distance, a model prior not measured phase durations. Tail weights1or4, intermediate TUG12..20 weight sqrt(tailweight). No target clipping, no time-record/turn/clinical inputs. 16configs, inner3 stratified, same maxnormalized3metric selection; selected and equal top3ensemble.',selection='min max(MAE/.9,p95/3,slowMAE/3) inner3 stratified TUG<12/12..20/>=20 seed20261103; first tie. No outer fitting or selection.',verification='All960 inner and60final models saved and independently replayed, exact cohort and training-only scaler/weights/selection independently rebuilt.',expected_additional_GB=.3,limitations='Repeated outer development data. Not clinical or RGB validation. Slow group9people. Random convolution is ROCKET-inspired custom implementation, not original paper reproduction.')
    if not(h/'protocol.json').exists():save(h/'protocol.json',protocol)
    p=read(h/'protocol.json');assert p['code_sha256']==sha(Path(__file__)) and p['features_sha256']==sha(b/'features.npz');done=0
    for cond in conditions:
      mode,kind=cond.split('-');X=data[mode][:,:1] if kind=='speed' else data[mode];assert np.isfinite(X).all()
      for outer in range(5):
        stem=f'{cond}-{outer}';out=b/(stem+'.json')
        if out.exists():done+=1;continue
        tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);np.savez_compressed(b/(stem+'-context.npz'),Xtrain=X[tr],ytrain=y[tr],Xtest=X[te],train_ids=ids[tr],test_ids=ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[];inner=[]
        for ci,cfg in enumerate(configs):
          ip=np.zeros(len(tr));start=time.monotonic()
          for k,(it,iv) in enumerate(splits):
            m=fit4(X[tr[it]],y[tr[it]],cfg);pr=predict4(m,X[tr[iv]]);mp=b/f'{stem}-candidate{ci}-inner{k}.joblib';joblib.dump(m,mp);assert np.allclose(pr,predict4(joblib.load(mp),X[tr[iv]]),atol=1e-9,rtol=1e-9);ip[iv]=pr;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pr.tolist(),model=str(mp),sha256=sha(mp)))
          scores.append(metrics(y[tr],ip));print(stem,ci,round(time.monotonic()-start,2),'s',round(scores[-1]['score'],3),flush=True)
        top=np.argsort([s['score'] for s in scores],kind='stable')[:3];ps=[];models=[]
        for ci in top:
          m=fit4(X[tr],y[tr],configs[ci]);mp=b/f'{stem}-final{ci}.joblib';joblib.dump(m,mp);pr=predict4(m,X[te]);assert np.allclose(pr,predict4(joblib.load(mp),X[te]),atol=1e-9,rtol=1e-9);ps.append(pr);models.append(dict(candidate=int(ci),path=str(mp),sha256=sha(mp)))
        save(b/(stem+'-inner.json'),inner);save(out,dict(condition=cond,fold=outer,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner_scores=scores,top3=top.tolist(),models=models,selected_prediction=ps[0].tolist(),ensemble_prediction=np.mean(ps,axis=0).tolist(),context=str(b/(stem+'-context.npz')),protocol_sha256=sha(h/'protocol.json')));done+=1;save(h/'progress.json',dict(status='running',complete=done,expected=20,last=stem))
    save(h/'run-receipt.json',dict(status='complete_pending_independent_review',cases=done,hashes={str(f):sha(f) for f in b.glob('*') if f.is_file()},protocol_sha256=sha(h/'protocol.json')))
if __name__=='__main__':main()
