"""Independent CatBoost border/split/model/metric audit; no refitting on test people."""
from pathlib import Path
import sys,hashlib,time
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round6 import *
def check_model(rec,X,y,cfg,h):
    assert rec['cfg']==cfg and sha(rec['path'])==rec['sha256'] and sha(rec['borders'])==rec['borders_sha256'];assert rec['parameters']==parameters(cfg)
    target=y*X[:,0] if cfg['target']=='velocity_product' else y.copy();w=np.where(y>=20,cfg['tail_weight'],np.where(y>=12,cfg['tail_weight']**.5,1.))
    for key,a in [('target_sha256',target),('weights_sha256',w),('training_X_sha256',X)]:assert rec[key]==hashlib.sha256(a.tobytes()).hexdigest()
    # Recreate training-only quantization from original source rows independently.
    p=Pool(X,label=target,weight=w,thread_count=2);p.quantize(border_count=64,feature_border_type='Median',nan_mode='Min',random_seed=20261104);temp=h/'verification-borders.txt';p.save_quantization_borders(str(temp));assert temp.read_bytes()==Path(rec['borders']).read_bytes()
    m=CatBoostRegressor();m.load_model(rec['path']);params=m.get_all_params();assert m.tree_count_==400 and params['depth']==cfg['depth'] and params['boosting_type']=='Ordered' and params['bootstrap_type']=='No' and params['random_seed']==20261104 and params['nan_mode']=='Min';assert np.float32(params['learning_rate'])==np.float32(.03) and params['l2_leaf_reg']==10
    if cfg['loss']=='RMSE':assert params['loss_function']=='RMSE'
    else:assert params['loss_function'].startswith('Huber:delta=') and float(params['loss_function'].split('=')[1])==1.
    available={}
    for line in Path(rec['borders']).read_text().splitlines():
        parts=line.split('\t');available.setdefault(int(parts[0]),set()).add(np.float32(float(parts[1])).tobytes())
    for col,values in m.get_borders().items():assert all(np.float32(value).tobytes() in available.get(col,set()) for value in values)
    leafcounts=m.get_tree_leaf_counts();leafweights=m.get_leaf_weights();ends=np.r_[0,np.cumsum(leafcounts)];assert all(np.isclose(leafweights[ends[k]:ends[k+1]].sum(),w.sum(),atol=1e-8,rtol=1e-10) for k in range(len(leafcounts)))
    return m
def direct(m,X,cfg):
    pred=np.asarray(m.predict(X,thread_count=2),dtype=float)
    return pred/X[:,0] if cfg['target']=='velocity_product' else pred
def main():
    h=H/'round06';b=B/'round06';proto=read(h/'protocol.json');assert proto['code_sha256']==sha(H/'round6.py') and proto['round1_library_sha256']==sha(H/'round1.py') and proto['source_sha256']==sha(SRC/'cohorts.json');assert proto['features_sha256']==sha(B/'round05/features.json') and proto['catboost_binary_sha256']==sha(LIB/'catboost/_catboost.pyd');assert catboost.__version__==proto['catboost_version'];xx,y,ids,fold=load();out={};oof=[];checks=[]
    for cond in proto['conditions']:
      receipt=read(h/f'receipt-{cond}.json');assert receipt['cases']==5 and receipt['protocol_sha256']==sha(h/'protocol.json')
      for p,v in receipt['hashes'].items():assert sha(p)==v
      X=xx[:,:1] if cond=='speed' else xx;pred={k:np.zeros(len(y)) for k in ['selected','ensemble']};seen=[]
      for f in range(5):
        rec=read(b/cond/f'fold{f}.json');ctx=np.load(rec['context']);tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);assert rec['train_ids']==ctx['train_ids'].tolist()==ids[tr].tolist() and rec['test_ids']==ctx['test_ids'].tolist()==ids[te].tolist();assert np.array_equal(ctx['Xtrain'],X[tr],equal_nan=True) and np.array_equal(ctx['Xtest'],X[te],equal_nan=True) and np.array_equal(ctx['ytrain'],y[tr]);assert not set(ids[tr])&set(ids[te]);assert rec['protocol_sha256']==sha(h/'protocol.json');inner=read(b/cond/f'fold{f}-inner.json');assert len(inner)==48;splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));scores=[];maxdelta=0.
        for ci,cfg in enumerate(proto['candidates']):
          ip=np.zeros(len(tr))
          for k,(it,iv) in enumerate(splits):
            matches=[q for q in inner if q['candidate']==ci and q['inner']==k];assert len(matches)==1;q=matches[0];assert q['train_ids']==ids[tr[it]].tolist() and q['test_ids']==ids[tr[iv]].tolist();m=check_model(q['model'],X[tr[it]],y[tr[it]],cfg,h);p=direct(m,X[tr[iv]],cfg);assert np.allclose(p,q['prediction'],atol=1e-10,rtol=1e-10);maxdelta=max(maxdelta,float(np.max(np.abs(p-q['prediction']))));ip[iv]=q['prediction']
          e=np.abs(ip-y[tr]);score=max(e.mean()/.9,np.quantile(e,.95)/3,e[y[tr]>=20].mean()/3);assert np.isclose(score,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(score)
        top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];ps=[];assert len(rec['models'])==3
        for pos,mr in enumerate(rec['models']):assert mr['candidate']==top[pos];m=check_model(mr['model'],X[tr],y[tr],proto['candidates'][mr['candidate']],h);ps.append(direct(m,X[te],proto['candidates'][mr['candidate']]))
        assert np.allclose(ps[0],rec['selected_prediction'],atol=1e-10,rtol=1e-10) and np.allclose(np.mean(ps,axis=0),rec['ensemble_prediction'],atol=1e-10,rtol=1e-10)
        for kind in pred:pred[kind][te]=rec[kind+'_prediction']
        seen.extend(ids[te]);checks.append(dict(condition=cond,fold=f,inner_models_replayed=48,final_models_replayed=3,training_only_quantization_rebuilt=True,training_leaf_weight_sums_verified=True,max_replay_delta=maxdelta,top3=top.tolist()));print('verified',cond,f,flush=True)
      assert len(seen)==594 and set(seen)==set(ids)
      for kind,p in pred.items():
        e=np.abs(p-y);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['score']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[cond+'/'+kind]=m;oof.extend(dict(id=int(i),condition=cond,kind=kind,actual=float(t),prediction=float(pr)) for i,t,pr in zip(ids,y,p))
    save(b/'heldout.json',oof);save(h/'comparison.json',out);save(h/'verification.json',dict(status='passed',outer_cases=10,inner_models_replayed=480,final_models_replayed=30,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(h/'comparison.json')));save(H/'progress.json',dict(status='round06_verified',goal_met=any(v['goal_met'] for k,v in out.items() if k.startswith('bilateral/')),round06=out));print(out)
if __name__=='__main__':main()
