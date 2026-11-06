"""Independent split audit, weighted ridge solve, saved-model replay and metrics."""
from pathlib import Path
import sys
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round4 import *
def check_model(m,X,y,cfg):
    assert m['cfg']==cfg
    a=np.column_stack([X,1/X[:,0]]);sc=StandardScaler().fit(a)
    assert np.array_equal(sc.mean_,m['scaler'].mean_) and np.array_equal(sc.var_,m['scaler'].var_)
    a=sc.transform(a);w=np.where(y>=20,cfg['tail_weight'],np.where(y>=12,cfg['tail_weight']**.5,1.));t=y-6/X[:,0] if cfg['target']=='overhead' else y
    # Solve weighted ridge directly, independently of sklearn.fit.
    am=np.average(a,axis=0,weights=w);tm=np.average(t,weights=w);ac=(a-am)*np.sqrt(w[:,None]);tc=(t-tm)*np.sqrt(w)
    if ac.shape[1]>ac.shape[0]:coef=ac.T@np.linalg.solve(ac@ac.T+cfg['alpha']*np.eye(len(ac)),tc)
    else:coef=np.linalg.solve(ac.T@ac+cfg['alpha']*np.eye(ac.shape[1]),ac.T@tc)
    intercept=tm-am@coef
    assert np.allclose(coef,m['model'].coef_,atol=1e-8,rtol=1e-8) and np.isclose(intercept,m['model'].intercept_,atol=1e-8,rtol=1e-8)
def main():
    h=H/'round04';b=B/'round04';receipt=read(h/'run-receipt.json');assert receipt['cases']==20
    for p,v in receipt['hashes'].items():assert sha(p)==v
    proto=read(h/'protocol.json');assert proto['code_sha256']==sha(H/'round4.py') and proto['round1_library_sha256']==sha(H/'round1.py') and proto['source_sha256']==sha(SRC/'cohorts.json');assert proto['features_sha256']==sha(b/'features.npz') and proto['kernel_bank_sha256']==sha(b/'kernel-bank.json') and proto['feature_code_sha256']==sha(H/'features4.py')
    rows=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rows]);y=np.array([r['tug'] for r in rows]);fold=np.array([r['fold'] for r in rows]);data=np.load(b/'features.npz');assert np.array_equal(ids,data['ids']) and len(ids)==594 and sum(y>=20)==9
    out={};oof=[];checks=[]
    for cond in proto['conditions']:
      mode,kind=cond.split('-');X=data[mode][:,:1] if kind=='speed' else data[mode];pred={k:np.zeros(len(y)) for k in ['selected','ensemble']};seen=[]
      for outer in range(5):
        stem=f'{cond}-{outer}';rec=read(b/(stem+'.json'));ctx=np.load(rec['context']);tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);assert ctx['train_ids'].tolist()==rec['train_ids']==ids[tr].tolist() and ctx['test_ids'].tolist()==rec['test_ids']==ids[te].tolist();assert np.array_equal(ctx['Xtrain'],X[tr]) and np.array_equal(ctx['Xtest'],X[te]) and np.array_equal(ctx['ytrain'],y[tr]);assert not set(ids[tr])&set(ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])))
        inner=read(b/(stem+'-inner.json'));assert len(inner)==len(proto['candidates'])*3;scores=[];delta=0.
        for ci,cfg in enumerate(proto['candidates']):
          pp=np.zeros(len(tr));seeninner=[]
          for k,(it,iv) in enumerate(splits):
            matches=[r for r in inner if r['candidate']==ci and r['inner']==k];assert len(matches)==1;r=matches[0];assert r['train_ids']==ids[tr[it]].tolist() and r['test_ids']==ids[tr[iv]].tolist();assert sha(r['model'])==r['sha256'];m=joblib.load(r['model']);check_model(m,X[tr[it]],y[tr[it]],cfg);p=predict4(m,X[tr[iv]]);assert np.allclose(p,r['prediction'],atol=1e-9,rtol=1e-9);delta=max(delta,float(np.max(np.abs(p-r['prediction']))));pp[iv]=r['prediction'];seeninner.extend(r['test_ids'])
          assert len(seeninner)==len(tr) and set(seeninner)==set(ids[tr]);e=np.abs(pp-y[tr]);score=max(e.mean()/.9,np.percentile(e,95)/3,e[y[tr]>=20].mean()/3);assert np.isclose(score,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(score)
        top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];ps=[]
        for pos,mr in enumerate(rec['models']):
          assert mr['candidate']==top[pos] and sha(mr['path'])==mr['sha256'];m=joblib.load(mr['path']);check_model(m,X[tr],y[tr],proto['candidates'][mr['candidate']]);ps.append(predict4(m,X[te]))
        assert np.allclose(ps[0],rec['selected_prediction'],atol=1e-9,rtol=1e-9);assert np.allclose(np.mean(ps,axis=0),rec['ensemble_prediction'],atol=1e-9,rtol=1e-9)
        for kind in pred:pred[kind][te]=rec[kind+'_prediction']
        seen.extend(ids[te]);checks.append(dict(condition=cond,fold=outer,inner_models_replayed=len(inner),final_models_replayed=len(ps),weighted_ridge_independently_solved=True,max_replay_delta=delta,top3=top.tolist()));print('verified',stem,flush=True)
      assert len(seen)==len(ids) and set(seen)==set(ids)
      for kind,p in pred.items():
        e=np.abs(p-y);m=dict(mae=float(e.mean()),p95=float(np.percentile(e,95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['score']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[cond+'/'+kind]=m
        oof.extend(dict(id=int(i),condition=cond,kind=kind,actual=float(t),prediction=float(pr)) for i,t,pr in zip(ids,y,p))
    save(b/'heldout.json',oof);save(h/'comparison.json',out);save(h/'verification.json',dict(status='passed',outer_cases=20,inner_models_replayed=960,final_models_replayed=60,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(h/'comparison.json')));save(H/'progress.json',dict(status='round04_verified',goal_met=any(v['goal_met'] for k,v in out.items() if '-features/' in k),round04=out));print(out)
if __name__=='__main__':main()
