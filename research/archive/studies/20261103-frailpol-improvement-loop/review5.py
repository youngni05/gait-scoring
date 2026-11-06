"""Independent saved-model, split, feature selection and learned-combination audit."""
from pathlib import Path
import sys
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round5 import *
from review2 import check_context_model
def check_meta(m,A,y,tail):
    sc=StandardScaler().fit(A);assert np.array_equal(sc.mean_,m['scaler'].mean_) and np.array_equal(sc.var_,m['scaler'].var_) and m['tail']==tail
    a=sc.transform(A);w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));am=np.average(a,axis=0,weights=w);ym=np.average(y,weights=w);coef=m['coef'];assert np.all(coef>=0)
    assert np.isclose(m['intercept'],ym-am@coef,atol=1e-10,rtol=1e-10)
    grad=(a-am).T@(w*((a-am)@coef-(y-ym)))+10*coef
    assert np.max(np.abs(grad[coef>1e-10]),initial=0)<1e-6 and np.min(grad[coef<=1e-10],initial=0)>-1e-6
    # Convex quadratic + nonnegative coefficients: KKT conditions prove the optimum.
def main():
    h=H/'round05';b=B/'round05';receipt=read(h/'run-receipt.json');assert receipt['cases']==10
    for p,v in receipt['hashes'].items():assert sha(p)==v
    proto=read(h/'protocol.json')
    for key,name in [('code_sha256','round5.py'),('round3_library_sha256','round3.py'),('round1_library_sha256','round1.py'),('review2_library_sha256','review2.py')]:assert proto[key]==sha(H/name)
    assert proto['source_sha256']==sha(SRC/'cohorts.json') and proto['features_sha256']==sha(b/'features.json')
    old=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');assert proto['factory_sha256']==sha(old/'models.py')
    for w in read(old/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    rows=[r for r in read(SRC/'cohorts.json') if r['status']=='ok' and r['tug'] is not None];ids=np.array([r['id'] for r in rows]);y=np.array([r['tug'] for r in rows]);fold=np.array([r['fold'] for r in rows]);expanded={r['id']:r['features'] for r in read(b/'features.json')};names=list(expanded[int(ids[0])]);xx=np.array([[expanded[int(i)][k] for k in names] for i in ids],float);assert len(ids)==594 and sum(y>=20)==9
    out={};oof=[];checks=[]
    for cond in proto['conditions']:
      X=xx[:,:1] if cond=='speed' else xx;configs=proto['candidates_by_condition'][cond];assert configs==configs_for(X.shape[1]);pred={k:np.zeros(len(y)) for k in ['selected','ensemble','stacked','tail_stacked']};seen=[]
      for f in range(5):
        stem=f'{cond}-{f}';rec=read(b/(stem+'.json'));ctx=np.load(rec['context']);meta_ctx=np.load(rec['meta_context']);tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);assert rec['train_ids']==ctx['train_ids'].tolist()==meta_ctx['train_ids'].tolist()==ids[tr].tolist();assert rec['test_ids']==ctx['test_ids'].tolist()==meta_ctx['test_ids'].tolist()==ids[te].tolist();assert not set(ids[tr])&set(ids[te])
        assert np.array_equal(ctx['Xtrain'],X[tr],equal_nan=True) and np.array_equal(ctx['Xtest'],X[te],equal_nan=True) and np.array_equal(ctx['ytrain'],y[tr]);assert rec['protocol_sha256']==sha(h/'protocol.json');splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));inner=read(b/(stem+'-inner.json'));assert len(inner)==len(configs)*3;scores=[];original_oof=np.zeros((len(tr),len(configs)));delta=0.
        for ci,cfg in enumerate(configs):
          for k,(it,iv) in enumerate(splits):
            matches=[r for r in inner if r['candidate']==ci and r['inner']==k];assert len(matches)==1;q=matches[0];assert q['train_ids']==ids[tr[it]].tolist() and q['test_ids']==ids[tr[iv]].tolist();assert sha(q['model'])==q['sha256'];m=joblib.load(q['model']);check_context_model(m,X[tr[it]],y[tr[it]],cfg);p=predict2(m,X[tr[iv]]);assert np.allclose(p,q['prediction'],atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(p-q['prediction']))));original_oof[iv,ci]=q['prediction']
          e=np.abs(original_oof[:,ci]-y[tr]);s=max(e.mean()/.9,np.quantile(e,.95)/3,e[y[tr]>=20].mean()/3);assert np.isclose(s,rec['inner_scores'][ci]['score'],atol=1e-12,rtol=1e-12);scores.append(s)
        assert np.array_equal(original_oof,meta_ctx['oof_matrix']);top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];test=np.zeros((len(te),len(configs)));assert len(rec['models'])==len(configs)
        for mr in rec['models']:
          ci=mr['candidate'];assert sha(mr['path'])==mr['sha256'];m=joblib.load(mr['path']);check_context_model(m,X[tr],y[tr],configs[ci]);p=predict2(m,X[te]);assert np.allclose(p,meta_ctx['test_matrix'][:,ci],atol=1e-6,rtol=1e-6);test[:,ci]=p;delta=max(delta,float(np.max(np.abs(p-meta_ctx['test_matrix'][:,ci]))))
        assert np.array_equal(meta_ctx['test_matrix'][:,top[0]],rec['predictions']['selected']);assert np.array_equal(np.mean(meta_ctx['test_matrix'][:,top],axis=1),rec['predictions']['ensemble']);meta_delta=0.;meta_bound=0.
        assert len(rec['meta_models'])==2
        for mr in rec['meta_models']:
          assert sha(mr['path'])==mr['sha256'];m=joblib.load(mr['path']);tail=1 if mr['kind']=='stacked' else 4;check_meta(m,original_oof,y[tr],tail);p=predict_meta(m,meta_ctx['test_matrix']);assert np.allclose(p,rec['predictions'][mr['kind']],atol=1e-10,rtol=1e-10)
          replay=predict_meta(m,test);bound=np.abs(test-meta_ctx['test_matrix'])@(np.abs(m['coef'])/m['scaler'].scale_);assert np.all(np.abs(replay-p)<=bound+1e-10);meta_delta=max(meta_delta,float(np.max(np.abs(replay-p))));meta_bound=max(meta_bound,float(np.max(bound)))
        for kind in pred:pred[kind][te]=rec['predictions'][kind]
        seen.extend(ids[te]);checks.append(dict(condition=cond,fold=f,inner_models_replayed=len(inner),final_models_replayed=len(configs),meta_models_replayed=2,original_selection_exact=True,training_only_meta_context_exact=True,meta_convex_optimum_KKT_passed=True,max_base_replay_delta_seconds=delta,max_meta_pipeline_replay_delta_seconds=meta_delta,max_meta_propagated_base_error_bound_seconds=meta_bound));print('verified',stem,flush=True)
      assert len(seen)==594 and set(seen)==set(ids)
      for kind,p in pred.items():
        e=np.abs(p-y);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['score']=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[cond+'/'+kind]=m;oof.extend(dict(id=int(i),condition=cond,kind=kind,actual=float(t),prediction=float(pr)) for i,t,pr in zip(ids,y,p))
    save(b/'heldout.json',oof);save(h/'comparison.json',out);save(h/'verification.json',dict(status='passed',outer_cases=10,inner_models_replayed=sum(c['inner_models_replayed'] for c in checks),final_models_replayed=sum(c['final_models_replayed'] for c in checks),meta_models_replayed=20,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(h/'comparison.json')));save(H/'progress.json',dict(status='round05_verified',goal_met=any(v['goal_met'] for k,v in out.items() if k.startswith('bilateral/')),round05=out));print(out)
if __name__=='__main__':main()
