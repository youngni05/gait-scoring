"""Replay every transformed-target/base/meta model without using outer labels in fitting."""
from pathlib import Path
import sys
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round8 import read,save,sha,np,joblib,predict2,StratifiedKFold,B,GB,R,O,CONFIGS
from review2 import check_context_model
from review5 import check_meta
def transform_check(m,X,y,cfg):
    assert m['transform']==cfg['target'];v=X[:,0]
    if cfg['target']=='speed_product':assert m['coefficient'] is None;t=y*v
    else:
        a=np.column_stack([np.ones(len(y)),1/v]);coef=np.linalg.lstsq(a,y,rcond=None)[0];assert np.array_equal(coef,m['coefficient']);assert np.max(np.abs(a.T@(a@coef-y)))<1e-7;t=y-a@coef
    check_context_model(m['base'],X,t,cfg)
def predict(m,X):
    raw=predict2(m['base'],X)
    return raw/X[:,0] if m['transform']=='speed_product' else raw+np.column_stack([np.ones(len(X)),1/X[:,0]])@m['coefficient']
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def main():
    proto=read(R/'protocol.json');receipt=read(R/'run-receipt.json');assert receipt['cases']==10 and receipt['protocol_sha256']==sha(R/'protocol.json');assert proto['code_sha256']==sha(H/'round8.py') and proto['library_sha256']==sha(H/'round3.py') and proto['meta_library_sha256']==sha(H/'round5.py') and proto['data_sha256']==sha(GB/'data.npz') and proto['candidates']==CONFIGS
    for p,v in receipt['hashes'].items():assert sha(p)==v
    old=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');assert proto['factory_sha256']==sha(old/'models.py')
    for w in read(old/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    d=np.load(GB/'data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];out={};held=[];checks=[];allpred={}
    for cond in ['speed','bilateral']:
      X=d['F'][:,:1] if cond=='speed' else d['F'];preds={k:np.full(594,np.nan) for k in ['selected','ensemble','tail4_stacked','tail16_stacked']}
      for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);rec=read(O/f'{cond}-{f}.json');ctx=np.load(rec['context']);assert rec['context_sha256']==sha(rec['context']) and rec['protocol_sha256']==sha(R/'protocol.json');assert np.array_equal(ctx['Xtrain'],X[tr],equal_nan=True) and np.array_equal(ctx['Xtest'],X[te],equal_nan=True) and np.array_equal(ctx['ytrain'],y[tr]) and np.array_equal(ctx['train_ids'],ids[tr]) and np.array_equal(ctx['test_ids'],ids[te]);assert not set(ids[tr])&set(ids[te]);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));a=np.zeros((len(tr),8));scores=[];delta=0
        for ci,cfg in enumerate(CONFIGS):
          for k,(it,iv) in enumerate(splits):
            rows=[r for r in rec['inner'] if r['candidate']==ci and r['inner']==k];assert len(rows)==1;r=rows[0];assert r['train_ids']==ids[tr[it]].tolist() and r['test_ids']==ids[tr[iv]].tolist();assert sha(r['model'])==r['sha256'];m=joblib.load(r['model']);transform_check(m,X[tr[it]],y[tr[it]],cfg);p=predict(m,X[tr[iv]]);assert np.allclose(p,r['prediction'],atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(p-r['prediction']))));a[iv,ci]=r['prediction']
          sc=metric(y[tr],a[:,ci]);assert all(np.isclose(val,rec['inner_scores'][ci][key],atol=1e-12,rtol=1e-12) for key,val in sc.items());scores.append(sc['score'])
        assert np.array_equal(a,ctx['oof_matrix']);top=np.argsort(scores,kind='stable')[:3];assert top.tolist()==rec['top3'];z=np.zeros_like(ctx['test_matrix']);assert len(rec['models'])==8
        for r in rec['models']:
            ci=r['candidate'];assert sha(r['path'])==r['sha256'];m=joblib.load(r['path']);transform_check(m,X[tr],y[tr],CONFIGS[ci]);p=predict(m,X[te]);assert np.allclose(p,ctx['test_matrix'][:,ci],atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(p-ctx['test_matrix'][:,ci]))));z[:,ci]=p
        assert np.array_equal(ctx['test_matrix'][:,top[0]],rec['predictions']['selected']) and np.array_equal(ctx['test_matrix'][:,top].mean(1),rec['predictions']['ensemble'])
        for r in rec['meta']:
            assert sha(r['path'])==r['sha256'];m=joblib.load(r['path']);tail=r['tail'];check_meta(m,a,y[tr],tail);orig=m['scaler'].transform(ctx['test_matrix'])@m['coef']+m['intercept'];replay=m['scaler'].transform(z)@m['coef']+m['intercept'];assert np.allclose(orig,rec['predictions'][f'tail{tail}_stacked'],atol=1e-10,rtol=1e-10);bound=np.abs(z-ctx['test_matrix'])@(np.abs(m['coef'])/m['scaler'].scale_);assert np.all(np.abs(replay-orig)<=bound+1e-10)
        for key in preds:preds[key][te]=rec['predictions'][key]
        checks.append(dict(condition=cond,fold=f,inner_models_replayed=24,final_models_replayed=8,meta_models_replayed=2,target_transform_training_only_verified=True,original_selection_verified=True,max_base_replay_delta_seconds=delta));print('verified',cond,f,flush=True)
      for key,p in preds.items():
        assert np.isfinite(p).all();name=cond+'/'+key;out[name]=metric(y,p);m=out[name];m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);allpred[name]=p;held.extend(dict(id=int(i),condition=name,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,p))
    rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];unc={}
    for name,p in allpred.items():
        e=np.abs(p-y);control=name.replace('bilateral/','speed/');diff=e-np.abs(allpred[control]-y);unc[name]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),difference_matching_speed=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((p-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]})
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap;9severeseparately;reuseddevelopmentfolds,notclinical/external/selectionadjusted.',conditions=unc));save(R/'verification.json',dict(status='passed',outer_cases=10,inner_models_replayed=240,final_models_replayed=80,meta_models_replayed=20,checks=checks,review_sha256=sha(Path(__file__)),comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json')));print(out)
if __name__=='__main__':main()
