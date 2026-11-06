"""Replay fixed multi-target-space fusion: source/person joins and NNLS optimum."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round10';O=B/'round10'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def main():
    proto=read(R/'protocol.json');receipt=read(R/'run-receipt.json');assert proto['code_sha256']==sha(H/'round10.py') and proto['data_sha256']==sha(GB/'data.npz') and proto['round07_verification_sha256']==sha(H/'round07/verification.json') and proto['round08_protocol_sha256']==sha(H/'round08/protocol.json');assert read(H/'round08/verification.json')['status']=='passed' and receipt['source08_verification_sha256']==sha(H/'round08/verification.json') and receipt['protocol_sha256']==sha(R/'protocol.json') and receipt['cases']==10 and receipt['models']==30
    for p,v in receipt['hashes'].items():assert sha(p)==v
    data=np.load(GB/'data.npz');ids=data['ids'];y=data['y'];preds={};checks=[];n=0
    for cond in ['speed','combined']:
      for f in range(5):
        rec=read(O/f'{cond}-{f}.json');assert rec['protocol_sha256']==sha(R/'protocol.json') and sha(rec['context'])==rec['context_sha256'] and sha(rec['source07'])==rec['source07_sha256'] and sha(rec['source08'])==rec['source08_sha256'];a=np.load(rec['source07']);b=np.load(rec['source08']);ctx=np.load(rec['context']);tr=np.flatnonzero(data['fold']!=f);te=np.flatnonzero(data['fold']==f)
        for c in [a,b,ctx]:assert np.array_equal(c['train_ids'],ids[tr]) and np.array_equal(c['test_ids'],ids[te])
        assert not set(ids[tr])&set(ids[te]) and np.array_equal(ctx['y'],y[tr]) and np.array_equal(a['y'],y[tr]) and np.array_equal(b['ytrain'],y[tr]);assert np.array_equal(a['X'][:,-1],b['Xtrain'][:,0]) and np.array_equal(a['T'][:,-1],b['Xtest'][:,0]);X=np.concatenate([a['X'][:,:-1],b['oof_matrix'],(1/a['X'][:,-1])[:,None]],axis=1);T=np.concatenate([a['T'][:,:-1],b['test_matrix'],(1/a['T'][:,-1])[:,None]],axis=1);assert np.array_equal(X,ctx['X']) and np.array_equal(T,ctx['T']);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;A=(X-mu)/sd;Z=(T-mu)/sd
        for r in rec['models']:
            assert sha(r['path'])==r['sha256'];m=np.load(r['path']);assert np.array_equal(mu,m['mean']) and np.array_equal(sd,m['std']);tail=r['tail'];w=np.where(y[tr]>=20,tail,np.where(y[tr]>=12,np.sqrt(tail),1.));assert np.array_equal(w,m['weight']);am=np.average(A,axis=0,weights=w);ym=np.average(y[tr],weights=w);coef=m['coef'];assert np.min(coef)>=0;g=(A-am).T@(w*((A-am)@coef-(y[tr]-ym)))+10*coef;assert np.min(g)>=-1e-6 and np.max(np.abs(g*coef))<1e-6 and np.isclose(m['intercept'],ym-am@coef,atol=1e-12,rtol=1e-12);p=Z@coef+m['intercept'];assert np.allclose(p,r['prediction'],atol=1e-12,rtol=1e-12);key=f'{cond}/tail{tail}';preds.setdefault(key,np.full(594,np.nan))[te]=r['prediction'];n+=1
        checks.append(dict(condition=cond,fold=f,models_replayed=3,training_only_source_matrices_and_IDs_exact=True,meta_KKT_optimum_passed=True))
    assert n==30;rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];out={};unc={};held=[]
    for key,p in preds.items():
        assert np.isfinite(p).all();e=np.abs(y-p);out[key]=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[slow].mean()),slow_bias=float((p-y)[slow].mean()),goal_met=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3));diff=e-np.abs(preds[key.replace('combined/','speed/')]-y);unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),difference_matching_speed=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((p-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=key,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,p))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap;9severeseparately;reuseddevelopmentfoldsnotexternalclinical/selectionadjusted.',conditions=unc));save(R/'verification.json',dict(status='passed',contexts=10,models_replayed=30,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
