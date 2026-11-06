"""Source/person/hinge/KKT and inference review, independently calculated metrics."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round09';O=B/'round09'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def main():
    proto=read(R/'protocol.json');receipt=read(R/'run-receipt.json');assert proto['code_sha256']==sha(H/'round9.py') and proto['source_verification_sha256']==sha(H/'round07/verification.json') and proto['data_sha256']==sha(GB/'data.npz') and receipt['cases']==10 and receipt['models']==60 and receipt['protocol_sha256']==sha(R/'protocol.json')
    for p,v in receipt['hashes'].items():assert sha(p)==v
    data=np.load(GB/'data.npz');y=data['y'];ids=data['ids'];preds={};checks=[];n=0
    for cond in ['speed','combined']:
      for f in range(5):
        rec=read(O/f'{cond}-{f}.json');assert rec['protocol_sha256']==sha(R/'protocol.json') and sha(rec['context'])==rec['context_sha256'] and sha(rec['source'])==rec['source_sha256'];base=np.load(rec['source']);ctx=np.load(rec['context']);tr=np.flatnonzero(data['fold']!=f);te=np.flatnonzero(data['fold']==f);assert np.array_equal(ctx['train_ids'],ids[tr]) and np.array_equal(ctx['test_ids'],ids[te]) and np.array_equal(ctx['y'],y[tr]) and not set(ids[tr])&set(ids[te]);assert np.array_equal(base['train_ids'],ids[tr]) and np.array_equal(base['test_ids'],ids[te])
        for key,sourcekey in [('X','X'),('T','T')]:
            raw=base[sourcekey];p=raw[:,:-1];x=np.concatenate([p,np.where(p>12,p-12,0),np.where(p>20,p-20,0),(1/raw[:,-1])[:,None]],axis=1);assert np.array_equal(ctx[key],x)
        X=ctx['X'];T=ctx['T'];mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;A=(X-mu)/sd;Z=(T-mu)/sd
        for r in rec['models']:
            assert sha(r['path'])==r['sha256'];m=np.load(r['path']);assert np.array_equal(m['mean'],mu) and np.array_equal(m['std'],sd);tail=r['tail'];alpha=r['alpha'];w=np.where(y[tr]>=20,tail,np.where(y[tr]>=12,np.sqrt(tail),1.));assert np.array_equal(m['weight'],w);am=np.average(A,axis=0,weights=w);ym=np.average(y[tr],weights=w);coef=m['coef'];assert np.min(coef)>=0;gradient=(A-am).T@(w*((A-am)@coef-(y[tr]-ym)))+alpha*coef;assert np.min(gradient)>=-1e-6 and np.max(np.abs(gradient*coef))<1e-6;assert np.isclose(m['intercept'],ym-am@coef,atol=1e-12,rtol=1e-12);p=Z@coef+m['intercept'];assert np.allclose(p,r['prediction'],atol=1e-12,rtol=1e-12);key=f'{cond}/alpha{alpha}/tail{tail}';preds.setdefault(key,np.full(594,np.nan))[te]=r['prediction'];n+=1
        checks.append(dict(condition=cond,fold=f,models_replayed=6,source_people_and_hinge_expansion_exact=True,training_only_moments_weights_KKT_passed=True))
    assert n==60;rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];out={};unc={};held=[]
    for key,p in preds.items():
        assert np.isfinite(p).all();e=np.abs(y-p);out[key]=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[slow].mean()),slow_bias=float((p-y)[slow].mean()),goal_met=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3));diff=e-np.abs(preds[key.replace('combined/','speed/')]-y);unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),difference_matching_speed=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((p-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=key,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,p))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap,9severeseparately,reuseddevelopmentfoldsnotclinical/external/selectionadjusted.',conditions=unc));save(R/'verification.json',dict(status='passed',contexts=10,models_replayed=60,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
