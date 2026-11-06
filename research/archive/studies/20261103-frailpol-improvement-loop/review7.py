"""Independent normal-equation/KKT and person-boundary replay for round07."""
from pathlib import Path
import json,hashlib
import numpy as np
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round07';O=B/'round07'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def main():
    proto=read(R/'protocol.json');receipt=read(R/'run-receipt.json');assert proto['code_sha256']==sha(H/'round7.py') and proto['data_sha256']==sha(GB/'data.npz');assert receipt['cases']==10 and receipt['models']==60 and receipt['protocol_sha256']==sha(R/'protocol.json')
    for p,v in receipt['hashes'].items():assert sha(p)==v
    d=np.load(GB/'data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];preds={};count=0;checks=[]
    for cond in ['speed','combined']:
      for f in range(5):
        rec=read(O/f'{cond}-{f}.json');tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);ctx=np.load(rec['context']);assert rec['context_sha256']==sha(rec['context']) and rec['protocol_sha256']==sha(R/'protocol.json');assert np.array_equal(ctx['train_ids'],ids[tr]) and np.array_equal(ctx['test_ids'],ids[te]) and np.array_equal(ctx['y'],y[tr]);assert not set(ids[tr])&set(ids[te])
        for p,v in rec['base_artifacts'].items():assert sha(p)==v
        old='speed' if cond=='speed' else 'bilateral';base=np.load(B/'round05'/f'{old}-{f}-meta-context.npz');train=[base['oof_matrix']];test=[base['test_matrix']]
        for c in (['speed_mlp'] if cond=='speed' else ['speed_mlp','features_mlp','motion_speed','motion_features']):
            neural=read(GB/'local-main'/c/f'fold{f}/complete.json');lookup={};epoch=neural['selected_epoch']
            for inner in neural['inner']:
                ic=read(inner['context']);assert not set(ic['train_ids'])&set(ic['eval_ids']);assert set(ic['train_ids'])|set(ic['eval_ids'])==set(ids[tr]);row=[r for r in read(inner['curve']) if r['epoch']==epoch][0]
                for pid,pr in zip(ic['eval_ids'],row['prediction']):assert pid not in lookup;lookup[pid]=pr
            assert set(lookup)==set(ids[tr]);train.append(np.array([lookup[int(pid)] for pid in ids[tr]])[:,None]);assert neural['test_ids']==ids[te].tolist();test.append(np.array(neural['prediction'])[:,None])
        train.append(d['F'][tr,:1]);test.append(d['F'][te,:1]);X=np.column_stack(train);T=np.column_stack(test);assert np.array_equal(X,ctx['X']) and np.array_equal(T,ctx['T']);mean=X.mean(0);std=X.std(0);std[std<1e-8]=1;A=(X-mean)/std;Z=(T-mean)/std
        for row in rec['models']:
            family=row['family'];tail=row['tail'];assert sha(row['path'])==row['sha256'];m=np.load(row['path']);assert np.array_equal(m['mean'],mean) and np.array_equal(m['std'],std);w=np.where(y[tr]>=20,tail,np.where(y[tr]>=12,np.sqrt(tail),1.));assert np.array_equal(m['weight'],w);ym=np.average(y[tr],weights=w)
            if family=='positive_ridge':
                am=np.average(A,axis=0,weights=w);ac=A-am;yc=y[tr]-ym;coef=m['coef'];g=ac.T@(w*(ac@coef-yc))+10*coef;assert np.min(coef)>=0 and np.min(g)>=-1e-6 and np.max(np.abs(g*coef))<1e-6;assert np.isclose(m['intercept'],ym-am@coef,atol=1e-12,rtol=1e-12);p=Z@coef+m['intercept']
            else:
                assert np.array_equal(m['anchor'],A) and m['gamma']==1/A.shape[1] and m['intercept']==ym;K=np.exp(-m['gamma']*((A[:,None,:]-A[None,:,:])**2).sum(-1));res=(K+np.diag(10/w))@m['dual']-(y[tr]-ym);assert np.max(np.abs(res))<1e-8;p=np.exp(-m['gamma']*((Z[:,None,:]-A[None,:,:])**2).sum(-1))@m['dual']+ym
            assert np.allclose(p,row['prediction'],atol=1e-12,rtol=1e-12);key=f'{cond}/{family}/tail{tail}';preds.setdefault(key,np.full(594,np.nan))[te]=row['prediction'];count+=1
        checks.append(dict(condition=cond,fold=f,people_separated=True,meta_inputs_independently_rebuilt=True,training_only_normalization_and_weights=True,models_replayed=6))
    assert count==60;out={};unc={};rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,len(slow),(5000,len(slow)))];held=[]
    for key,p in preds.items():
        assert np.isfinite(p).all();e=np.abs(p-y);out[key]=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[slow].mean()),slow_bias=float((p-y)[slow].mean()),goal_met=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3));control=key.replace('combined/','speed/');difference=e-np.abs(preds[control]-y);unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),difference_matching_speed=float(difference.mean()),difference_ci95=np.quantile(difference[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((p-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=key,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,p))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000 paired person bootstrap, conditional on reused development folds; severe9 separately. Not selection-adjusted/external/clinical.',conditions=unc));save(R/'verification.json',dict(status='passed',contexts=10,models_replayed=60,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
