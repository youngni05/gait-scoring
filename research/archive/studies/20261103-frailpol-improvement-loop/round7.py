"""Frozen training-only fusion of verified classical/foundation and temporal predictors."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
from scipy.optimize import nnls
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round07';O=B/'round07';TAILS=[1,4,16]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def source(condition,fold):
    d=np.load(GB/'data.npz');tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold);old='speed' if condition=='speed' else 'bilateral';context=np.load(B/'round05'/f'{old}-{fold}-meta-context.npz');assert np.array_equal(context['train_ids'],d['ids'][tr]) and np.array_equal(context['test_ids'],d['ids'][te]);a=[context['oof_matrix']];b=[context['test_matrix']];proof=[str(B/'round05'/f'{old}-{fold}-meta-context.npz')]
    for c in (['speed_mlp'] if condition=='speed' else ['speed_mlp','features_mlp','motion_speed','motion_features']):
        rec=read(GB/'local-main'/c/f'fold{fold}/complete.json');ep=rec['selected_epoch'];v=np.zeros(len(tr));seen=[]
        for inner in rec['inner']:
            ctx=read(inner['context']);idx=np.array(ctx['eval_indices']);row=[r for r in read(inner['curve']) if r['epoch']==ep][0];assert not set(ctx['train_indices'])&set(idx) and set(ctx['train_indices'])|set(idx)==set(tr);positions=np.array([np.flatnonzero(tr==i)[0] for i in idx]);v[positions]=row['prediction'];seen.extend(idx);proof.extend([inner['context'],inner['curve']])
        assert len(seen)==len(tr) and set(seen)==set(tr);a.append(v[:,None]);b.append(np.array(rec['prediction'])[:,None]);proof.append(str(GB/'local-main'/c/f'fold{fold}/complete.json'))
    a.append(d['F'][tr,:1]);b.append(d['F'][te,:1]);return d,tr,te,np.column_stack(a),np.column_stack(b),{p:sha(p) for p in proof}
def metric(y,p):
    e=np.abs(y-p);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);return m
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);assert read(H/'round05/verification.json')['status']=='passed';assert read(G/'local-directml/verification.json')['status']=='passed';proto=dict(round=7,code_sha256=sha(Path(__file__)),data_sha256=sha(GB/'data.npz'),classical_verification_sha256=sha(H/'round05/verification.json'),neural_verification_sha256=sha(G/'local-directml/verification.json'),conditions=['speed','combined'],fusion=['positive_ridge','rbf_ridge'],tail_weights=TAILS,alpha=10,gamma='1/number_of_columns after training-onlystandardization',design='Frozen594 and sameouter5; allmeta inputs are training-only innerOOF base predictions plus measuredspeed. Combined uses13bilateralclassical/foundation bases+4neural bases. Speed uses7speedbases+speedMLP. Samefinal refits fromverifiedpreviousstudies supply outertestfeatures. CNN epoch selected withinoutertrain as previouslyspecified, not reselected. Meta models fixed inadvance, no meta candidate selection on outertest or reported meta training performance.',weights='TUG>=20weighttail;12..20sqrt(tail),else1. positiveRidge centered by weightedmeans and NNLS withalpha10;RBFkernel regression K+alpha*diag(1/weight),target centeredweightedmean. No output clipping. Every preprocessing/meta fit onlyoutertrain.',verification='Save60models+10contexts; independent rebuildfromsaved verifiedbasepredictions and exactIDs; trainingnormalization/weights/NNLSKKT or kernel equation checked; modelreloadand metrics rebuilt;pairedpersonbootstrap including9severeseparately. Reuseddevelopmentfolds,notclinical/externalvalidation.',expected_additional_GB=.03)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==proto
    else:save(R/'protocol.json',proto)
    for cond in proto['conditions']:
      for f in range(5):
        path=O/f'{cond}-{f}.json'
        if path.exists():continue
        d,tr,te,X,T,proof=source(cond,f);y=d['y'][tr];mean=X.mean(0);std=X.std(0);std[std<1e-8]=1;A=(X-mean)/std;Z=(T-mean)/std;ctx=O/f'{cond}-{f}-context.npz';np.savez_compressed(ctx,X=X,T=T,train_ids=d['ids'][tr],test_ids=d['ids'][te],y=y);rows=[]
        for family in proto['fusion']:
          for tail in TAILS:
            w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));ym=np.average(y,weights=w);model=O/f'{cond}-{f}-{family}-{tail}.npz'
            if family=='positive_ridge':
                am=np.average(A,axis=0,weights=w);coef,_=nnls(np.vstack([(A-am)*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(A.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(A.shape[1])],maxiter=10000);intercept=ym-am@coef;p=Z@coef+intercept;np.savez_compressed(model,mean=mean,std=std,coef=coef,intercept=intercept,weight=w)
            else:
                gamma=1/A.shape[1];K=np.exp(-gamma*((A[:,None,:]-A[None,:,:])**2).sum(2));dual=np.linalg.solve(K+np.diag(10/w),y-ym);kt=np.exp(-gamma*((Z[:,None,:]-A[None,:,:])**2).sum(2));p=kt@dual+ym;np.savez_compressed(model,mean=mean,std=std,anchor=A,dual=dual,intercept=ym,gamma=gamma,weight=w)
            assert np.isfinite(p).all();rows.append(dict(family=family,tail=tail,path=str(model),sha256=sha(model),prediction=p.tolist()))
        save(path,dict(condition=cond,fold=f,models=rows,context=str(ctx),context_sha256=sha(ctx),base_artifacts=proof,protocol_sha256=sha(R/'protocol.json')));print(cond,f,flush=True)
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',cases=10,models=60,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}))
if __name__=='__main__':main()
