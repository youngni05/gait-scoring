"""Second batch: preserve prior feature-only experts and add new experts."""
from pathlib import Path
import json,sys
import numpy as np,joblib
from scipy.optimize import linprog,nnls
import sensor_loop as s
HERE=s.HERE;HEAVY=s.HEAVY
VIEWS={'speed':'speed','timing7':'timing','metric12':'calibrated_spatiotemporal'}
def fit_lp(A,y,tail=1):
    n,k=A.shape;targetw=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));targetw/=targetw.sum()
    # Convex model weights, free intercept, absolute residuals, L1 deviation from uniform.
    c=np.r_[np.zeros(k+1),targetw,np.full(k,.01)]
    left=np.c_[-A,-np.ones(n),-np.eye(n),np.zeros((n,k))]
    right=np.c_[A,np.ones(n),-np.eye(n),np.zeros((n,k))]
    reg1=np.c_[np.eye(k),np.zeros((k,1+n)),-np.eye(k)]
    reg2=np.c_[-np.eye(k),np.zeros((k,1+n)),-np.eye(k)]
    U=np.vstack([left,right,reg1,reg2]);B=np.r_[-y,y,np.full(k,1/k),np.full(k,-1/k)]
    E=np.r_[np.ones(k),np.zeros(1+n+k)][None,:]
    res=linprog(c,A_ub=U,b_ub=B,A_eq=E,b_eq=[1],bounds=[(0,None)]*k+[(None,None)]+[(0,None)]*(n+k),method='highs')
    assert res.success
    return {'weights':res.x[:k],'bias':res.x[k],'tail':tail,'objective':res.fun}
def fit_nn(A,y,alpha):
    mean=A.mean(0);std=np.maximum(A.std(0),1e-6);z=(A-mean)/std;ym=y.mean()
    w=nnls(np.vstack([z,np.sqrt(alpha)*np.eye(A.shape[1])]),np.r_[y-ym,np.zeros(A.shape[1])])[0]
    return {'center':mean,'scale':std,'ymean':ym,'coef':w,'alpha':alpha}
def main():
    assert json.loads((HERE/'sensor-verification.json').read_text())['passed']
    data=np.load(s.SOURCE);y=data['y'];records=json.loads((HERE/'sensor-receipt.json').read_text())
    protocol={'code_sha256':s.sha(__file__),'source_new_verification_sha256':s.sha(HERE/'sensor-verification.json'),
      'old_features':'same1/7/12 allowed quantities only; reuse prior verified innerOOF bases, no fullsensor/CNNoutputs',
      'views':VIEWS,'primary':'all old+new expert innerOOF columns, fixed nonnegativeRidgealpha10; every heldout person excluded from all corresponding bases',
      'secondary':'Ridgealpha1/100 and convex MAE mixtures with training-only high-score weights1/4; no choosing by outer results',
      'reason':'First new recipe was worse than original0.982; retain original stronger feature-only experts instead of replacing them',
      'limits':'adaptive followup on reused development cohort; no untouched external accuracy claim'}
    if not (HERE/'refinement-protocol.json').exists():s.save(HERE/'refinement-protocol.json',protocol)
    else:assert json.loads((HERE/'refinement-protocol.json').read_text())==protocol
    (HEAVY/'refinement').mkdir(exist_ok=True);out=[];pred={}
    for view,oldview in VIEWS.items():
      for r in records:
        if r['view']!=view:continue
        f=r['fold'];tr=np.array(r['train']);te=np.array(r['test'])
        newpath=HEAVY/'sensor-models'/r['context'];assert s.sha(newpath)==r['context_sha256'];nc=np.load(newpath)
        oldrecord=s.OLD/f'{oldview}-{f}.json';o=json.loads(oldrecord.read_text());op=Path(o['context']);assert s.sha(op)==o['context_sha256'];oc=np.load(op)
        assert o['train_ids']==r['train_ids'] and o['test_ids']==r['test_ids'];assert np.array_equal(oc['ytrain'],y[tr])
        A=np.column_stack([oc['A'],nc['A']]);T=np.column_stack([oc['T'],nc['T']])
        models={f'ridge{alpha}':fit_nn(A,y[tr],alpha) for alpha in [1,10,100]}
        models.update({f'mae_tail{tail}':fit_lp(A,y[tr],tail) for tail in [1,4]})
        modelpath=HEAVY/'refinement'/f'{view}-{f}.joblib';joblib.dump(models,modelpath)
        ctx=HEAVY/'refinement'/f'{view}-{f}.npz';np.savez_compressed(ctx,A=A,T=T,train=tr,test=te)
        for policy,m in models.items():
            if policy.startswith('ridge'):pp=np.maximum(.1,(T-m['center'])/m['scale']@m['coef']+m['ymean'])
            else:pp=np.maximum(.1,T@m['weights']+m['bias'])
            key=view+'/'+policy
            if key not in pred:pred[key]=np.full(len(y),np.nan)
            pred[key][te]=pp
        out.append({'view':view,'fold':f,'train':tr.tolist(),'test':te.tolist(),'model':modelpath.name,'model_sha256':s.sha(modelpath),
          'context':ctx.name,'context_sha256':s.sha(ctx),'source_old_record':str(oldrecord),'source_old_context_sha256':s.sha(op),'source_new_context_sha256':s.sha(newpath)})
        s.save(HERE/'refinement-receipt.json',out);print('REFINED',view,f,flush=True)
    np.savez_compressed(HEAVY/'refinement/predictions.npz',y=y,ids=data['ids'],**{k.replace('/','__'):p for k,p in pred.items()})
    print('REFINEMENT_FITTING_COMPLETE',len(out),flush=True)
if __name__=='__main__':main()
