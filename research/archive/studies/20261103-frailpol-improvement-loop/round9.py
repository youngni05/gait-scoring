"""Fixed monotone hinge fusion: permit different corrections across predicted TUG ranges."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.optimize import nnls
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round09';O=B/'round09'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def expand(x):
    p=x[:,:-1];speed=x[:,-1];assert np.isfinite(speed).all() and np.all(speed>0);return np.column_stack([p,np.maximum(p-12,0),np.maximum(p-20,0),1/speed])
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);assert read(H/'round07/verification.json')['status']=='passed';d=np.load(GB/'data.npz');proto=dict(round=9,code_sha256=sha(Path(__file__)),source_verification_sha256=sha(H/'round07/verification.json'),data_sha256=sha(GB/'data.npz'),conditions=['speed','combined'],alpha=[10,100],tail=[1,4,16],knots=[12,20],design='Same594/frozen5outerfolds; verifiedround07innerOOF+finaltestbasepredictions. InputpredictedTUGvalues pluspositivehinges at12/20secondsand1/observedspeed. No sourceTUGlabelsincalibrationfeatures. Sameexpansionbothconditions; allpreprocessing/metaoptimizationoutertrainingonly. FixednonnegativeRidgeweights produce piecewise monotone correction of basepredictions. Noperformance-basedknotselection orclipping.',selection='All6fixedalpha/tailvariantsreported, no meta search/outerchoice/fitontest. Weightedcentering,NNLSridge withalpha10or100;TUG>=20tail,12..20sqrt(tail),else1. No metafittrainingperformanceusedasvalidation.',verification='Save60models/10contexts; independentsourcejoin/nonlinearfeaturerebuild, training-onlymoments/weights/KKToptimum andsavedmodelinference/metrics;paired5000personbootstrapwith9severeseparately. Reuseddevelopmentfolds,notclinical/external.',expected_additional_GB=.01)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==proto
    else:save(R/'protocol.json',proto)
    for cond in proto['conditions']:
      for f in range(5):
        path=O/f'{cond}-{f}.json'
        if path.exists():continue
        source=B/'round07'/f'{cond}-{f}-context.npz';c=np.load(source);X=expand(c['X']);T=expand(c['T']);y=c['y'];mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;A=(X-mu)/sd;Z=(T-mu)/sd;ctx=O/f'{cond}-{f}-context.npz';np.savez_compressed(ctx,X=X,T=T,train_ids=c['train_ids'],test_ids=c['test_ids'],y=y);rows=[]
        for alpha in proto['alpha']:
          for tail in proto['tail']:
            w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));am=np.average(A,axis=0,weights=w);ym=np.average(y,weights=w);coef,_=nnls(np.vstack([(A-am)*np.sqrt(w[:,None]),np.sqrt(alpha)*np.eye(A.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(A.shape[1])],maxiter=10000);intercept=ym-am@coef;pr=Z@coef+intercept;model=O/f'{cond}-{f}-a{alpha}-t{tail}.npz';np.savez_compressed(model,mean=mu,std=sd,coef=coef,intercept=intercept,weight=w);rows.append(dict(alpha=alpha,tail=tail,path=str(model),sha256=sha(model),prediction=pr.tolist()))
        save(path,dict(condition=cond,fold=f,source=str(source),source_sha256=sha(source),context=str(ctx),context_sha256=sha(ctx),models=rows,protocol_sha256=sha(R/'protocol.json')))
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',cases=10,models=60,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}));print('round09fitcomplete')
if __name__=='__main__':main()
