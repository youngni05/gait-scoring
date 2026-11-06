"""Predeclared train-only fusion of raw-target and speed-adjusted-target predictors."""
from pathlib import Path
import json,hashlib,time,argparse
import numpy as np
from scipy.optimize import nnls
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round10';O=B/'round10'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def freeze():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);proto=dict(round=10,code_sha256=sha(Path(__file__)),created_unix=time.time(),data_sha256=sha(GB/'data.npz'),round07_verification_sha256=sha(H/'round07/verification.json'),round08_protocol_sha256=sha(H/'round08/protocol.json'),conditions=['speed','combined'],alpha=10,tail=[1,4,16],design='Freezebeforeviewinground08outerresults. Same594/frozen5personfolds. Concatenateverifiedround07rawtargetbaseinnerOOF andround08speed-product/inverse-speed-residualbaseinnerOOF plus1/observedspeed. Preserveallbases,notchooseusingouterperformance. MatchedsourceIDs/order; testinputsfromcorrespondingbasefinalrefits. Noouterlabels infeatureconstruction,normalizers,weights,metafit. Round07combined17bases andround08bilateral8bases;speed8and8respectively. Originalsourcecandidatecounts differfromdeduplicatedfeaturecaps,so thisiswholepipelinecomparison,notisolatedequal-sizeensemblecausality.',models='PositiveRidgealpha10,train-onlystandardization andweightedcentering;tail1/4/16withmiddle sqrt(tail). All3fixedvariantsreported;no outer/meta hyperparam selection, clipping orscore-basedposthocweights.',verification='Round08fullindependentreplaymustpassfirst. All30meta models and10contexts saved; sourceprotocol/verifier/hash/IDmatches independentlyrebuilt,training-onlymoments/weights/NNLSKKT,savedmodelinference/metrics/paireduncertainty. Allsourcebase_replays coveredbytheirunchangedreceipts. Reuseddevelopmentfolds,notexternalclinicalvalidation.',expected_additional_GB=.01)
    if(R/'protocol.json').exists():
        old=read(R/'protocol.json');proto['created_unix']=old['created_unix'];assert old==proto
    else:save(R/'protocol.json',proto)
    return read(R/'protocol.json')
def main():
    p=freeze();assert read(H/'round08/verification.json')['status']=='passed' and read(H/'round07/verification.json')['status']=='passed';d=np.load(GB/'data.npz')
    for cond in p['conditions']:
      for f in range(5):
        path=O/f'{cond}-{f}.json'
        if path.exists():continue
        src7=B/'round07'/f'{cond}-{f}-context.npz';src8=B/'round08'/f'{"speed" if cond=="speed" else "bilateral"}-{f}-context.npz';a=np.load(src7);b=np.load(src8);assert np.array_equal(a['train_ids'],b['train_ids']) and np.array_equal(a['test_ids'],b['test_ids']) and np.array_equal(a['y'],b['ytrain']);assert np.array_equal(a['X'][:,-1],b['Xtrain'][:,0]) and np.array_equal(a['T'][:,-1],b['Xtest'][:,0]);X=np.column_stack([a['X'][:,:-1],b['oof_matrix'],1/a['X'][:,-1]]);T=np.column_stack([a['T'][:,:-1],b['test_matrix'],1/a['T'][:,-1]]);y=a['y'];mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1.;A=(X-mu)/sd;Z=(T-mu)/sd;context=O/f'{cond}-{f}-context.npz';np.savez_compressed(context,X=X,T=T,train_ids=a['train_ids'],test_ids=a['test_ids'],y=y);models=[]
        for tail in p['tail']:
            w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));am=np.average(A,axis=0,weights=w);ym=np.average(y,weights=w);coef,_=nnls(np.vstack([(A-am)*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(A.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(A.shape[1])],maxiter=10000);intercept=ym-am@coef;pr=Z@coef+intercept;model=O/f'{cond}-{f}-tail{tail}.npz';np.savez_compressed(model,mean=mu,std=sd,coef=coef,intercept=intercept,weight=w);models.append(dict(tail=tail,path=str(model),sha256=sha(model),prediction=pr.tolist()))
        save(path,dict(condition=cond,fold=f,source07=str(src7),source08=str(src8),source07_sha256=sha(src7),source08_sha256=sha(src8),context=str(context),context_sha256=sha(context),models=models,protocol_sha256=sha(R/'protocol.json')))
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',cases=10,models=30,protocol_sha256=sha(R/'protocol.json'),source08_verification_sha256=sha(H/'round08/verification.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}));print('round10fitcomplete')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args();freeze() if a.freeze_only else main()
