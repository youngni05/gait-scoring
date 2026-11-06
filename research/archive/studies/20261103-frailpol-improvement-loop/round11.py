"""Frozen input-support fallback and fusion of entirely reviewed short-walk experts."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.optimize import nnls
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round11';O=B/'round11';CONDS=['speed','speed_duration','combined'];TAILS=[1,4,8,16];WAVES=['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16']
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def verified_json(path,receipt):
    assert sha(path)==receipt['hashes'][str(path)];return read(path)
def neural(root,c,f,train_ids,test_ids,receipt):
    path=root/c/f'fold{f}/fold.json';r=verified_json(path,receipt);lookup={};proof={str(path):sha(path)};epoch=r['epoch']
    for fit in r['inner']:
        p=Path(fit['context']);ctx=verified_json(p,receipt);assert not set(ctx['train_ids'])&set(ctx['eval_ids']) and set(ctx['train_ids'])|set(ctx['eval_ids'])==set(train_ids);row=next(q for q in fit['curve'] if q['epoch']==epoch)
        for pid,pred in zip(ctx['eval_ids'],row['prediction']):assert pid not in lookup;lookup[pid]=pred
        proof[str(p)]=sha(p)
    assert set(lookup)==set(train_ids) and r['test_ids']==test_ids.tolist();return np.array([lookup[int(i)] for i in train_ids]),np.array(r['prediction']),proof
def sources(cond,f,d):
    tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);ids=d['ids'];c7='combined' if cond=='combined' else 'speed';c8='bilateral' if cond=='combined' else 'speed';p7=B/'round07'/f'{c7}-{f}-context.npz';p8=B/'round08'/f'{c8}-{f}-context.npz';r7=read(H/'round07/run-receipt.json');r8=read(H/'round08/run-receipt.json');assert sha(p7)==r7['hashes'][str(p7)] and sha(p8)==r8['hashes'][str(p8)];a=np.load(p7);b=np.load(p8);assert np.array_equal(a['train_ids'],ids[tr]) and np.array_equal(a['test_ids'],ids[te]) and np.array_equal(b['train_ids'],ids[tr]) and np.array_equal(b['test_ids'],ids[te]) and np.array_equal(a['y'],d['y'][tr]) and np.array_equal(b['ytrain'],d['y'][tr]);assert np.array_equal(a['X'][:,-1],d['F'][tr,0]) and np.array_equal(a['T'][:,-1],d['F'][te,0]);A=[a['X'][:,:-1],b['oof_matrix']];T=[a['T'][:,:-1],b['test_matrix']];names=[f'round07base{i}' for i in range(A[0].shape[1])]+[f'round08base{i}' for i in range(8)];proof={str(p7):sha(p7),str(p8):sha(p8)}
    manual_receipt=read(G/'manual-controls/receipt.json');wave_receipt=read(G/'norm-aux/receipt.json');fallback=None;manuals=['speed_tail4','speed_tail16','speed_aux_tail4']
    if cond!='speed':manuals+=['speed_duration_tail4','speed_duration_aux_tail4']
    for c in manuals:
        x,z,q=neural(GB/'manual-controls',c,f,ids[tr],ids[te],manual_receipt);A.append(x[:,None]);T.append(z[:,None]);names.append('manual/'+c);proof.update(q)
        if c==('speed_tail4' if cond=='speed' else 'speed_duration_tail4'):fallback=z
    if cond=='combined':
        passed=read(G/'norm-aux/verification.json')['passed_conditions'];assert set(WAVES)<=set(passed)
        for c in WAVES:
            x,z,q=neural(GB/'norm-aux',c,f,ids[tr],ids[te],wave_receipt);A.append(x[:,None]);T.append(z[:,None]);names.append('wave/'+c);proof.update(q)
            if c=='norm_tail4':fallback=z
    X=np.column_stack(A);Z=np.column_stack(T);assert np.isfinite(X).all() and np.isfinite(Z).all() and fallback is not None;return tr,te,X,Z,np.array(fallback),names,proof
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);v=read(G/'norm-aux/verification.json');assert set(WAVES)<=set(v['passed_conditions']) and v['failed_conditions']==['symnorm_aux_tail4'];assert all(read(H/s/'verification.json')['status']=='passed' for s in ['round07','round08']);assert read(G/'manual-controls/verification.json')['status']=='passed';p=dict(round=11,created_unix=time.time(),code_sha256=sha(Path(__file__)),data_sha256=sha(GB/'data.npz'),source_verification_hashes={str(path):sha(path) for path in [H/'round07/verification.json',H/'round08/verification.json',G/'norm-aux/verification.json',G/'manual-controls/verification.json']},conditions=CONDS,tail_weights=TAILS,alpha=10,design='Same594/outer5 andentiretraining-onlyinnerOOFsourceexperts. Fulluses17rawtargetbases+8speedadjustedtargetbases+5manualbases+4passedwavebases;speed19bases,speed_duration21bases. Rawmodeltargetexpertsfromround07,round08andverifiedCPUstudies. Failedauxiliarywaveconditionneverused. OnlytrainOOFvaluesfitmeta,onlyoutertrainingrowsfitnormalizers/weights. Morefamiliesinfull,so wholepipelinecomparisonnotisolatedequalensemble-sizecausality.',guard='Computeobservedspeedmin/maxfromoutertrainingpeopleonly. If testpersonspeedstrictlyoutside[min,max],guardedpolicyusesfixedreviewedfallback:manualspeed_tail4/speed_duration_tail4/orderednorm_tail4for3conditions. Otherwiseusesmetafusion. Unrestrictedpolicyusesmetafusionforall. Samefittedmetaweightsinbothpolicies. All594retained;nooutput/targetclipping. Thisdoesnotprovevelocitymeasurementerror;rawsignalsarestillvalidandfallbackhypothesisistested.',models='FixedpositiveRidgealpha10withtail1/4/8/16,middle sqrt(tail),else1;trainingstandardizationandweightedcentering,NNLS. All24fixedcondition/tail/guardpoliciesreported,noouter-basedhyperparameterchoiceandnofittedgateusingtestlabels. Sourceepochsselectedwithinoutertrainingpeopleaspreviouslyfrozen. Metaweightsfitalloutertrainingrows,norowsexcluded.',verification='60savedmetamodels,15sourcecontexts,120policyfoldoutputs;independentrebuildsourceOOF/IDs/currentoriginalreceiptshashes,train-onlyscalers/weights/KKT,bounds/fallbackandmetrics/paireduncertainty. Source75+25wave/modelcheckpointscoveredbycondition-specificfullreviews. Failedauxiliaryconditionexcludedasmodel,notexcludingpeople. Reuseddevelopmentfolds/severe9/singleinit,notclinicalexternal.',expected_additional_GB=.02)
    if(R/'protocol.json').exists():
        old=read(R/'protocol.json');p['created_unix']=old['created_unix'];assert old==p
    else:save(R/'protocol.json',p)
    d=np.load(GB/'data.npz')
    for c in CONDS:
      for f in range(5):
        path=O/f'{c}-{f}.json'
        if path.exists():continue
        tr,te,X,Z,fallback,names,proof=sources(c,f,d);speed=d['F'][:,0];lo=float(speed[tr].min());hi=float(speed[tr].max());flag=(speed[te]<lo)|(speed[te]>hi);mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1;A=(X-mu)/sd;T=(Z-mu)/sd;y=d['y'][tr];ctx=O/f'{c}-{f}-context.npz';np.savez_compressed(ctx,X=X,T=Z,train_ids=d['ids'][tr],test_ids=d['ids'][te],y=y,train_speed=speed[tr],test_speed=speed[te],fallback=fallback,guard_flags=flag);models=[]
        for tail in TAILS:
            w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));am=np.average(A,axis=0,weights=w);ym=np.average(y,weights=w);coef,_=nnls(np.vstack([(A-am)*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(A.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(A.shape[1])],maxiter=10000);intercept=ym-am@coef;pred=T@coef+intercept;guarded=np.where(flag,fallback,pred);model=O/f'{c}-{f}-tail{tail}.npz';np.savez_compressed(model,mean=mu,std=sd,coef=coef,intercept=intercept,weight=w,speed_min=lo,speed_max=hi);models.append(dict(tail=tail,path=str(model),sha256=sha(model),prediction=pred.tolist(),guarded_prediction=guarded.tolist()))
        save(path,dict(condition=c,fold=f,models=models,context=str(ctx),context_sha256=sha(ctx),source_artifacts=proof,feature_names=names,protocol_sha256=sha(R/'protocol.json')));print(c,f,int(flag.sum()),flush=True)
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',metamodels=60,contexts=15,policy_fold_outputs=120,protocol_sha256=sha(R/'protocol.json'),hashes={str(path):sha(path) for path in O.glob('*') if path.is_file()}))
if __name__=='__main__':main()
