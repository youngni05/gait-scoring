"""Training-only speed-adjusted targets with fixed foundation/classical learners."""
from pathlib import Path
import sys,time
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(H))
from round3 import read,save,sha,np,joblib,fit2,predict2,metrics,StratifiedKFold
from round5 import fit_meta,predict_meta
B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'round08';O=B/'round08'
CONFIGS=[dict(target=t,kind=k,k=32,**({'alpha':10} if k=='ridge' else {'C':100,'gamma':.03} if k=='svr' else {})) for t in ['speed_product','inverse_speed_residual'] for k in ['ridge','svr','TabICLv2','TabPFNv2']]
def transformed_fit(X,y,cfg):
    speed=X[:,0];assert np.isfinite(speed).all() and np.all(speed>0)
    if cfg['target']=='speed_product':target=y*speed;coef=None
    else:coef=np.linalg.lstsq(np.column_stack([np.ones(len(y)),1/speed]),y,rcond=None)[0];target=y-np.column_stack([np.ones(len(y)),1/speed])@coef
    return dict(base=fit2(X,target,cfg),transform=cfg['target'],coefficient=coef)
def transformed_predict(m,X):
    p=predict2(m['base'],X);speed=X[:,0]
    return p/speed if m['transform']=='speed_product' else p+np.column_stack([np.ones(len(X)),1/speed])@m['coefficient']
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);d=np.load(GB/'data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];F=d['F'];proto=dict(round=8,code_sha256=sha(Path(__file__)),data_sha256=sha(GB/'data.npz'),library_sha256=sha(H/'round3.py'),meta_library_sha256=sha(H/'round5.py'),factory_sha256=sha(Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk/models.py')),conditions=['speed','bilateral'],candidates=CONFIGS,design='Same594/frozenouter5/inner3. Compare two fixed target transforms: actualTUG*observedspeed, invert by sameobservedspeed; or actualTUG minus training-onlyOLS[1,1/speed],addbaselinebackatinference. No clinical covariates/output clipping. Speedproduct is mathematical target,not literal walked distance. Positive measuredspeed asserted,noneexcluded. Train-only featureselection usestransformedtarget, speedretained. All8candidateinnerfits saved; all8finalmodelsrefit. OriginalTUGseconds usedforallevaluation/selection.',selection='AggregateinnerOOF inoutertrainorder; minimize max(MAE/.9,p95/3,severeMAE/3),firsttie. Report selected/equaltop3 plusfixedpositiveRidgealpha10fusiononinnerOOF withtail4and16,middleweight sqrt(tail),allother1;nooutertuning. No meta hyperparametersearch.',verification='240inner+80final+20metamodel replay; train-onlymedian/ranking/transformcoefficient verified, exactpersonpartitions/selection;NNLSKKT; originalmetricsandreplaypointtolerances unchanged;5000pairedpersonbootstrap. Fixedglobalfoundationweights hashed viaoriginalfactory. Reuseddevelopmentfolds,notclinical/external.',expected_additional_GB=.5)
    if(R/'protocol.json').exists():assert read(R/'protocol.json')==proto
    else:save(R/'protocol.json',proto)
    done=0
    for cond in proto['conditions']:
      X=F[:,:1] if cond=='speed' else F
      for f in range(5):
        stem=f'{cond}-{f}';path=O/(stem+'.json')
        if path.exists():done+=1;continue
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261103).split(tr,np.digitize(y[tr],[12,20])));inner=[];scores=[];a=np.zeros((len(tr),8));z=np.zeros((len(te),8));models=[]
        for ci,cfg in enumerate(CONFIGS):
            start=time.perf_counter()
            for k,(it,iv) in enumerate(splits):
                model=transformed_fit(X[tr[it]],y[tr[it]],cfg);pr=transformed_predict(model,X[tr[iv]]);mp=O/f'{stem}-candidate{ci}-inner{k}.joblib';joblib.dump(model,mp);a[iv,ci]=pr;inner.append(dict(candidate=ci,inner=k,train_ids=ids[tr[it]].tolist(),test_ids=ids[tr[iv]].tolist(),prediction=pr.tolist(),model=str(mp),sha256=sha(mp)));save(O/(stem+'-partial-inner.json'),inner)
            scores.append(metrics(y[tr],a[:,ci]));model=transformed_fit(X[tr],y[tr],cfg);mp=O/f'{stem}-final{ci}.joblib';joblib.dump(model,mp);z[:,ci]=transformed_predict(model,X[te]);models.append(dict(candidate=ci,path=str(mp),sha256=sha(mp)));print(stem,ci,cfg['target'],cfg['kind'],round(time.perf_counter()-start,1),flush=True)
        top=np.argsort([s['score'] for s in scores],kind='stable')[:3];pred=dict(selected=z[:,top[0]].tolist(),ensemble=z[:,top].mean(1).tolist());meta=[]
        for tail in [4,16]:
            m=fit_meta(a,y[tr],tail);mp=O/f'{stem}-meta{tail}.joblib';joblib.dump(m,mp);pred[f'tail{tail}_stacked']=predict_meta(m,z).tolist();meta.append(dict(tail=tail,path=str(mp),sha256=sha(mp)))
        context=O/(stem+'-context.npz');np.savez_compressed(context,Xtrain=X[tr],Xtest=X[te],ytrain=y[tr],train_ids=ids[tr],test_ids=ids[te],oof_matrix=a,test_matrix=z);save(path,dict(condition=cond,fold=f,inner=inner,inner_scores=scores,top3=top.tolist(),models=models,meta=meta,predictions=pred,context=str(context),context_sha256=sha(context),protocol_sha256=sha(R/'protocol.json')));done+=1;save(R/'progress.json',dict(status='running',complete=done,expected=10,last=stem))
    save(R/'run-receipt.json',dict(status='complete_pending_independent_review',cases=10,protocol_sha256=sha(R/'protocol.json'),hashes={str(p):sha(p) for p in O.glob('*') if p.is_file()}))
if __name__=='__main__':main()
