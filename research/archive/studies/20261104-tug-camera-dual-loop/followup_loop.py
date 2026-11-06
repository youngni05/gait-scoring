"""Predeclared follow-up: bagging, leakage-safe teacher targets, temporal kernels."""
from pathlib import Path
import sys,argparse,time
import dual_loop as lib
import numpy as np,joblib
from scipy.special import expit
from scipy.optimize import minimize
H=lib.H;B=lib.B;read=lib.read;sha=lib.sha;save=lib.save
def freeze():
    p=dict(created_unix=time.time(),code_sha256=sha(__file__),library_sha256=sha(H/'dual_loop.py'),parent_verification_sha256=sha(H/'verification.json'),rationale='Initial loop: rawmeanMAEimprovement hurt9slowpeople; feature-only candidates failed strongspeed; video gains uncertain. Follow-up tests stability/teachertransfer/temporalrepresentation,notouteroptimizedthresholds.',sensor_round3=dict(bags=16,seed=20261110,recipe='PerouterfoldstratifiedbootstrapofoutertrainingpeoplewithinTUGbands<12/12..20/>=20. RefitpositiveRidgetail1and16expertsandCPU64sigmoidgate5featuresbeta1,aux.1,L2.01 foreachbag. Same34sourceOOFcolumns/speed/duration. Per-baginputsupportguardbasedonbagtrainmin/max,samewavefallback. Reportmean/medianandfixedhalfblendwithoriginalmodel. Nobagselectedbyouterperformance.'),proxy_round2=dict(conditions=['speed','core3','timing7','metric12','availability5'],C=[1,10,100],teacher_weight=[0,.25,.5,.75],recipe='FixedRBFstudentSVRepsilon.15,gamma1/d. Trainingtarget=(1-lambda)*trueTUG+lambda*originalteacherpredictionONOUTERTRAIN. TeacherisoriginalR12config1fittedonlyonthisoutertrainingcohort. NeveruseglobalOOFteacherlabelsforstudenttraining:thoseotherouterfoldmodelswouldincludeheldoutpeople. No studentinnerhyperparameterselection. AllC/lambda reported. Camera-observablequantitiesaresensor-derivedproxies;nodistillationfromunpairedactualvideo.'),camera_round2=dict(conditions=['motion_kernels','mechanics_motion','motion_flow'],kernels=64,seed=20261110,recipe='Outcome-blindfixedrandomtemporalconvolutionsofsame120x24pelviscenteredscaled2Dpose. Eachkernel4channels,3/5/7taps,dilation1/2/4/8,zero-meanL2normalizedweights,fixednormalbias. max/meanabs/positivefraction overonlywindowsallselectedpointsvalid. Maskgapnotbridged;missingfeaturesremainNaN. Fixedkernelsnotlearnedfrom14labels. NestedLOOperpersonwithsame5Ridge/SVRcandidates;noadditionalobservationtime.'),limits='Post-initial-loop exploratory recipes and reusedpeople. Everycondition reported. Baggingdoesnotcreateindependentpeople. Teacherin-sampletrainingactivationsallowedasstudenttargets,neverreportedaspredictionaccuracy. Camera14/oneTUG>=20;no clinical orphonevalidatedclaim.')
    if(H/'followup-protocol.json').exists():old=read(H/'followup-protocol.json');p['created_unix']=old['created_unix'];assert p==old
    else:save(H/'followup-protocol.json',p)
    return p
def teacher(fold,data):
    source=lib.OB/'round12'/f'combined-{fold}.json';r=read(source);receipt=read(lib.OLD/'round12/run-receipt.json');assert sha(source)==receipt['hashes'][str(source)];ctx=np.load(r['context']);mrec=next(x for x in r['models'] if x['index']==1);assert sha(mrec['path'])==mrec['sha256'];m=np.load(mrec['path']);z=(ctx['raw_phi']-m['mean'])/m['std'];p=ctx['low_train']+expit(z@m['gate_weight']+m['gate_bias'])*(ctx['high_train']-ctx['low_train']);tr=np.flatnonzero(data['fold']!=fold);assert np.array_equal(ctx['train_ids'],data['ids'][tr]);return p,dict(source=str(source),source_sha256=sha(source),context=r['context'],context_sha256=sha(r['context']),model=mrec['path'],model_sha256=sha(mrec['path']))
def bagged():
    p=freeze();d=np.load(lib.G/'data.npz');native=np.load(lib.G/'native-motion/data.npz');records=[];rng=np.random.default_rng(p['sensor_round3']['seed'])
    for fold in range(5):
        root=B/'sensor_round3'/f'fold{fold}';root.mkdir(parents=True,exist_ok=True);dest=root/'fold.json'
        # Draw every fold's samples even if completed, so resume RNG stays identical.
        tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold);groups=[np.flatnonzero(np.digitize(d['y'][tr],[12,20])==j) for j in range(3)];samples=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(16)]
        if dest.exists():records.append(read(dest));continue
        src=lib.OB/'round11'/f'combined-{fold}.json';sr=read(src);assert sha(sr['context'])==sr['context_sha256'];c=np.load(sr['context']);A=c['X'];T=c['T'];y=d['y'][tr];raw=np.column_stack([A[:,31],A[:,15],np.log1p(A.std(1)),np.log(c['train_speed']),native['F'][tr,1]]);test=np.column_stack([T[:,31],T[:,15],np.log1p(T.std(1)),np.log(c['test_speed']),native['F'][te,1]]);models=[];allp=[]
        for b,ii in enumerate(samples):
            low=lib.positive(A[ii],y[ii],1);high=lib.positive(A[ii],y[ii],16);L=lib.line(low,A[ii]);U=lib.line(high,A[ii]);Lt=lib.line(low,T);Ut=lib.line(high,T);mu=raw[ii].mean(0);sd=raw[ii].std(0);sd[sd<1e-8]=1;P=(raw[ii]-mu)/sd;Q=(test-mu)/sd;yy=y[ii];target=(yy>=20).astype(float);pw=(len(yy)-target.sum())/target.sum()
            def obj(theta):
                z=P@theta[:-1]+theta[-1];g=expit(z);err=L+g*(U-L)-yy;h=np.where(abs(err)<1,.5*err**2,abs(err)-.5);ce=(1-target)*np.logaddexp(0,z)+pw*target*np.logaddexp(0,-z);v=np.mean(h)+.1*np.mean(ce)+.01*np.sum(theta[:-1]**2);dz=(np.clip(err,-1,1)*(U-L)*g*(1-g)+.1*((1-target)*g-pw*target*(1-g)))/len(yy);return v,np.r_[P.T@dz+.02*theta[:-1],dz.sum()]
            result=minimize(obj,np.r_[np.zeros(5),-2.],jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8));assert np.isfinite(result.fun);pr=Lt+expit(Q@result.x[:-1]+result.x[-1])*(Ut-Lt);lo=c['train_speed'][ii].min();hi=c['train_speed'][ii].max();guard=(c['test_speed']<lo)|(c['test_speed']>hi);pr=np.where(guard,c['fallback'],pr);model=dict(low=low,high=high,mean=mu,std=sd,theta=result.x,sample=ii,objective=float(result.fun),gradient=obj(result.x)[1]);path=root/f'bag{b}.joblib';joblib.dump(model,path);models.append(dict(path=str(path),sha256=sha(path),prediction=pr.tolist()));allp.append(pr)
        orig=next(x for x in read(lib.OB/'round12'/f'combined-{fold}.json')['models'] if x['index']==1)['guarded_prediction'];arr=np.array(allp);policies=dict(bag_mean=arr.mean(0).tolist(),bag_median=np.median(arr,axis=0).tolist(),half_original_half_bagmean=(.5*np.array(orig)+.5*arr.mean(0)).tolist());r=dict(fold=fold,source=str(src),source_sha256=sha(src),context=sr['context'],context_sha256=sha(sr['context']),train_ids=d['ids'][tr].tolist(),test_ids=d['ids'][te].tolist(),models=models,policies=policies);save(dest,r);records.append(r);print('bagged',fold,flush=True)
    save(H/'sensor_round3-receipt.json',dict(records=records,status='complete_pending_review',protocol_sha256=sha(H/'followup-protocol.json')))
def distill():
    p=freeze();d=np.load(lib.G/'data.npz');records=[]
    for condition in p['proxy_round2']['conditions']:
        src=B/'proxy_round1'/(condition+'-data.npz');data=np.load(src);X=data['X'];y=data['y'];sp=data['speed'];assert np.array_equal(data['ids'],d['ids'])
        for fold in range(5):
            tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold);soft,proof=teacher(fold,d)
            for C in [1,10,100]:
                for weight in [0,.25,.5,.75]:
                    root=B/'proxy_round2';root.mkdir(exist_ok=True);dest=root/f'{condition}-{fold}-C{C}-w{weight}.json'
                    if dest.exists():records.append(read(dest));continue
                    target=(1-weight)*y[tr]+weight*soft;m=lib.fit_simple(X[tr],target,dict(kind='svr',C=C,transform='direct'));pr=lib.simple_predict(m,X[te]);path=dest.with_suffix('.joblib');joblib.dump(m,path);r=dict(condition=condition,fold=fold,C=C,teacher_weight=weight,source_data=str(src),source_data_sha256=sha(src),teacher=proof,teacher_train_prediction=soft.tolist(),training_target=target.tolist(),train_ids=d['ids'][tr].tolist(),test_ids=d['ids'][te].tolist(),model=str(path),model_sha256=sha(path),prediction=pr.tolist());save(dest,r);records.append(r)
            print('student',condition,fold,flush=True)
    save(H/'proxy_round2-receipt.json',dict(records=records,status='complete_pending_review',protocol_sha256=sha(H/'followup-protocol.json')))
def kernels(x,mask,weights):
    result=[]
    for k in weights:
        j=np.array(k['channels']);taps=np.array(k['taps']);w=np.array(k['weight']);a=[];okay=[]
        for i in range(len(x)-taps[-1]):
            window=x[i+taps][:,j];valid=mask[i+taps][:,j//2];a.append(float(np.sum(window*w)+k['bias']));okay.append(bool(valid.all()))
        a=np.array(a)[np.array(okay)];result.extend([float(a.max()),float(np.mean(abs(a))),float(np.mean(a>0))] if len(a)>2 else [np.nan]*3)
    return result
def camera_motion():
    p=freeze();d=np.load(lib.MB/'bottom14-inputs.npz');ids=np.array(sorted(set(d['person_ids'])));ix=np.array([np.flatnonzero(d['person_ids']==i)[0] for i in ids]);rng=np.random.default_rng(p['camera_round2']['seed']);weights=[]
    for j in range(64):
        channels=rng.choice(24,4,replace=False);n=int(rng.choice([3,5,7]));dilation=int(rng.choice([1,2,4,8]));w=rng.normal(size=(n,4));w-=w.mean(0);w/=np.linalg.norm(w);weights.append(dict(channels=channels.tolist(),taps=(np.arange(n)*dilation).tolist(),weight=w.tolist(),bias=float(rng.normal(scale=.1))))
    save(H/'camera-motion-kernels.json',dict(weights=weights,source_sha256=sha(lib.MB/'bottom14-inputs.npz'),first_window_indices=ix.tolist(),labels_used=False));F=np.array([kernels(d['pelvis_stable_xy'][i],d['pelvis_mask'][i],weights) for i in ix]);mechanics=np.load(B/'camera_round1/mechanics-data.npz');flow=np.load(B/'camera_round1/optical_flow-data.npz');sets=dict(motion_kernels=F,mechanics_motion=np.column_stack([mechanics['X'],F]),motion_flow=np.column_stack([F,flow['X']]));base=read(H/'protocol.json');lib.nested('camera_round2',sets,mechanics['y'],ids,np.arange(14),None,base['camera_round1']['candidates'],balanced=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','bagged','distill','camera']);args=ap.parse_args();{'freeze':freeze,'bagged':bagged,'distill':distill,'camera':camera_motion}[args.mode]()
