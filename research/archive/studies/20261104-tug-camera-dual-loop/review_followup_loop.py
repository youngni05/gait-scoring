"""Replay follow-up bags, safe teacher/student targets and actual video kernels."""
from pathlib import Path
import json,sys
import numpy as np,joblib
from scipy.special import expit
import review_dual_loop as rev
H=rev.H;B=rev.B;read=rev.read;sha=rev.sha;save=rev.save
def bagged():
    rec=read(H/'sensor_round3-receipt.json');assert rec['protocol_sha256']==sha(H/'followup-protocol.json');data=np.load(rev.G/'data.npz');native=np.load(rev.G/'native-motion/data.npz');outputs={};count=0;rng=np.random.default_rng(20261110)
    for r in rec['records']:
        f=r['fold'];tr=np.flatnonzero(data['fold']!=f);te=np.flatnonzero(data['fold']==f);assert r['train_ids']==data['ids'][tr].tolist() and r['test_ids']==data['ids'][te].tolist();assert sha(r['source'])==r['source_sha256'] and sha(r['context'])==r['context_sha256'];c=np.load(r['context']);assert np.array_equal(c['train_ids'],data['ids'][tr]);A=c['X'];T=c['T'];y=data['y'][tr];groups=[np.flatnonzero(np.digitize(y,[12,20])==j) for j in range(3)];samples=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(16)];raw=np.column_stack([A[:,31],A[:,15],np.log1p(A.std(1)),np.log(c['train_speed']),native['F'][tr,1]]);test=np.column_stack([T[:,31],T[:,15],np.log1p(T.std(1)),np.log(c['test_speed']),native['F'][te,1]]);arr=[]
        for row,ii in zip(r['models'],samples):
            assert sha(row['path'])==row['sha256'];m=joblib.load(row['path']);assert np.array_equal(ii,m['sample']);X=A[ii];yy=y[ii]
            for expert,tail in [(m['low'],1),(m['high'],16)]:
                assert np.array_equal(expert['mean'],X.mean(0));sd=X.std(0);sd[sd<1e-8]=1;assert np.array_equal(sd,expert['std']);Z=(X-expert['mean'])/sd;w=np.where(yy>=20,tail,np.where(yy>=12,np.sqrt(tail),1.));err=rev.lin(expert,X)-yy;g=Z.T@(w*err)+10*expert['coef'];on=expert['coef']>1e-7;assert (abs(g[on]).max() if on.any() else 0)<1e-6 and (np.maximum(-g[~on],0).max() if (~on).any() else 0)<1e-6 and abs(np.sum(w*err))<1e-7
            L=rev.lin(m['low'],X);U=rev.lin(m['high'],X);Lt=rev.lin(m['low'],T);Ut=rev.lin(m['high'],T);assert np.array_equal(raw[ii].mean(0),m['mean']);sd=raw[ii].std(0);sd[sd<1e-8]=1;assert np.array_equal(sd,m['std']);P=(raw[ii]-m['mean'])/sd;Q=(test-m['mean'])/sd;theta=m['theta'];z=P@theta[:-1]+theta[-1];g=expit(z);e=L+g*(U-L)-yy;target=(yy>=20).astype(float);pw=(len(yy)-target.sum())/target.sum();obj=np.mean(np.where(abs(e)<1,.5*e**2,abs(e)-.5))+.1*np.mean((1-target)*np.logaddexp(0,z)+pw*target*np.logaddexp(0,-z))+.01*np.sum(theta[:-1]**2);dz=(np.clip(e,-1,1)*(U-L)*g*(1-g)+.1*((1-target)*g-pw*target*(1-g)))/len(yy);grad=np.r_[P.T@dz+.02*theta[:-1],dz.sum()];assert abs(obj-m['objective'])<1e-10 and np.allclose(grad,m['gradient'],atol=1e-8,rtol=1e-8);p=Lt+expit(Q@theta[:-1]+theta[-1])*(Ut-Lt);lo=c['train_speed'][ii].min();hi=c['train_speed'][ii].max();p=np.where((c['test_speed']<lo)|(c['test_speed']>hi),c['fallback'],p);rev.close(p,row['prediction']);arr.append(p);count+=1
        arr=np.array(arr);orig=next(x for x in read(rev.OB/'round12'/f'combined-{f}.json')['models'] if x['index']==1)['guarded_prediction'];policies=dict(bag_mean=arr.mean(0),bag_median=np.median(arr,axis=0),half_original_half_bagmean=.5*np.array(orig)+.5*arr.mean(0))
        for key,p in policies.items():rev.close(p,r['policies'][key]);outputs.setdefault(key,np.full(len(y) if False else 594,np.nan))[te]=p
    baseline=np.array(read(B/'heldout.json')['sensor']['predictions']['baseline/original_0.881']);return data['ids'],data['y'],outputs,dict(original_0_881=baseline),dict(bags_replayed=count,expert_models=2*count,gates=count)
def students():
    rec=read(H/'proxy_round2-receipt.json');assert rec['protocol_sha256']==sha(H/'followup-protocol.json');data=np.load(rev.G/'data.npz');out={};zero={};count=0;checks=[];source_receipt=read(rev.OLD/'round12/run-receipt.json')
    for r in rec['records']:
        f=r['fold'];tr=np.flatnonzero(data['fold']!=f);te=np.flatnonzero(data['fold']==f);assert r['train_ids']==data['ids'][tr].tolist() and r['test_ids']==data['ids'][te].tolist() and not set(r['train_ids'])&set(r['test_ids']);assert sha(r['source_data'])==r['source_data_sha256'];d=np.load(r['source_data']);assert np.array_equal(d['ids'],data['ids']);t=r['teacher'];assert sha(t['source'])==source_receipt['hashes'][t['source']]==t['source_sha256'] and sha(t['context'])==t['context_sha256'] and sha(t['model'])==t['model_sha256'];ctx=np.load(t['context']);net=np.load(t['model']);assert np.array_equal(ctx['train_ids'],data['ids'][tr]) and np.array_equal(ctx['test_ids'],data['ids'][te]);soft=ctx['low_train']+expit(((ctx['raw_phi']-net['mean'])/net['std'])@net['gate_weight']+net['gate_bias'])*(ctx['high_train']-ctx['low_train']);rev.close(soft,r['teacher_train_prediction']);target=(1-r['teacher_weight'])*d['y'][tr]+r['teacher_weight']*soft;rev.close(target,r['training_target']);assert sha(r['model'])==r['model_sha256'];m=joblib.load(r['model']);rev.prepcheck(m,d['X'][tr]);assert m['config']==dict(kind='svr',C=r['C'],transform='direct');p=rev.predict(m,d['X'][te]);rev.close(p,r['prediction']);key=f"{r['condition']}/C{r['C']}/w{r['teacher_weight']}";out.setdefault(key,np.full(594,np.nan))[te]=p;count+=1
    # All zero-teacher settings are matched controls; do not choose them by test outcomes.
    baseline={k:v for k,v in out.items() if k.endswith('/w0')}
    return data['ids'],data['y'],out,baseline,dict(students_replayed=count,teacher_outertrain_only=True,global_OOF_teacher_never_used_as_training_targets=True)
def audit_motion_features():
    d=np.load(rev.MB/'bottom14-inputs.npz');r=read(H/'camera-motion-kernels.json');assert sha(rev.MB/'bottom14-inputs.npz')==r['source_sha256'];features=[]
    for ix in r['first_window_indices']:
        x=d['pelvis_stable_xy'][ix];m=d['pelvis_mask'][ix];f=[]
        for spec in r['weights']:
            taps=np.array(spec['taps']);js=np.array(spec['channels']);w=np.array(spec['weight']);starts=np.arange(len(x)-taps[-1]);ixs=starts[:,None]+taps;xx=x[ixs[:,:,None],js[None,None,:]];valid=m[ixs[:,:,None],(js//2)[None,None,:]].all((1,2));response=(xx*w).sum((1,2))+spec['bias'];a=response[valid];f.extend([float(np.max(a)),float(np.mean(abs(a))),float(np.mean(a>0))] if len(a)>2 else [np.nan]*3)
        features.append(f)
    F=np.array(features);actual=np.load(B/'camera_round2/motion_kernels-data.npz');assert np.array_equal(F,actual['X'],equal_nan=True);mechanics=np.load(B/'camera_round1/mechanics-data.npz')['X'];flow=np.load(B/'camera_round1/optical_flow-data.npz')['X'];assert np.array_equal(np.column_stack([mechanics,F]),np.load(B/'camera_round2/mechanics_motion-data.npz')['X'],equal_nan=True);assert np.array_equal(np.column_stack([F,flow]),np.load(B/'camera_round2/motion_flow-data.npz')['X'],equal_nan=True)
    return dict(people=14,kernels=64,independent_vectorized_feature_replay=True,no_label_based_kernel_fit=True)
def main():
    proto=read(H/'followup-protocol.json');assert proto['code_sha256']==sha(H/'followup_loop.py') and proto['library_sha256']==sha(H/'dual_loop.py') and proto['parent_verification_sha256']==sha(H/'verification.json');parent=read(H/'protocol.json');pc=dict(parent);pc['camera_round2']=parent['camera_round1'];audits={};summary={};held={};rng=np.random.default_rng(20261111)
    for track in ['sensor_round3','proxy_round2','camera_round2']:
        if track=='sensor_round3':ids,y,out,baseline,audit=bagged()
        elif track=='proxy_round2':ids,y,out,baseline,audit=students()
        else:feature_audit=audit_motion_features();ids,y,out,baseline,audit=rev.nested(track,pc);audit['motion_features']=feature_audit
        boot=rng.integers(0,len(y),(5000,len(y)));rows={}
        for key,p in {**{'baseline/'+k:v for k,v in baseline.items()},**out}.items():
            assert np.isfinite(p).all();v=rev.metrics(y,p);error=abs(p-y);v['mae_ci95']=np.quantile(error[boot].mean(1),[.025,.975]).tolist();v['comparisons']={}
            targets=baseline if track!='proxy_round2' else {k:ref for k,ref in baseline.items() if k.split('/')[0]==key.removeprefix('baseline/').split('/')[0] and k.split('/')[1]==key.removeprefix('baseline/').split('/')[1]}
            for name,ref in targets.items():diff=error-abs(ref-y);v['comparisons'][name]=dict(mae_difference=float(diff.mean()),ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
            rows[key]=v
        summary[track]=rows;audits[track]=audit;held[track]=dict(ids=ids.tolist(),y=y.tolist(),predictions={**{('baseline/'+k):v.tolist() for k,v in baseline.items()},**{k:v.tolist() for k,v in out.items()}});print(track,'passed',flush=True)
    save(B/'followup-heldout.json',held);save(H/'followup-comparison.json',summary);save(H/'followup-verification.json',dict(status='passed',audits=audits,protocol_sha256=sha(H/'followup-protocol.json'),review_code_sha256=sha(__file__),comparison_sha256=sha(H/'followup-comparison.json'),heldout_sha256=sha(B/'followup-heldout.json'),limits='Same reuseddevelopmentcohorts,conditionalpairedperson95percentCI,notmultiplicityadjusted. Allrecipesreported. No phoneclinicalperformanceclaim.'))
if __name__=='__main__':main()
