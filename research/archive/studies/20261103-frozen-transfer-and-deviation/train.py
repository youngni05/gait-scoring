from prepare import *
ALPHAS=np.array([.1,1,10,100,1000.])
def folds(indices,people,k,seed):
 out=np.full(len(people),-1,int);rng=np.random.default_rng(seed)
 for g in sorted(set((people[i]['cohort'],people[i]['age']>=65) for i in indices)):
  ids=np.array([i for i in indices if (people[i]['cohort'],people[i]['age']>=65)==g]);rng.shuffle(ids);out[ids]=np.arange(len(ids))%k
 return out
def targets(train,query,curves,people,older):
 ref=np.array([i for i in train if people[i]['cohort']=='healthy' and (people[i]['age']>=65 or not older)])
 assert len(ref)>=3
 normals=curves[ref].mean(axis=1);total=normals.sum(0);values=[]
 for i in query:
  mean=(total-curves[i].mean(0))/(len(ref)-1) if i in ref else total/len(ref)
  values.append(np.sqrt(((curves[i]-mean)**2).mean(axis=(1,2))).mean())
 return np.array(values),ref
def ridge(x,y,q,alpha):
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1;z=(x-mean)/scale;ym=y.mean();u,s,vt=np.linalg.svd(z,full_matrices=False);w=vt.T@((s/(s*s+alpha))*(u.T@(y-ym)))
 return (q-mean)/scale@w+ym,dict(mean=mean,scale=scale,weight=w,intercept=np.array(ym))
def select(train,x,curves,people,older,seed):
 inner=folds(train,people,3,seed);error=np.zeros(5);count=0;audit=[]
 for j in range(3):
  tr=train[inner[train]!=j];va=train[inner[train]==j];assert not set(tr)&set(va)
  yt,ref=targets(tr,tr,curves,people,older);yv,_=targets(tr,va,curves,people,older);assert not set(ref)&set(va)
  for a,alpha in enumerate(ALPHAS):pred,_=ridge(x[tr],yt,x[va],alpha);error[a]+=np.abs(pred-yv).sum()
  count+=len(va);audit.append(dict(train=tr.tolist(),validation=va.tolist(),reference=ref.tolist()))
 return float(ALPHAS[np.argmin(error)]),(error/count).tolist(),audit

def main():
 prep=read(HERE/'preparation.json');assert sha(HERE/'protocol.json')==prep['protocol_sha256']
 people=read(HERE/'people.json');z=np.load(BIG/'features.npz');x=z['person_features'];ids=z['person_id'];archive=np.load('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/person-curves.npz')
 curves=np.stack([archive[p['cohort']][p['mat_index']-1,:,:,::20] for p in people]);assert curves.shape==(len(people),2,9,51) and np.isfinite(curves).all();allidx=np.arange(len(people));outer=folds(allidx,people,5,20261103);report={};arrays={'person_id':ids,'fold':outer}
 (BIG/'new-model').mkdir(exist_ok=True)
 for name,older in [('age65plus',True),('allhealthy',False)]:
  pred=np.full(len(people),np.nan);truth=pred.copy();baseline=pred.copy();audit=[]
  for f in range(5):
   tr=allidx[outer!=f];te=allidx[outer==f];alpha,errors,inner=select(tr,x,curves,people,older,20261103+f)
   yt,ref=targets(tr,tr,curves,people,older);yv,_=targets(tr,te,curves,people,older);assert not set(ref)&set(te)
   pp,model=ridge(x[tr],yt,x[te],alpha);pred[te]=pp;truth[te]=yv;baseline[te]=yt.mean()
   file=BIG/'new-model'/f'{name}-fold{f}.npz';np.savez_compressed(file,**model,train=tr,test=te,reference=ref,reference_curve=curves[ref].mean(axis=(0,1)),alpha=alpha)
   audit.append(dict(fold=f,train=tr.tolist(),test=te.tolist(),reference=ref.tolist(),alpha=alpha,inner_mae=errors,inner=inner,checkpoint=str(file),sha256=sha(file),test_mae=float(np.abs(pp-yv).mean())))
   print(name,'fold',f,'n',len(te),'MAE',round(float(np.abs(pp-yv).mean()),3),flush=True)
  assert np.isfinite(pred).all();arrays[name+'_prediction']=pred;arrays[name+'_truth']=truth;arrays[name+'_baseline']=baseline
  alpha,errors,inner=select(allidx,x,curves,people,older,20261133);yt,ref=targets(allidx,allidx,curves,people,older);_,model=ridge(x,yt,x,alpha);np.savez_compressed(BIG/'new-model'/f'{name}-deployment.npz',**model,reference=ref,reference_curve=curves[ref].mean(axis=(0,1)),alpha=alpha)
  report[name]=dict(n=len(people),mae_degrees=float(np.abs(pred-truth).mean()),baseline_mae_degrees=float(np.abs(baseline-truth).mean()),folds=audit,deployment_alpha=alpha,deployment_reference_n=len(ref),deployment_inner_mae=errors)
 np.savez_compressed(BIG/'new-model-oof.npz',**arrays)
 save(HERE/'new-model-results.json',dict(protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),curves_sha256=sha('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/person-curves.npz'),predictions_sha256=sha(BIG/'new-model-oof.npz'),models=report,source='Synthetic side-view common joints',not_original_GDI=True))
if __name__=='__main__':main()
