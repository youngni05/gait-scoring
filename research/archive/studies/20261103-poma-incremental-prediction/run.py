"""Independent auxiliary deviation training, then nested person-held-out POMA."""
from prepare import *
import time
ALPHA=np.array([.01,.1,1,10,100.]);UP_ALPHA=np.array([.1,1,10,100,1000.])

def fit(x,y,alpha,block=None):
 mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
 if block is not None:
  scale[:block]*=np.sqrt(block)
  if x.shape[1]>block:scale[block:]*=np.sqrt(x.shape[1]-block)
 z=(x-mean)/scale;u,s,vt=np.linalg.svd(z,full_matrices=False)
 weight=vt.T@((s/(s*s+alpha))*(u.T@(y-y.mean())))
 return dict(mean=mean,scale=scale,weight=weight,intercept=np.array(y.mean()))
def predict(m,x,clip=False):
 p=(x-m['mean'])/m['scale']@m['weight']+m['intercept']
 return np.clip(p,0,28) if clip else p

def design(speed,extra,degree):return np.column_stack([*[speed**d for d in range(1,degree+1)],extra])

def nested(speed,extra,y,name,write=True):
 n=len(y);out=np.full(n,np.nan);audit=[];allidx=np.arange(n)
 for outer in range(n):
  tr=allidx[allidx!=outer];order=tr.copy();np.random.default_rng(20261103+outer).shuffle(order);fold=np.full(n,-1);fold[order]=np.arange(len(tr))%4
  errors=[];inner=[]
  for f in range(4):inner.append(dict(train=tr[fold[tr]!=f].tolist(),validation=tr[fold[tr]==f].tolist()))
  for degree in [1,2,3]:
   x=design(speed,extra,degree)
   for alpha in ALPHA:
    err=0
    for s in inner:
     ti=np.array(s['train']);vi=np.array(s['validation']);m=fit(x[ti],y[ti],alpha,degree);pp=predict(m,x[vi],True);err+=np.abs(pp-y[vi]).sum()
    errors.append(dict(degree=degree,alpha=float(alpha),mae=float(err/len(tr))))
  best=min(errors,key=lambda a:a['mae']);x=design(speed,extra,best['degree']);m=fit(x[tr],y[tr],best['alpha'],best['degree']);out[outer]=predict(m,x[[outer]],True)[0]
  record=dict(test=[outer],train=tr.tolist(),inner=inner,selected=best,selection=errors)
  if write:
   path=BIG/'models'/f'{name}-person{outer:02d}.npz';np.savez_compressed(path,**m,train=tr,test=[outer],degree=best['degree'],alpha=best['alpha']);record.update(checkpoint=str(path),sha256=sha(path))
  audit.append(record)
 return out,audit

def upstream(z,people,clinical_ids,older):
 ids=z['person_id'].tolist();aux=np.array([i for i,p in enumerate(ids) if p not in clinical_ids]);assert len(aux)==33
 linkage=read(BASE/'person-linkage.json');archive=np.load('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/person-curves.npz')
 normal={old.normid(p['supplement_candidate_ids'][0]):archive['healthy'][p['mat_index']-1,:,:,::20] for p in linkage['healthy']['rows'] if p['values']['Age']>=65 or not older}
 curves=np.stack([archive[p['cohort']][p['mat_index']-1,:,:,::20] for p in people]);x=z['common']
 def target(query,excluded):
  refs=sorted(set(normal)-set(excluded));total=np.stack([normal[p].mean(0) for p in refs]).sum(0);values=[]
  for i in query:
   p=ids[i];ref=(total-normal[p].mean(0))/(len(refs)-1) if p in refs else total/len(refs)
   values.append(np.sqrt(((curves[i]-ref)**2).mean(axis=(1,2))).mean())
  return np.array(values),refs
 fold=np.full(len(ids),-1);rng=np.random.default_rng(20261103)
 for group in ['healthy','stroke']:
  ix=np.array([i for i in aux if people[i]['cohort']==group]);rng.shuffle(ix);fold[ix]=np.arange(len(ix))%3
 errs=np.zeros(len(UP_ALPHA));audit=[];name='age65plus' if older else 'allhealthy'
 for f in range(3):
  tr=aux[fold[aux]!=f];va=aux[fold[aux]==f];excluded=[ids[i] for i in va]+clinical_ids;yt,ref=target(tr,excluded);yv,_=target(va,excluded)
  for j,a in enumerate(UP_ALPHA):errs[j]+=np.abs(predict(fit(x[tr],yt,a),x[va])-yv).sum()
  audit.append(dict(train=[ids[i] for i in tr],validation=[ids[i] for i in va],reference=ref))
 alpha=UP_ALPHA[np.argmin(errs)];yt,refs=target(aux,clinical_ids);m=fit(x[aux],yt,alpha);p=predict(m,x);truth,_=target(np.arange(len(ids)),clinical_ids)
 path=BIG/'models'/f'auxiliary-{name}.npz';np.savez_compressed(path,**m,train=aux,alpha=alpha,target=yt,reference=np.array(refs),reference_curve=np.stack([normal[p] for p in refs]).mean(axis=(0,1)))
 return p,truth,dict(train=[ids[i] for i in aux],excluded_clinical=clinical_ids,reference=refs,alpha=float(alpha),inner_mae=(errs/len(aux)).tolist(),inner=audit,checkpoint=str(path),sha256=sha(path))

def metrics(y,p):
 err=p-y;return dict(n=len(y),mae=float(np.abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),median_absolute_error=float(np.median(np.abs(err))),max_absolute_error=float(np.abs(err).max()),within2=int((np.abs(err)<=2).sum()),over5=int((np.abs(err)>5).sum()))

def main():
 t=time.time();prep=read(HERE/'preparation.json');assert prep['protocol_sha256']==sha(HERE/'protocol.json');assert prep['features_sha256']==sha(BIG/'matched-features.npz')
 (BIG/'models').mkdir(exist_ok=True);protocol=read(HERE/'protocol.json');ids=protocol['primary_ids'];z=np.load(BIG/'matched-features.npz');people=read(HERE/'people.json');ci=np.array([z['person_id'].tolist().index(p) for p in ids]);scores={p['id']:p for p in read(OLD/'scores.json')}
 speed=np.array([scores[p]['clinical']['speed']['mean_mps'] for p in ids]);y=np.array([scores[p]['clinical']['POMA'] for p in ids]);assert np.isfinite(speed).all() and np.isfinite(y).all()
 p,truth,a=upstream(z,people,ids,True);pall,_,b=upstream(z,people,ids,False);save(HERE/'upstream-audit.json',dict(age65plus=a,allhealthy=b))
 extras={'A_speed':np.empty((25,0)),'B_frozen_GDI':z['gdi'][ci,None],'C_predicted_deviation':p[ci,None],'D_joint2D':z['joint2D'][ci],'E_joint3D':z['joint3D'][ci],'F_measured_deviation':truth[ci,None],'C_allhealthy':pall[ci,None]}
 np.savez_compressed(BIG/'clinical-inputs.npz',person_id=np.array(ids),speed=speed,y=y,**extras)
 preds={};audits={}
 for name,extra in extras.items():
  preds[name],audits[name]=nested(speed,extra,y,name);print(name,'MAE',round(metrics(y,preds[name])['mae'],4),'elapsed',round(time.time()-t,1),flush=True)
 preds['median_baseline']=np.array([np.median(np.delete(y,i)) for i in range(len(y))])
 controls=[]
 for seed in range(20261113,20261118):
  permutation=np.random.default_rng(seed).permutation(len(y));pp,aa=nested(speed,extras['D_joint2D'][permutation],y,'shuffle',False);preds['shuffle_'+str(seed)]=pp
  controls.append(dict(seed=seed,permutation=permutation.tolist(),mae=metrics(y,pp)['mae'],nested=aa));print('shuffle',seed,round(controls[-1]['mae'],4),flush=True)
 np.savez_compressed(BIG/'poma-predictions.npz',person_id=np.array(ids),y=y,**preds);save(HERE/'fit-audit.json',audits);save(HERE/'shuffle-audit.json',controls)
 rng=np.random.default_rng(20261103);boot=rng.integers(0,len(y),(10000,len(y)));base=np.abs(preds['A_speed']-y);results={}
 for name in extras:
  p=preds[name];e=np.abs(p-y);delta=e-base;diff=delta[boot].mean(1);r=metrics(y,p);r.update(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),delta_mae_vs_speed=float(delta.mean()),delta_ci95=np.quantile(diff,[.025,.975]).tolist(),delta_ci_family95=np.quantile(diff,[.05/6,1-.05/6]).tolist() if name in ['B_frozen_GDI','C_predicted_deviation','D_joint2D'] else None,improved_people=int((delta<0).sum()),bands={})
  for lo,hi in [(0,18),(19,24),(25,28)]:
   mask=(y>=lo)&(y<=hi);r['bands'][f'{lo}-{hi}']=metrics(y[mask],p[mask]) if mask.any() else {'n':0}
  low=y<=18;predlow=p<=18;r['low_band_descriptive']=dict(actual_low=int(low.sum()),missed_low=int((low&~predlow).sum()),nonlow=int((~low).sum()),false_alarm=int((~low&predlow).sum()))
  results[name]=r
 diagnostic=np.abs(preds['E_joint3D']-y)-np.abs(preds['D_joint2D']-y)
 report=dict(protocol_sha256=sha(HERE/'protocol.json'),primary_n=len(y),auxiliary_n=33,healthy_reference_n=40,normal_all_n=138,results=results,median_baseline=metrics(y,preds['median_baseline']),diagnostic_3D_minus_2D=dict(delta_mae=float(diagnostic.mean()),ci95=np.quantile(diagnostic[boot].mean(1),[.025,.975]).tolist()),shuffle_control_mae=[c['mae'] for c in controls],elapsed_seconds=time.time()-t,feature_sha256=sha(BIG/'matched-features.npz'),clinical_input_sha256=sha(BIG/'clinical-inputs.npz'),prediction_sha256=sha(BIG/'poma-predictions.npz'))
 save(HERE/'results.json',report)
 save(HERE/'person-predictions.json',[dict(id=p,actual_POMA=float(y[i]),speed_mps=float(speed[i]),predictions={k:float(v[i]) for k,v in preds.items() if not k.startswith('shuffle')}) for i,p in enumerate(ids)])
 print('Finished; runtime',round(time.time()-t,1),flush=True)
if __name__=='__main__':main()
