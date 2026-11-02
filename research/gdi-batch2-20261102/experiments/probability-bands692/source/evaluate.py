"""Final held-out predictions only after all head selection and calibration frozen."""
from core import *

def outer_features(f,j):
 parts=[];hashes={}
 for family in ('CNN','STT'):
  p=CACHE/'outer'/family/('fold%d_inner%d.npz'%(f,j));proof=read(p.with_suffix('.json'));assert sha(p)==proof['sha256']
  assert sha(proof['checkpoint'])==proof['checkpoint_sha256'];parts.append(arrays(p));hashes[str(p)]=sha(p)
 a,b=parts
 for key in ('window_record','window_start','row_index','person_id','video_id','fold'):np.testing.assert_array_equal(a[key],b[key])
 c=arrays(OLD/'CNN'/('fold%d_inner%d.npz'%(f,j)));s=arrays(OLD/'STT'/('fold%d_inner%d.npz'%(f,j)))
 x,base=features(a['window_raw'],b['window_raw'],a['window_record'],a['row_index'],c['calibration'][0],s['calibration'][0])
 return a,x,base,hashes

def metrics(p,y,ids,hard=False):
 w=weights(ids);target=labels(y);pred=p.argmax(1);correct=pred==target;onehot=np.eye(K)[target]
 loss=-np.log(np.maximum(p[np.arange(len(y)),target],EPS))
 values=dict(accuracy=float(w@correct),two_or_more_band_error=float(w@(np.abs(pred-target)>=2)),
    nll=None if hard else float(w@loss),brier=float(w@((p-onehot)**2).sum(1)),
    rps=float(w@((np.cumsum(p,axis=1)[:,:-1]-np.cumsum(onehot,axis=1)[:,:-1])**2).mean(1)))
 confidence=p.max(1);bins=np.minimum((confidence*10).astype(int),9);table=[];ece=0
 for b in range(10):
  mask=bins==b;mass=w[mask].sum()
  if not mask.any():continue
  conf=float(w[mask]@confidence[mask]/mass);acc=float(w[mask]@correct[mask]/mass);ece+=mass*abs(conf-acc)
  table.append(dict(lower=b/10,upper=(b+1)/10,mass=float(mass),people=int(len(np.unique(ids[mask]))),observations=int(mask.sum()),confidence=conf,accuracy=acc))
 values.update(ece10=float(ece),mean_confidence=float(w@confidence),reliability=table)
 return values

def main():
 manifest=read(O/'training.json');assert manifest['complete'] and manifest['protocol_sha256']==sha(H/'protocol.json')
 oldpath=S/'mean-difference692/analysis/predictions.npz';assert sha(oldpath)==read(S/'mean-difference692/analysis/verification.json')['predictions_sha256'];old=arrays(oldpath)
 n=len(old['y'])*2;ids=np.repeat(old['person_id'],2);y=old['y'].reshape(-1);fold=np.repeat(old['fold'],2)
 names=['residual_raw','residual_calibrated','score_only_raw','score_only_calibrated','score_spread_raw','score_spread_calibrated','selected_raw','selected_calibrated']
 out={k:np.full((n,K),np.nan) for k in names};matched=np.full(n,np.nan);hashes={};all_experts={}
 for f in range(5):
  rows=np.flatnonzero(old['fold']==f);ix=np.flatnonzero(fold==f);experts=[];point=[]
  for j in range(5):
   if j==f:continue
   entry=next(e for e in manifest['experts'] if e['outer']==f and e['inner']==j);path=Path(entry['path']);assert sha(path)==entry['sha256'];model=read(path)
   a,x,base,h=outer_features(f,j);hashes.update(h);np.testing.assert_array_equal(a['row_index'],rows)
   for key in ('person_id','video_id','fold'):np.testing.assert_array_equal(a[key],old[key][rows])
   disjoint(model['base_people'],model['train_people'],model['selection_people'],model['calibration_people'],np.unique(old['person_id'][rows]))
   e={}
   for family in ['score_only','score_spread']:
    raw=predict(model['models'][family],x,base);e[family+'_raw']=raw;e[family+'_calibrated']=temperature(raw,model['models'][family]['temperature'])
   e['selected_raw']=e[model['selected_family']+'_raw'];e['selected_calibrated']=e[model['selected_family']+'_calibrated']
   raw=probabilities(base,np.full(len(base),model['residual_scale']));e['residual_raw']=raw;e['residual_calibrated']=temperature(raw,model['residual_temperature'])
   experts.append(e);point.append(base);all_experts.update({'f%d_j%d_%s'%(f,j,k):v for k,v in e.items()})
  for name in names:out[name][ix]=np.mean([e[name] for e in experts],axis=0)
  matched[ix]=np.mean(point,axis=0)
 out['matched_hard']=np.eye(K)[labels(matched)]
 for name,key in [('legacy_CNN_STT_hard','existing_blend'),('legacy_mean_difference_hard','inner_selected_difference')]:out[name]=np.eye(K)[labels(old[key].reshape(-1))]
 rp=read(R/'analysis/comparison.json');assert sha(rp['predictions_path'])==rp['predictions_sha256'];robust=arrays(rp['predictions_path'])
 for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(robust[key],old[key])
 out['legacy_robust_hard']=np.eye(K)[labels(robust['inner_selected'].reshape(-1))]
 report=dict(metrics={},fold_accuracy={},paired={},people=692,records=1061,observations=n,training_sha256=sha(O/'training.json'),
   protocol_sha256=sha(H/'protocol.json'),source_hashes=hashes,limitations=read(H/'protocol.json')['limitations'])
 for name,p in out.items():
  assert p.shape==(n,K) and np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
  report['metrics'][name]=metrics(p,y,ids,name.endswith('_hard'))
  report['fold_accuracy'][name]=[metrics(p[fold==f],y[fold==f],ids[fold==f],name.endswith('_hard'))['accuracy'] for f in range(5)]
 people=np.unique(ids);target=labels(y);truth=np.eye(K)[target]
 def per_person(p,metric):
  if metric=='accuracy':v=p.argmax(1)==target
  elif metric=='nll':v=-np.log(p[np.arange(n),target])
  else:v=((p-truth)**2).sum(1)
  return np.array([v[ids==i].mean() for i in people])
 for metric,reference in [('accuracy','legacy_mean_difference_hard'),('accuracy','legacy_robust_hard'),('accuracy','matched_hard'),('accuracy','residual_calibrated'),('nll','residual_calibrated'),('brier','residual_calibrated'),('nll','selected_raw'),('brier','selected_raw')]:
  delta=per_person(out['selected_calibrated'],metric)-per_person(out[reference],metric);rng=np.random.default_rng(20261031)
  boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)])
  report['paired'][metric+'_vs_'+reference]=dict(delta=float(delta.mean()),descriptive_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
 # Score-band support is descriptive; participants may appear in multiple bands.
 report['support']=[dict(class_index=k,observations=int(sum(target==k)),people=int(len(np.unique(ids[target==k])))) for k in range(K)]
 dest=CACHE/'evaluation.npz';np.savez_compressed(dest,**out,y=y,person_id=ids,fold=fold,video_id=np.repeat(old['video_id'],2),matched_point=matched)
 expath=CACHE/'expert_predictions.npz';np.savez_compressed(expath,**all_experts)
 report.update(predictions_path=str(dest),predictions_sha256=sha(dest),experts_sha256=sha(expath),protected_final_data_opened=False,neural_optimizer_steps=0,
   hard_nll_note='Hard bins do not supply probabilities; log loss is infinite on any wrong class, reported as null rather than arbitrary epsilon clipping')
 save(O/'comparison.json',report)
 print(json.dumps({k:{m:v[m] for m in ['accuracy','nll','brier','ece10','mean_confidence']} for k,v in report['metrics'].items()}),flush=True)
if __name__=='__main__':main()
