"""Replay head selection/fitting/calibration and independently check held-out scores."""
from core import *
from evaluate import outer_features

def independent_probability(model,x,base):
 z=(x-np.array(model['mean']))/np.array(model['sd']);n=3 if model['family']=='score_only' else 7
 design_x=np.c_[np.ones(len(x)),z[:,:n]];design_s=np.ones((len(x),1)) if n==3 else np.c_[np.ones(len(x)),z[:,3:7]]
 theta=np.array(model['theta']);mu=base+10*np.sum(design_x*theta[:n+1],axis=1);s=np.exp(np.sum(design_s*theta[n+1:],axis=1))
 cdf=np.exp(-np.logaddexp(0,-(EDGES[None,:]-mu[:,None])/s[:,None]))
 p=np.maximum(np.diff(np.c_[np.zeros(len(x)),cdf,np.ones(len(x))],axis=1),EPS);return p/p.sum(1,keepdims=True)
def independent_temp(p,t):
 a=p**(1/t);return a/a.sum(1,keepdims=True)
def independently_weighted(values,ids):return float(np.mean([values[ids==i].mean() for i in np.unique(ids)]))

cfg=read(H/'protocol.json');manifest=read(O/'training.json');report=read(O/'comparison.json')
assert report['training_sha256']==sha(O/'training.json') and manifest['protocol_sha256']==sha(H/'protocol.json')
assert report['predictions_sha256']==sha(report['predictions_path']) and report['experts_sha256']==sha(CACHE/'expert_predictions.npz')
z=arrays(report['predictions_path']);ep=arrays(CACHE/'expert_predictions.npz');base=arrays(S/'mean-difference692/analysis/predictions.npz')
all_people=np.unique(base['person_id']);maxerr=0;fitcount=0;tempcount=0;selectioncount=0;proofs=[]
for entry in manifest['experts']:
 f,j=entry['outer'],entry['inner'];path=Path(entry['path']);assert sha(path)==entry['sha256'];model=read(path);d=inner_data(f,j)
 for p,h in model['source_hashes'].items():assert sha(p)==h
 groups=partition(d['ids'],f,j);outer=np.unique(base['person_id'][base['fold']==f]);disjoint(d['base_people'],*groups,outer)
 assert set(np.r_[d['base_people'],*groups,outer])==set(all_people)
 for key,g in zip(('train_people','selection_people','calibration_people'),groups):assert model[key]==g.tolist()
 tr,va,ca=[np.isin(d['ids'],g) for g in groups];refit=tr|va;choices=[]
 # Poisoning every outer target cannot change a single meta training target.
 poisoned=base['y'].copy();poisoned[base['fold']==f]=1e9
 np.testing.assert_array_equal(poisoned[d['rows']].reshape(-1),d['y'])
 a,x,point,hashes=outer_features(f,j)
 for p,h in hashes.items():assert report['source_hashes'][p]==h
 calpath=path.with_suffix('.npz');assert sha(calpath)==model['calibration_predictions_sha256'];savedcal=arrays(calpath)
 np.testing.assert_array_equal(savedcal['y'],d['y'][ca]);np.testing.assert_array_equal(savedcal['person_id'],d['ids'][ca])
 for family in cfg['families']:
  scores=[]
  for lam in cfg['ridge_grid']:
   fit=fit_head(d['x'][tr],d['base'][tr],d['y'][tr],d['ids'][tr],family,lam)
   pp=independent_probability(fit,d['x'][va],d['base'][va]);score=independently_weighted(-np.log(pp[np.arange(sum(va)),labels(d['y'][va])]),d['ids'][va])
   old=next(q for q in model['candidates'] if q['family']==family and q['lambda_']==lam)
   assert abs(score-old['selection_nll'])<1e-9;scores.append(score);selectioncount+=1
  k=int(np.argmin(scores));lam=cfg['ridge_grid'][k];assert model['models'][family]['lambda_']==lam;choices.append((scores[k],family))
  fitted=fit_head(d['x'][refit],d['base'][refit],d['y'][refit],d['ids'][refit],family,lam)
  for key in ('mean','sd','theta'):np.testing.assert_allclose(fitted[key],model['models'][family][key],atol=1e-10,rtol=0)
  # Check standardization directly with training+selection people only.
  w=weights(d['ids'][refit]);mu=np.average(d['x'][refit],weights=w,axis=0);sd=np.sqrt(np.average((d['x'][refit]-mu)**2,weights=w,axis=0));sd[sd<1e-6]=1
  np.testing.assert_allclose(mu,fitted['mean'],atol=1e-12,rtol=0);np.testing.assert_allclose(sd,fitted['sd'],atol=1e-12,rtol=0)
  cp=independent_probability(fitted,d['x'][ca],d['base'][ca]);np.testing.assert_allclose(cp,savedcal[family+'_raw'],atol=1e-12,rtol=0)
  t=fit_temperature(cp,d['y'][ca],weights(d['ids'][ca]));assert abs(t-model['models'][family]['temperature'])<1e-6
  calibrated=independent_temp(cp,model['models'][family]['temperature']);np.testing.assert_allclose(calibrated,savedcal[family+'_calibrated'],atol=1e-12,rtol=0)
  np.testing.assert_array_equal(cp.argmax(1),calibrated.argmax(1));assert nll(calibrated,d['y'][ca],weights(d['ids'][ca]))<=nll(cp,d['y'][ca],weights(d['ids'][ca]))+1e-9
  op=independent_probability(fitted,x,point);key='f%d_j%d_%s'%(f,j,family)
  err=float(np.abs(op-ep[key+'_raw']).max());maxerr=max(maxerr,err);assert err<1e-12
  np.testing.assert_allclose(independent_temp(op,model['models'][family]['temperature']),ep[key+'_calibrated'],atol=1e-12,rtol=0)
  fitcount+=1;tempcount+=1
 assert min(choices,key=lambda q:q[0])[1]==model['selected_family']
 for suffix in ('raw','calibrated'):
  np.testing.assert_array_equal(ep['f%d_j%d_selected_%s'%(f,j,suffix)],ep['f%d_j%d_%s_%s'%(f,j,model['selected_family'],suffix)])
 scale=fit_residual(d['base'][refit],d['y'][refit],d['ids'][refit]);assert abs(scale-model['residual_scale'])<1e-10
 raw=probabilities(d['base'][ca],np.full(sum(ca),scale));temp=fit_temperature(raw,d['y'][ca],weights(d['ids'][ca]));assert abs(temp-model['residual_temperature'])<1e-6;tempcount+=1
 rawouter=probabilities(point,np.full(len(point),scale));np.testing.assert_allclose(rawouter,ep['f%d_j%d_residual_raw'%(f,j)],atol=1e-12,rtol=0)
 np.testing.assert_allclose(independent_temp(rawouter,temp),ep['f%d_j%d_residual_calibrated'%(f,j)],atol=1e-8,rtol=0)
 # Poison the calibration targets and repeat a head fit: head parameters stay fixed.
 poison=d['y'].copy();poison[ca]=-999
 chosen=model['selected_family'];lam=model['models'][chosen]['lambda_'];pfit=fit_head(d['x'][refit],d['base'][refit],poison[refit],d['ids'][refit],chosen,lam)
 np.testing.assert_array_equal(pfit['theta'],model['models'][chosen]['theta'])
 proofs.append(dict(outer=f,inner=j,base_people=len(d['base_people']),meta_counts=[len(g) for g in groups],outer_people=len(outer),disjoint=True))
 print('VERIFIED_EXPERT',f,j,flush=True)
ids=z['person_id'];y=z['y'];target=labels(y);truth=np.eye(K)[target]
for name,m in report['metrics'].items():
 p=z[name];assert np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
 pred=p.argmax(1);correct=pred==target
 assert abs(independently_weighted(correct,ids)-m['accuracy'])<1e-12
 assert abs(independently_weighted((p-truth).__pow__(2).sum(1),ids)-m['brier'])<1e-12
 if not name.endswith('_hard'):
  assert abs(independently_weighted(-np.log(p[np.arange(len(y)),target]),ids)-m['nll'])<1e-12
  for f in range(5):
   expected=sum(ep['f%d_j%d_%s'%(f,j,name)] for j in range(5) if j!=f)/4
   np.testing.assert_array_equal(expected,p[z['fold']==f])
 confidence=p.max(1);w=weights(ids);reliability=[];ece=0
 for b in range(10):
  mask=(confidence>=b/10)&((confidence<(b+1)/10) if b<9 else confidence<=1)
  if not mask.any():continue
  mass=w[mask].sum();ece+=abs(np.dot(w[mask],confidence[mask]-correct[mask]))
  reliability.append((float(mass),float(np.dot(w[mask],confidence[mask])/mass),float(np.dot(w[mask],correct[mask])/mass)))
 assert abs(ece-m['ece10'])<1e-12
 for row,(mass,conf,acc) in zip(m['reliability'],reliability):np.testing.assert_allclose([row['mass'],row['confidence'],row['accuracy']],[mass,conf,acc],atol=1e-12,rtol=0)
save(O/'verification.json',dict(passed=True,experts=20,heads_refitted=fitcount,hyperparameter_fits_repeated=selectioncount,temperatures_recomputed=tempcount,
 splits=proofs,base_head_selection_calibration_outer_disjoint_per_expert=True,all_692_people_accounted_for_per_expert=True,
 independent_probabilities_max_error=maxerr,all_outer_probability_mixtures_checked=True,person_equal_metrics_recomputed=True,
 outer_target_poison_invariance=True,calibration_targets_do_not_fit_head=True,source_hashes_checked=True,comparison_sha256=sha(O/'comparison.json'),
 training_sha256=sha(O/'training.json'),predictions_sha256=sha(report['predictions_path']),neural_optimizer_steps=0,protected_final_data_opened=False,
 code_hashes={n:sha(H/n) for n in ['core.py','train.py','export_outer.py','evaluate.py','verify.py','test_core.py']}))
print('PROBABILITY_VERIFICATION_PASSED',flush=True)
