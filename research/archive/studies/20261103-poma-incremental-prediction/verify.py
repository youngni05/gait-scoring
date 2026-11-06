"""Independent dual-system replay and metric/split verification; does not import fitting code."""
from prepare import *

def independent_fit(x,y,alpha,block=None):
 mean=np.mean(x,axis=0);scale=np.std(x,axis=0);scale=np.where(scale<1e-12,1,scale)
 if block is not None:
  scale[:block]=scale[:block]*np.sqrt(block)
  if x.shape[1]>block:scale[block:]=scale[block:]*np.sqrt(x.shape[1]-block)
 z=(x-mean)/scale;intercept=y.mean()
 weight=z.T@np.linalg.solve(z@z.T+alpha*np.eye(len(y)),y-intercept)
 return mean,scale,weight,intercept

def estimate(x,y,q,alpha,block=None,clip=False):
 m,s,w,b=independent_fit(x,y,alpha,block);v=((q-m)/s)@w+b
 return np.clip(v,0,28) if clip else v

def main():
 manifest=read(HERE/'input-manifest.json')
 for item in manifest:assert sha(item['path'])==item['sha256']
 result=read(HERE/'results.json');assert result['protocol_sha256']==sha(HERE/'protocol.json')
 for name,key in [('matched-features.npz','feature_sha256'),('clinical-inputs.npz','clinical_input_sha256'),('poma-predictions.npz','prediction_sha256')]:assert sha(BIG/name)==result[key]
 z=np.load(BIG/'matched-features.npz');c=np.load(BIG/'clinical-inputs.npz');pred=np.load(BIG/'poma-predictions.npz');protocol=read(HERE/'protocol.json');ids=c['person_id'].tolist();y=c['y'];speed=c['speed'];assert ids==protocol['primary_ids']==pred['person_id'].tolist();assert np.array_equal(y,pred['y'])
 clinical={old.normid(p['id']):p for p in read(BASE/'speed-control/participants.json')};assert len(ids)==len(set(ids))==25
 original_clinical={old.normid(p['supplement_candidate_ids'][0]):p for p in read(BASE/'person-linkage.json')['stroke']['rows']}
 for i,p in enumerate(ids):
  assert clinical[p]['POMA']==y[i] and clinical[p]['age']>=65 and clinical[p]['speed']['mean_mps']==speed[i]
  assert original_clinical[p]['values']['POMA']==y[i] and original_clinical[p]['values']['Age']==clinical[p]['age']
 source=read(HERE/'source-audit.json');rows=read(OLD/'rows.json');orig=np.load(OB/'features.npz');frozen=np.load(OB/'frozen-predictions.npz')
 expected=[i for i,r in enumerate(rows) if r['primary'] and r['starts']];assert [s['row'] for s in source]==expected
 aggregation={p:[] for p in z['person_id']}
 for s in source:
  r=rows[s['row']];assert r['primary'] and s['starts']==r['starts'];wi=s['window_indices'];assert np.all(orig['owner'][wi]==s['row']) and orig['start'][wi].tolist()==s['starts'];aggregation[s['person']].append(float(np.mean(frozen['fusion'][wi])))
 for i,p in enumerate(z['person_id']):assert abs(np.mean(aggregation[p])-z['gdi'][i])<1e-10
 # Representation invariances are meaningful checks independent of POMA outcomes.
 rng=np.random.default_rng(123);test=rng.normal(size=(2,124,3,3));test[:,:,2,2]+=3
 invariance={}
 for dim in [2,3]:
  points=test[:,:,:,:dim];a=describe(points)
  err=max(float(np.max(np.abs(a-describe(points*3.2+np.arange(dim)[None,None,None,:])))),float(np.max(np.abs(a-describe(points[::-1])))))
  assert err<1e-8;invariance[str(dim)]=err
 # Verify auxiliary provenance, target references, and all upstream hyperparameter candidates.
 linkage=read(BASE/'person-linkage.json');curves=np.load('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot/person-curves.npz');people=read(HERE/'people.json');idx={p:i for i,p in enumerate(z['person_id'])};ci=np.array([idx[p] for p in ids]);up=read(HERE/'upstream-audit.json');upcount=0;max_replay=0.
 for name,audit in up.items():
  normal={old.normid(p['supplement_candidate_ids'][0]):curves['healthy'][p['mat_index']-1,:,:,::20] for p in linkage['healthy']['rows'] if name=='allhealthy' or p['values']['Age']>=65}
  def labels(personids,refs):
   values=[]
   for person in personids:
    reference=np.mean(np.stack([normal[p].mean(0) for p in refs if p!=person]),axis=0);m=people[idx[person]];own=curves[m['cohort']][m['mat_index']-1,:,:,::20]
    values.append(np.mean(np.sqrt(np.mean((own-reference)**2,axis=(1,2)))))
   return np.array(values)
  assert set(audit['train']).isdisjoint(ids) and set(audit['reference']).isdisjoint(ids);assert len(audit['train'])==33
  alphas=[.1,1,10,100,1000];errors=np.zeros(5)
  for fold in audit['inner']:
   assert set(fold['train']).isdisjoint(fold['validation']) and set(fold['reference']).isdisjoint(fold['validation'])
   assert set(fold['train']+fold['validation']).isdisjoint(ids);assert set(fold['reference'])==set(normal)-set(fold['validation'])
   tr=[idx[p] for p in fold['train']];va=[idx[p] for p in fold['validation']];yt=labels(fold['train'],fold['reference']);yv=labels(fold['validation'],fold['reference'])
   for j,alpha in enumerate(alphas):errors[j]+=np.abs(estimate(z['common'][tr],yt,z['common'][va],alpha)-yv).sum();upcount+=1
  assert np.allclose(errors/33,audit['inner_mae'],atol=1e-8);assert alphas[int(np.argmin(errors))]==audit['alpha']
  path=Path(audit['checkpoint']);assert sha(path)==audit['sha256'];model=np.load(path);tr=[idx[p] for p in audit['train']];yt=labels(audit['train'],audit['reference']);assert np.allclose(yt,model['target'],atol=1e-10)
  pp=estimate(z['common'][tr],yt,z['common'][ci],audit['alpha']);from_saved=(z['common'][ci]-model['mean'])/model['scale']@model['weight']+model['intercept'];err=float(np.max(np.abs(pp-from_saved)));assert err<1e-7;max_replay=max(max_replay,err)
  key='C_predicted_deviation' if name=='age65plus' else 'C_allhealthy';assert np.allclose(from_saved,c[key][:,0],atol=1e-10)
  if name=='age65plus':assert np.allclose(labels(ids,audit['reference']),c['F_measured_deviation'][:,0],atol=1e-10)
 # Reconstruct each chosen model and every inner selection candidate via independent linear algebra.
 audits=read(HERE/'fit-audit.json');checks=0;saved=0;control=read(HERE/'shuffle-audit.json')
 experiments=[(name,c[name],aa,pred[name],True) for name,aa in audits.items()]+[('shuffle_'+str(a['seed']),c['D_joint2D'][a['permutation']],a['nested'],pred['shuffle_'+str(a['seed'])],False) for a in control]
 for name,extra,aa,expected,has_saved in experiments:
  assert sorted(s['test'][0] for s in aa)==list(range(25))
  for row in aa:
   tr=np.array(row['train']);te=np.array(row['test']);assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(range(25))
   validation=[];computed=[]
   for fold in row['inner']:
    assert set(fold['train']).isdisjoint(fold['validation']);assert set(fold['train']+fold['validation'])==set(tr);validation+=fold['validation']
   assert sorted(validation)==sorted(tr)
   for candidate in row['selection']:
    degree=candidate['degree'];alpha=candidate['alpha'];x=np.column_stack([*[speed**d for d in range(1,degree+1)],extra]);errors=[]
    for fold in row['inner']:
     ti=np.array(fold['train']);vi=np.array(fold['validation']);p=estimate(x[ti],y[ti],x[vi],alpha,degree,True);errors.extend(np.abs(p-y[vi]));checks+=1
    value=float(np.mean(errors));assert abs(value-candidate['mae'])<1e-7;computed.append(value)
   best=row['selection'][int(np.argmin(computed))];assert best==row['selected'];degree=best['degree'];x=np.column_stack([*[speed**d for d in range(1,degree+1)],extra]);pp=estimate(x[tr],y[tr],x[te],best['alpha'],degree,True);assert abs(pp[0]-expected[te[0]])<1e-7
   if has_saved:
    path=Path(row['checkpoint']);assert sha(path)==row['sha256'];m=np.load(path);assert np.array_equal(tr,m['train']) and np.array_equal(te,m['test']);replay=np.clip((x[te]-m['mean'])/m['scale']@m['weight']+m['intercept'],0,28);err=float(np.max(np.abs(replay-pp)));assert err<1e-7;max_replay=max(max_replay,err);saved+=1
  print('verified',name,flush=True)
 # Recalculate paired bootstrap and score-band metrics independently.
 boot=np.random.default_rng(20261103).integers(0,25,(10000,25));base=np.abs(pred['A_speed']-y)
 for name,r in result['results'].items():
  errors=pred[name]-y;absolute=np.abs(errors);delta=absolute-base
  assert abs(absolute.mean()-r['mae'])<1e-12;assert np.allclose(np.quantile(delta[boot].mean(axis=1),[.025,.975]),r['delta_ci95'],atol=1e-12)
  assert int((absolute<=2).sum())==r['within2'] and int((absolute>5).sum())==r['over5']
  for band,v in r['bands'].items():
   low,high=map(int,band.split('-'));mask=(y>=low)&(y<=high);assert int(mask.sum())==v['n'];assert abs(absolute[mask].mean()-v['mae'])<1e-12
  low=y<=18;plow=pred[name]<=18;assert int((low&~plow).sum())==r['low_band_descriptive']['missed_low'];assert int((~low&plow).sum())==r['low_band_descriptive']['false_alarm']
 verification=dict(status='passed',independent_method='dual ridge linear solve vs original SVD',primary_people=25,auxiliary_people=33,all_sources_same_windows=True,original_frozen_predictions_reaggregated=True,upstream_selection_fits_checked=upcount,downstream_inner_fits_checked=checks,saved_functional_models_replayed=saved,saved_auxiliary_models_replayed=2,max_prediction_replay_error=max_replay,heldout_people_excluded_from_all_fitting=True,metrics_and_bootstrap_recomputed=True,invariance_max_errors=invariance,protocol_sha256=sha(HERE/'protocol.json'),results_sha256=sha(HERE/'results.json'),code_sha256={f:sha(HERE/f) for f in ['prepare.py','run.py','verify.py']})
 save(HERE/'verification.json',verification);print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
