"""Independent equations, model replay and clinical linkage checks."""
from prepare import *
from scipy.stats import spearmanr

def main():
 people=read(HERE/'people.json');byid={p['id']:p for p in people};trials=read(HERE/'trial-audit.json');results=read(HERE/'comparison.json');fits=read(BIG/'fits.json');receipt=read(HERE/'fit-receipt.json')
 for path,key in [(HERE/'protocol.json','protocol_sha256'),(HERE/'operational-note.json','operational_note_sha256'),(HERE/'people.json','people_sha256'),(HERE/'gdi-scores.json','gdi_scores_sha256'),(HERE/'comparison.json','comparison_sha256'),(BIG/'fits.json','fits_sha256')]:assert sha(path)==receipt[key]
 previous=Path('D:/gait-research-data/D_P/sppb-sample-size-cnn-20261101/bundle/target.npz')
 assert sha(previous)=='115f5c87b74218276f8b8db71c2c747322d2358ab3b3b5bd066c719318002330'
 with np.load(previous) as z:
  for person in people:
   mask=z['ids']==person['id'];assert mask.sum()==5
   assert (z['y'][mask]==person['sppb']).all() and (z['folds'][mask]==person['fold']).all()
 # Re-read every raw file and independently derive gait phase from event timestamps.
 max_feature_error=0.;rejected=sum(len(r['rejected']) for r in trials)
 for person in people:
  pp=[r for r in trials if r['id']==person['id']];assert len(pp)==5;pcis=[]
  for side in ['Left','Right']:
   phases=[]
   for r in pp:
    assert sha(RAW/r['file'])==r['source_sha256']
    for c in r['cycles']:
     if c['side']!=side:continue
     assert c['t0'] in r['events'][side]['Foot Strike'] and c['t1'] in r['events'][side]['Foot Strike']
     other='Right' if side=='Left' else 'Left';assert c['opposite'] in r['events'][other]['Foot Strike'] and c['off'] in r['events'][side]['Foot Off']
     ratio=(c['opposite']-c['t0'])/(c['t1']-c['t0']);phases.append(ratio)
   phases=np.array(phases);pci=100*np.std(phases,ddof=1)/np.mean(phases)+200*np.mean(np.abs(phases-.5));pcis.append(pci)
   assert len(phases)==person['phase_counts'][side]
   np.testing.assert_allclose(pci,person['PCI_by_side'][side],atol=1e-10)
  np.testing.assert_allclose(np.mean(pcis),person['features']['PCI'],atol=1e-10)
  for key,name in [('stride','stride_time_CV'),('length','step_length_CV'),('swing','swing_percent_CV')]:
   groups=[np.array([c[key] for c in r['cycles'] if c['side']==s]) for r in pp for s in ['Left','Right']]
   pooled=np.concatenate([v-v.mean() for v in groups]);n=sum(len(v)-1 for v in groups);mean=np.mean(np.concatenate(groups))
   value=100*np.sqrt(np.dot(pooled,pooled)/n)/mean;error=abs(value-person['features'][name]);max_feature_error=max(max_feature_error,error);assert error<1e-10
 # Native sample C3D vs CSV: genuinely released events, units and model centres.
 sys.path.insert(0,'D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot/deps')
 import ezc3d
 cpath=ROOT/'outputs/gait-public-access-20261015/OlderAdults44/Subject01_01.c3d'
 metadata=read(ROOT/'outputs/gait-public-access-20261015/older44-metadata.json')
 expected=next(f['content_details']['sha256_hash'] for f in metadata['files'] if f['filename']==cpath.name);assert sha(cpath)==expected
 native=dict(source_sha256=expected,source_hash_matches=True,status='not_checked')
 try:
  c=ezc3d.c3d(str(cpath))
 except OSError as exc:
  # Keep the planned native cross-check FAILED; other independent checks may continue.
  # Do not assert native equivalence or weaken its numeric tolerance.
  native.update(status='FAILED_SOURCE_FORMAT',reason=str(exc),other_parser='python-c3d also lacks POINT:LABELS and EVENT data on this exact published file; permissive ezc3d also fails')
 else:
  frames,q,events,params,dup=load_trial(RAW/'Subject01_01.csv')
  assert c['header']['points']['first_frame']+1==frames[0] and c['header']['points']['frame_rate']==100
  ev=c['parameters']['EVENT'];times=np.asarray(ev['TIMES']['value']);times=times[0]*60+times[1];evmax=0.
  for side in ['Left','Right']:
   for kind in ['Foot Strike','Foot Off']:
    vals=sorted(float(t) for t,s,k in zip(times,ev['CONTEXTS']['value'],ev['LABELS']['value']) if s==side and k==kind)
    np.testing.assert_allclose(vals,events[side][kind],atol=1e-5);evmax=max(evmax,float(np.max(np.abs(np.array(vals)-events[side][kind]))))
  native.update(status='PASS',event_time_max_error_s=evmax)
 # Numerical and boundary checks for PCI / centring, independent of clinical labels.
 ideal=np.full(20,.5);assert 100*ideal.std(ddof=1)/ideal.mean()+200*np.abs(ideal-.5).mean()==0
 irregular=np.tile([.4,.6],10);assert 100*irregular.std(ddof=1)/irregular.mean()+200*np.abs(irregular-.5).mean()>0
 assert pooled_cv([[1,1,1],[2,2,2]])==0
 # Each saved fit is independently solved using linear algebra and replayed.
 max_fit_error=0.;max_replay_error=0.
 for fit in fits:
  d=results[fit['cohort']+'/'+fit['variant']];ids=d['ids'];X=np.array(d['inputs'][fit['condition']]);y=np.array(d['y']);folds=np.array(d['folds'])
  tr=np.array([ids.index(p) for p in fit['train']]);te=np.array([ids.index(p) for p in fit['evaluation']]);assert not set(tr)&set(te)
  outer=fit['outer'];assert (folds[tr]!=outer).all()
  if fit['inner'] is not None:assert (folds[te]!=outer).all()
  else:assert set(te)==set(np.flatnonzero(folds==outer))
  mean=X[tr].mean(0);scale=X[tr].std(0);scale[scale==0]=1
  np.testing.assert_allclose(mean,fit['mean'],atol=1e-12);np.testing.assert_allclose(scale,fit['scale'],atol=1e-12)
  a=(X[tr]-mean)/scale;ac=a-a.mean(0);yc=y[tr]-y[tr].mean();coef=np.linalg.solve(ac.T@ac+fit['alpha']*np.eye(X.shape[1]),ac.T@yc);intercept=y[tr].mean()-a.mean(0)@coef
  err=max(float(np.max(np.abs(coef-fit['coef']))),abs(intercept-fit['intercept']));assert err<1e-8;max_fit_error=max(max_fit_error,err)
  prediction=(X[te]-np.array(fit['mean']))/np.array(fit['scale'])@np.array(fit['coef'])+fit['intercept'];err=float(np.max(np.abs(prediction-fit['prediction'])));assert err<1e-10;max_replay_error=max(max_replay_error,err)
  if fit['inner'] is None:np.testing.assert_allclose(prediction,np.array(d['predictions'][fit['condition']])[te],atol=1e-10)
 for key,d in results.items():
  ids=d['ids'];y=np.array(d['y']);folds=np.array(d['folds']);assert len(set(ids))==len(ids)
  assert [byid[p]['sppb'] for p in ids]==d['y'] and [byid[p]['fold'] for p in ids]==d['folds']
  for selection in d['selection']:
   condition=selection['condition'];outer=selection['outer'];scores=[]
   for alpha in [.1,1.,10.,100.,1000.]:
    ff=[f for f in fits if f['cohort']+'/'+f['variant']==key and f['outer']==outer and f['condition']==condition and f['inner'] is not None and f['alpha']==alpha];assert len(ff)==3
    seen=[p for f in ff for p in f['evaluation']];assert len(seen)==len(set(seen)) and set(seen)=={p for p,fo in zip(ids,folds) if fo!=outer}
    errors=[abs(pred-byid[pid]['sppb']) for f in ff for pid,pred in zip(f['evaluation'],f['prediction'])];scores.append(sum(errors)/len(errors))
   np.testing.assert_allclose(scores,selection['inner_MAE'],atol=1e-10);assert [.1,1.,10.,100.,1000.][int(np.argmin(scores))]==selection['alpha']
  for condition,p in d['predictions'].items():
   p=np.array(p);e=p-y;m=d['metrics'][condition];assert abs(sum(abs(e))/len(e)-m['MAE'])<1e-12
   assert abs(np.sqrt(np.dot(e,e)/len(e))-m['RMSE'])<1e-12
   assert int(((y<=8)&(p<=8)).sum())==m['TP'];assert int(((y>8)&(p<=8)).sum())==m['FP']
   if condition in ['Mean','Median']:
    expected=np.array([np.mean(y[folds!=f]) if condition=='Mean' else np.median(y[folds!=f]) for f in folds]);np.testing.assert_allclose(p,expected,atol=1e-12)
  indices=np.random.RandomState(20261103).randint(0,len(y),(10000,len(y)))
  for contrast,v in d['contrasts'].items():
   a,b=contrast.split('-');ea=np.abs(np.array(d['predictions'][a])-y);eb=np.abs(np.array(d['predictions'][b])-y);distribution=ea[indices].mean(1)-eb[indices].mean(1)
   np.testing.assert_allclose(np.percentile(distribution,[2.5,97.5]),v['CI95'],atol=1e-12)
 # Independent ensemble aggregation and equal trial weighting, not window weighting.
 fv=read(HERE/'frozen-verification.json');z=np.load(BIG/'features.npz');pred=np.load(BIG/'frozen-predictions.npz');manifest=read(ROOT/'gdi-research-harness/studies/20261102-final223-frozen/models.json')
 for check in fv['checks']:assert sha(check['checkpoint'])==check['checkpoint_sha256'] and sha(BIG/'members'/(check['key']+'.npz'))==check['prediction_sha256']
 fused=[]
 for f,s in enumerate(manifest['selections']):
  bags={family:np.mean([pred[f'{family}_{f}_{m["tag"]}'] for m in manifest['members'] if m['family']==family and m['fold']==f],0) for family in ['CNN','STT']}
  expected=s['cnn_weight']*bags['CNN']+(1-s['cnn_weight'])*bags['STT'];mean=expected.mean(1)
  np.testing.assert_allclose(pred[f'fusion_{f}'].mean(1),mean,atol=1e-10);fused.append(mean)
 mean_prediction=np.mean(fused,0);np.testing.assert_allclose(pred['fusion'].mean(1),mean_prediction,atol=1e-10)
 for person in read(HERE/'gdi-scores.json'):
  for variant,value in person['scores'].items():
   vals=[float(mean_prediction[(z['owner']==i)&(z['variant']==variant)].mean()) for i,r in enumerate(trials) if r['id']==person['id'] and r['common_starts']]
   assert abs(np.mean(vals)-value)<1e-10
 # Gate remains explicit: this is a camera simulation, not anatomical/RGB validation.
 assert read(HERE/'input-audit.json')['anatomical_equivalence'] is False
 result=dict(status='COMPUTATION_PASS_NATIVE_SOURCE_CROSSCHECK_FAILED' if native['status']!='PASS' else 'PASS',raw_files=220,prior44_labels_and_folds_match=True,native_C3D_crosscheck=native,feature_equation_max_error=max_feature_error,rejected_cycles=rejected,saved_SPPB_fits_recomputed=len(fits),fit_coefficient_max_error=max_fit_error,saved_SPPB_prediction_replay_max_error=max_replay_error,frozen_GDI_models=len(fv['checks']),prior_GDI_replay_max_error=max(v['prior_replay_max'] for v in fv['checks']),person_split_and_inner_selection=True,independent_bootstrap_and_metrics=True,anatomical_equivalence=False,clinical_validation=False,comparison_sha256=sha(HERE/'comparison.json'),protocol_sha256=sha(HERE/'protocol.json'))
 save(HERE/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
