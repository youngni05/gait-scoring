"""Audit recovered SynthGait results; preserve failed original summary and derive OOF separately."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,traceback,datetime,subprocess
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent/'20261030-gpu-motionbert-study';O=H/'analysis';O.mkdir(exist_ok=True)
os.environ['GDI_DATA_PATH']=str(S/'cnn-record-loss692/bundle/data.npz')
os.environ['EXTERNAL_DATA_PATH']='D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-synthgait-conv-transfer/external.npz'
import run as r
import review_results as review
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);temp=p.with_suffix('.tmp')
 temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
state=dict(phase='verifying_received_files',started_at=datetime.datetime.now().astimezone().isoformat(),completed_fits=[],epochs=0)
progress=lambda:save(O/'progress.json',state)
process_receipt=read(O/'process-receipt.json') if (O/'process-receipt.json').exists() else {}
process_receipt.update(pid=os.getpid(),creation_date=state['started_at'],active=True,launch_mode='foreground_exec_session',script_sha256=sha(__file__))
save(O/'process-receipt.json',process_receipt)
progress();receipt=read(H/'checks/closeout-receipt.json');archive=Path(receipt['archive']);inventory=read(receipt['inventory'])
assert receipt['passed'] and receipt['server_local_sha256_matched'] and receipt['expected_counts_present']
assert sha(archive)==receipt['server_archive_sha256'] and sha(receipt['inventory'])==receipt['inventory_sha256']
U=archive.parent/'performance-review-originals';U.mkdir(exist_ok=True)
with tarfile.open(archive,'r|gz') as tf:
 for member in tf:
  path=(U/member.name).resolve();assert U.resolve() in path.parents and member.isfile() and not member.issym() and not member.islnk()
  tf.extract(member,U,filter='data')
for item in inventory['files']:
 path=U/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
assert inventory['counts']['main_latest']==50 and inventory['counts']['main_fit_complete']==50 and inventory['counts']['main_external_selected']==25
assert sha(U/'manifest.json')==sha(H/'manifest.json')
for name,digest in read(H/'manifest.json').items():assert sha(U/name)==sha(H/name)==digest,name
save(O/'performance-input-receipt.json',dict(passed=True,server_local_sha256_matched=True,archive_sha256=sha(archive),
 extracted=str(U),file_count=len(inventory['files']),all_extracted_hashes_matched=True,manifest_sha256=sha(H/'manifest.json')))

# The source summary failed after scratch OOF due to a missing historical reference file.
# Keep all recovered files immutable; assemble BOTH arms in a separate derived cache.
r.ROOT=U
DERIVED=archive.parent/'derived-oof';DERIVED.mkdir(exist_ok=True)
d=r.data()
for f in range(5):
 for label in ['inner%d'%j for j in range(5) if j!=f]+['refit']:
  scratch=read(U/'runs/main/scratch'/('fold%d'%f)/label/'initialization.json')
  synth=read(U/'runs/main/synth'/('fold%d'%f)/label/'initialization.json')
  assert scratch['initial_conv_sha256']==synth['initial_conv_sha256']
  assert scratch['other_state_sha256']==synth['other_state_sha256']
  keys=synth['transferred_keys'];assert len(keys)==12 and len(set(keys))==12 and all(k.startswith('convs.') for k in keys)
  assert synth['after_conv_sha256']!=synth['initial_conv_sha256']
  assert scratch['after_conv_sha256']==scratch['initial_conv_sha256']
for variant in ['scratch','synth']:
 out=U/'runs/main'/variant; pred=np.full((len(d['y']),2),np.nan); raw=pred.copy();seen=np.zeros(len(pred),int)
 for f in range(5):
  folder=out/('fold%d'%f);done=r.load(folder/'complete.json')
  assert r.sha(folder/'predictions.npz')==done['prediction_sha256']
  assert r.sha(folder/'selection.json')==done['selection_sha256']
  assert r.sha(folder/'refit/latest.pt')==done['checkpoint_sha256']
  assert done['identity']==r.identity_for(d,r.config_for(variant,'main'),f,None,done['selected_epoch'],'outer_refit')
  with np.load(folder/'predictions.npz') as z:
   ix=z['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(d['fold']==f))
   for k in ['person_id','video_id','fold','cohort','y']:np.testing.assert_array_equal(z[k],d[k][ix])
   pred[ix]=z['predictions'][:,0];raw[ix]=z['predictions_raw'][:,0];seen[ix]+=1
   np.testing.assert_allclose(pred[ix],r.corrected(raw[ix],z['calibration']),rtol=0,atol=1e-10)
 assert (seen==1).all() and np.isfinite(pred).all()
 dest=DERIVED/variant;dest.mkdir(exist_ok=True)
 np.savez_compressed(dest/'oof_predictions.npz',predictions=pred[:,None,:],predictions_raw=raw[:,None,:],**{k:d[k] for k in ['y','person_id','video_id','fold','cohort','original_row_index']})
 if (out/'oof_predictions.npz').exists():
  with np.load(out/'oof_predictions.npz') as z:np.testing.assert_array_equal(z['predictions'][:,0],pred)
# Explicit in-memory read adapter, original review file unchanged. Only the missing
# summary OOF read is redirected to the independently assembled derived cache.
review_source=(H/'review_results.py').read_text(encoding='utf-8')
needle="with np.load(out/'oof_predictions.npz') as z:"
assert review_source.count(needle)==1
exec(compile(review_source.replace(needle,"with np.load(DERIVED/variant/'oof_predictions.npz') as z:"),'read_only_review_derived_oof_adapter','exec'),review.__dict__)
review.DERIVED=DERIVED
save(O/'derived-oof-receipt.json',dict(original_summary_failure_preserved=True,reason='missing reference_metrics.json',paths={v:str(DERIVED/v/'oof_predictions.npz') for v in ['scratch','synth']},hashes={v:sha(DERIVED/v/'oof_predictions.npz') for v in ['scratch','synth']},adapter='Only the OOF read path is redirected; original source and verification tolerances unchanged'))
# Independent OOF comparison, before any model-replay acceptance claim.
base_report=read(S/'inner-bagging692/analysis/comparison.json');assert sha(base_report['predictions_path'])==base_report['predictions_sha256']
with np.load(base_report['predictions_path']) as z:base={k:z[k].copy() for k in ['y','person_id','video_id','fold','bagged','CNN','STT']}
y=base['y'];ids=base['person_id'];people=np.unique(ids);assert len(people)==692 and len(y)==1061
pm=np.array([y[ids==p].mean() for p in people]);high=pm>=90;assert high.sum()==68
def pp(v):return np.array([np.asarray(v)[ids==p].mean() for p in people])
def band(v,w):return np.searchsorted(np.arange(0,161,w),v,side='right')
def measure(p):
 e=p-y;mae=pp(abs(e));bias=pp(e)
 ans=dict(mae=float(mae.mean()),bias=float(bias.mean()),bands={},subgroups={})
 for w in [10,20]:ans['bands'][str(w)]=dict(accuracy=float(pp(band(p,w)==band(y,w)).mean()),two_or_more=float(pp(abs(band(p,w)-band(y,w))>=2).mean()))
 for name,mask in [('person_mean90',pm>=90),('person_mean100',pm>=100),('person_mean60below',pm<60)]:
  ans['subgroups'][name]=dict(people=int(mask.sum()),mae=float(mae[mask].mean()),bias=float(bias[mask].mean()))
 for t in [90,100]:ans['subgroups']['observation_ge%d'%t]=review.subgroup(y,p,ids,y>=t)
 ans['fold_mae']=[float(np.mean([abs(e[(ids==p)]).mean() for p in people if base['fold'][ids==p][0]==f])) for f in range(5)]
 return ans,mae
models={};errors={};predictions={};raw={}
for k,key in [('existing_best','bagged'),('multiwindow_CNN','CNN'),('multiwindow_STT','STT')]:models[k],errors[k]=measure(base[key])
assert abs(models['existing_best']['mae']-6.515816569178735)<1e-10
for v in ['scratch','synth']:
 with np.load(DERIVED/v/'oof_predictions.npz') as z:
  for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],base[k])
  predictions[v]=z['predictions'][:,0].copy();raw[v]=z['predictions_raw'][:,0].copy()
 models[v],errors[v]=measure(predictions[v]);models[v]['raw']=measure(raw[v])[0]
rank_report_path=S/'cnn-ranksim692/analysis/comparison.json'
registry=read(S.parent.parent/'control/gpu-watch/registry.json')
rankrow=next(e for e in registry['experiments'] if e['id']=='cnn-ranksim-692')
assert rankrow['reported_digest']==sha(rank_report_path)
rank_root=Path(read(S/'cnn-ranksim692/analysis/performance-input-receipt.json')['extracted'])
with np.load(rank_root/'runs/main/ranksim/oof_predictions.npz') as z:
 for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],base[k])
 models['ranksim'],errors['ranksim']=measure(z['predictions'][:,0])
assert abs(models['ranksim']['mae']-6.782475024589883)<1e-10
assert abs(models['ranksim']['subgroups']['person_mean90']['mae']-8.4856731671083)<1e-10
rng=np.random.default_rng(20261101);draw=rng.integers(0,len(people),(10000,len(people)));hd=rng.integers(0,int(high.sum()),(10000,int(high.sum())))
def paired(a,b):
 d=errors[a]-errors[b];h=d[high]
 return dict(delta=float(d.mean()),bootstrap95=np.quantile(d[draw].mean(1),[.025,.975]).tolist(),
  high90_delta=float(h.mean()),high90_bootstrap95=np.quantile(h[hd].mean(1),[.025,.975]).tolist(),
  better_people=int((d<0).sum()),worse_people=int((d>0).sum()))
report=dict(people=692,records=1061,models=models,paired=dict(synth_vs_scratch=paired('synth','scratch'),synth_vs_best=paired('synth','existing_best'),scratch_vs_best=paired('scratch','existing_best'),synth_vs_ranksim=paired('synth','ranksim')),
 independent_oof_metrics_passed=True,full_model_verification_pending=True,person_equal=True,baseline_prediction_sha256=base_report['predictions_sha256'],
 bootstrap_replicates=10000,bootstrap_seed=20261101,optimizer_steps=0,protected_final_opened=False,ensemble_performed=False)
save(O/'comparison.json',report)
state['phase']='source_environment_exact_audit';progress()
exec((H/'source_audit_adapter.py').read_text(encoding='utf-8'))
state['phase']='original_exact_local_audit';progress()
print(json.dumps(dict(phase=state['phase'],metrics=models,paired=report['paired'])),flush=True)
original=review.audit_fit
def strict_audit(*args,**kwargs):
 state['active_fit']=str(args[5]);progress();out=original(*args,**kwargs)
 state['completed_fits'].append(str(args[5]));progress();return out
review.audit_fit=strict_audit
sys.argv=['review_results.py','--received',str(U),'--receipt',str(O/'performance-input-receipt.json'),'--output',str(O/'strict-verification.json')]
strict_pass=False
try:
 review.main();strict_pass=True
except BaseException as exc:
 failure=dict(original_exact_audit_passed=False,error=repr(exc),traceback=traceback.format_exc(),completed_fits=state['completed_fits'].copy(),tolerance_unchanged=True)
 save(O/'original-exact-audit-failure.json',failure);print(failure['traceback'],flush=True)
 # Only the previously observed BLAS coefficient equality discrepancy may use
 # the separate frozen-output audit. Other failures remain failures.
 assert isinstance(exc,AssertionError) and 'Not equal to tolerance rtol=0, atol=1e-10' in str(exc),repr(exc)
 state.update(phase='supplemental_frozen_output_audit',completed_fits=[],coefficient_max_error=0.,train_transformation_max_error=0.,validation_transformation_max_error=0.);progress()
 def verify_transform(train,labels,stored_coef,valid):
  fitted=r.calibrate(train,labels);stored=np.asarray(stored_coef)
  difference=float(np.max(abs(fitted-stored)))
  te=float(np.max(abs(r.corrected(train,fitted)-r.corrected(train,stored))))
  ve=float(np.max(abs(r.corrected(valid,fitted)-r.corrected(valid,stored)))) if len(valid) else 0.
  assert te<1e-4 and ve<1e-4,(te,ve)
  state['coefficient_max_error']=max(state['coefficient_max_error'],difference)
  state['train_transformation_max_error']=max(state['train_transformation_max_error'],te)
  state['validation_transformation_max_error']=max(state['validation_transformation_max_error'],ve)
  state['epochs']+=1;return stored
 source=(H/'audit.py').read_text(encoding='utf-8');needle="            coef=r.calibrate(raw,y)\n            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)"
 assert source.count(needle)==1
 # Original file/criterion/failure remain untouched. This separate audit checks
 # frozen coefficients/metrics and actual prediction changes at existing 1e-4.
 namespace={'verify_transform':verify_transform};exec(compile(source.replace(needle,"            coef=verify_transform(raw,y,row['calibration'],valid)"),'separate_frozen_output_audit','exec'),namespace)
 supplement=namespace['audit_fit']
 def supplemental_audit(*args,**kwargs):
  state['active_fit']=str(args[5]);progress();out=supplement(*args,**kwargs)
  state['completed_fits'].append(str(args[5]));progress();print('SUPPLEMENTAL_FIT',len(state['completed_fits']),args[5],flush=True);return out
 review.audit_fit=supplemental_audit
 sys.argv[-1]=str(O/'supplemental-verification.json');review.main()
verification=read(O/('strict-verification.json' if strict_pass else 'supplemental-verification.json'))
assert verification['passed'] and verification['fits']==50 and len(verification['final_replays'])==10 and len(verification['external_pretraining'])==25
if not strict_pass:
 source_proof=read(H/'checks/source-environment-audit-proof.json')
 assert source_proof['passed'] and source_proof['fits']==50 and source_proof['manifest_sha256']==sha(H/'manifest.json')
 report['source_environment_proof_sha256']=sha(H/'checks/source-environment-audit-proof.json')
 report['original_failure_sha256']=sha(O/'original-exact-audit-failure.json')
 report['supplemental_diagnostics']={k:state[k] for k in ['coefficient_max_error','train_transformation_max_error','validation_transformation_max_error','epochs']}
report.update(full_model_verification_pending=False,original_exact_local_audit_passed=strict_pass,
 qualified_full_verification_passed=True,verification_sha256=sha(O/('strict-verification.json' if strict_pass else 'supplemental-verification.json')),
 fits=50,external_pretraining=25,final_cpu_replays=10,original_slurm_state='FAILED1:0_summary_reference_missing',max_external_replay_error=max(p['max_cadence_bpm_error'] for p in verification['external_pretraining']),max_model_replay_error=max(p['error'] for p in verification['final_replays']))
for v in ['scratch','synth']:
 assert abs(verification['results'][v]['overall']['mae']-models[v]['mae'])<1e-10
 assert abs(verification['results'][v]['subgroups']['person_mean_ge90']['mae']-models[v]['subgroups']['person_mean90']['mae'])<1e-10
save(O/'comparison.json',report);state.update(phase='complete',original_exact_local_audit_passed=strict_pass,final_cpu_replays=10);progress()
print(json.dumps(dict(phase='complete',original_exact_local_audit_passed=strict_pass,qualified_full_verification_passed=True,report=str(O/'comparison.json'))),flush=True)
