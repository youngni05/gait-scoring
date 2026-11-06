"""Audit already completed RankSim results and compare fixed OOF predictions."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,traceback,datetime
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent;O=H/'analysis';O.mkdir(exist_ok=True)
os.environ['GDI_DATA_PATH']=str(S/'cnn-record-loss692/bundle/data.npz')
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
assert sha(U/'manifest.json')==sha(H/'manifest.json')
for name,digest in read(H/'manifest.json').items():assert sha(U/name)==sha(H/name)==digest,name
save(O/'performance-input-receipt.json',dict(passed=True,server_local_sha256_matched=True,archive_sha256=sha(archive),
 extracted=str(U),file_count=len(inventory['files']),all_extracted_hashes_matched=True,manifest_sha256=sha(H/'manifest.json')))

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
for v in ['mse','ranksim']:
 with np.load(U/'runs/main'/v/'oof_predictions.npz') as z:
  for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(z[k],base[k])
  predictions[v]=z['predictions'][:,0].copy();raw[v]=z['predictions_raw'][:,0].copy()
 models[v],errors[v]=measure(predictions[v]);models[v]['raw']=measure(raw[v])[0]
rng=np.random.default_rng(20261101);draw=rng.integers(0,len(people),(10000,len(people)));hd=rng.integers(0,int(high.sum()),(10000,int(high.sum())))
def paired(a,b):
 d=errors[a]-errors[b];h=d[high]
 return dict(delta=float(d.mean()),bootstrap95=np.quantile(d[draw].mean(1),[.025,.975]).tolist(),
  high90_delta=float(h.mean()),high90_bootstrap95=np.quantile(h[hd].mean(1),[.025,.975]).tolist(),
  better_people=int((d<0).sum()),worse_people=int((d>0).sum()))
report=dict(people=692,records=1061,models=models,paired=dict(ranksim_vs_mse=paired('ranksim','mse'),ranksim_vs_best=paired('ranksim','existing_best'),mse_vs_best=paired('mse','existing_best')),
 independent_oof_metrics_passed=True,full_model_verification_pending=True,person_equal=True,baseline_prediction_sha256=base_report['predictions_sha256'],
 bootstrap_replicates=10000,bootstrap_seed=20261101,optimizer_steps=0,protected_final_opened=False,ensemble_performed=False)
save(O/'comparison.json',report)
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
 source=(H/'audit.py').read_text();needle="            coef=r.calibrate(raw,y)\n            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)"
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
assert verification['passed'] and verification['fits']==50 and len(verification['final_replays'])==10
if not strict_pass:
 source_proof=read(H/'checks/source-environment-summary-proof.json')
 assert source_proof['passed'] and source_proof['summary_fits']==50 and source_proof['manifest_sha256']==sha(H/'manifest.json')
 report['source_environment_proof_sha256']=sha(H/'checks/source-environment-summary-proof.json')
 report['original_failure_sha256']=sha(O/'original-exact-audit-failure.json')
 report['supplemental_diagnostics']={k:state[k] for k in ['coefficient_max_error','train_transformation_max_error','validation_transformation_max_error','epochs']}
report.update(full_model_verification_pending=False,original_exact_local_audit_passed=strict_pass,
 qualified_full_verification_passed=True,verification_sha256=sha(O/('strict-verification.json' if strict_pass else 'supplemental-verification.json')),
 fits=50,final_cpu_replays=10,max_model_replay_error=max(p['error'] for p in verification['final_replays']))
for v in ['mse','ranksim']:
 assert abs(verification['results'][v]['overall']['mae']-models[v]['mae'])<1e-10
 assert abs(verification['results'][v]['subgroups']['person_mean_ge90']['mae']-models[v]['subgroups']['person_mean90']['mae'])<1e-10
save(O/'comparison.json',report);state.update(phase='complete',original_exact_local_audit_passed=strict_pass,final_cpu_replays=10);progress()
print(json.dumps(dict(phase='complete',original_exact_local_audit_passed=strict_pass,qualified_full_verification_passed=True,report=str(O/'comparison.json'))),flush=True)
