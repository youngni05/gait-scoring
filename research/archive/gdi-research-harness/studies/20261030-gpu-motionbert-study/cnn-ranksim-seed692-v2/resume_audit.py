"""Resume the same SHA-bound CPU audit; keep historical failures and tolerances."""
from pathlib import Path
import datetime,hashlib,json,os,sys,time,uuid,traceback
import numpy as np
H=Path(__file__).resolve().parent;O=H/'analysis'
os.environ.setdefault('GDI_DATA_PATH',str(H.parent/'cnn-record-loss692/bundle/data.npz'))
import run as r
import review_results as review

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for attempt in range(12):
  try:os.replace(t,p);return
  except PermissionError:
   if attempt==11:raise
   time.sleep(.1*(attempt+1))
def main():
 receipt=read(H/'checks/full-receipt.json');U=Path(receipt['archive']).parent/'performance-review-originals'
 old=O/'attempt-20261103-090511'
 if not old.exists():
  old.mkdir()
  for n in ['process-receipt.json','launch-receipt.json','progress.json','execution-failure.json','original-exact-audit-failure.json','performance-input-receipt.json']:
   p=O/n
   if p.exists():(old/n).write_bytes(p.read_bytes())
 prior=read(old/'progress.json');failure=read(old/'execution-failure.json');launch=read(old/'launch-receipt.json')
 assert read(old/'process-receipt.json')['active'] is False
 assert read(old/'process-receipt.json')['script_sha256']==sha(H/'audit_received.py')
 assert launch['archive_sha256']==receipt['archive_sha256']
 assert prior['phase']=='separate_frozen_output_audit'
 assert 'PermissionError' in failure['traceback']
 assert prior['completed_fits']==failure['state']['completed_fits']
 assert 'atol=1e-10' in read(old/'original-exact-audit-failure.json')['error']
 prior_complete=set(prior['completed_fits'])
 assert len(prior_complete)==18
 process=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),
  script_sha256=sha(__file__),archive_sha256=receipt['archive_sha256'],
  receipt_sha256=sha(H/'checks/full-receipt.json'),data_sha256=sha(os.environ['GDI_DATA_PATH']),
  source_manifest_sha256=sha(H/'manifest.json'),new_training=False,resume_source=str(old))
 state=dict(phase='revalidate_identical_inputs',completed_fits=[],reused_successful_fits=[],
  epochs=prior['epochs'],coefficient_max_error=prior['coefficient_max_error'],
  train_transformation_max_error=prior['train_transformation_max_error'],
  validation_transformation_max_error=prior['validation_transformation_max_error'])
 save(O/'process-receipt.json',process)
 try:
  save(O/'progress.json',state)
  assert process['data_sha256']=='7c1912cd27c63a82ed0d0324f98c0c0fae9a608174d52bd7a502f05e9a8c7a1e'
  assert receipt['passed'] and receipt['source_succeeded'] and receipt['full_fit_counts_present']
  assert sha(receipt['archive'])==receipt['archive_sha256'] and sha(receipt['inventory'])==receipt['inventory_sha256']
  inventory=read(receipt['inventory'])
  for item in inventory['files']:
   p=U/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
  assert sha(U/'manifest.json')==sha(H/'manifest.json')
  for n,h in read(H/'manifest.json').items():assert sha(U/n)==sha(H/n)==h,n
  from source_contract import verify
  assert verify()['passed']
  source_proof=read(H/'checks/source-environment-summary-proof.json')
  assert source_proof['passed'] and source_proof['summary_fits']==50 and source_proof['manifest_sha256']==sha(H/'manifest.json') and source_proof['fresh_server_reexecution'] is False
  marker=read(U/'runs/summary_complete.json');pair=read(U/'runs/pair_summary.json')
  assert sha(U/'runs/pair_summary.json')==marker['summary_sha256'] and pair['fits']==50 and pair['outer_folds']==10
  assert all(s['state']=='COMPLETED' and s['exit_code']=='0:0' for s in receipt['states'].values())
  workers=[read(U/'control'/('worker-%d.json'%i)) for i in range(3)]
  assert len({w['uuid'] for w in workers})==3 and {w['job_id'] for w in workers}==set(receipt['states'])
  assert all(w['exit_code']==0 and w['phase']=='assigned_folds_complete' for w in workers)
  for a in ['mse','ranksim']:
   q=read(U/'runs/main'/a/'summary.json')
   assert q==pair['variants'][a] and q['fits']==25 and q['oof_sha256']==sha(U/'runs/main'/a/'oof_predictions.npz')
  # Previous successful returns are reused only after all original file hashes
  # and the exact previous driver/input receipt have been revalidated.
  assert all(U.resolve() in Path(p).resolve().parents for p in prior_complete)
  assert sum(len(read(Path(p)/'curve.json')) for p in prior_complete)==prior['epochs']
  save(O/'resume-input-proof.json',dict(passed=True,all_files_revalidated=len(inventory['files']),
   archive_sha256=receipt['archive_sha256'],prior_progress_sha256=sha(old/'progress.json'),
   prior_execution_failure_sha256=sha(old/'execution-failure.json'),prior_driver_sha256=sha(H/'audit_received.py'),
   prior_successful_audit_returns=18,prior_epochs=prior['epochs'],reuse_basis='Identical per-file hashes and SHA-bound prior successful audit_fit returns; no unverified fit reused'))
  input_receipt=dict(passed=True,server_local_sha256_matched=True,archive_sha256=receipt['archive_sha256'],manifest_sha256=sha(H/'manifest.json'),originals=str(U))
  save(O/'performance-input-receipt.json',input_receipt)
  def verify_transform(train,labels,stored_coef,valid):
   fitted=r.calibrate(train,labels);stored=np.asarray(stored_coef)
   te=float(np.max(abs(r.corrected(train,fitted)-r.corrected(train,stored))))
   ve=float(np.max(abs(r.corrected(valid,fitted)-r.corrected(valid,stored)))) if len(valid) else 0.
   assert te<1e-4 and ve<1e-4,(te,ve)
   state['coefficient_max_error']=max(state['coefficient_max_error'],float(np.max(abs(fitted-stored))))
   state['train_transformation_max_error']=max(state['train_transformation_max_error'],te)
   state['validation_transformation_max_error']=max(state['validation_transformation_max_error'],ve)
   state['epochs']+=1
   return stored
  source=(H/'audit.py').read_text(encoding='utf-8')
  needle="            coef=r.calibrate(raw,y)\n            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)"
  assert source.count(needle)==1
  ns={'verify_transform':verify_transform}
  exec(compile(source.replace(needle,"            coef=verify_transform(raw,y,row['calibration'],valid)"),'separate_frozen_output_audit','exec'),ns)
  def audit(*args,**kwargs):
   folder=str(args[5]);state['active_fit']=folder;save(O/'progress.json',state)
   if folder in prior_complete:
    out=read(Path(folder)/'curve.json');state['reused_successful_fits'].append(folder)
   else:out=ns['audit_fit'](*args,**kwargs)
   state['completed_fits'].append(folder);save(O/'progress.json',state)
   print('AUDIT_FIT',len(state['completed_fits']),folder,flush=True)
   return out
  review.audit_fit=audit
  # Use robust output saving only in the local analysis process.
  r.save=save
  state['phase']='separate_frozen_output_audit';save(O/'progress.json',state)
  sys.argv=['review_results.py','--received',str(U),'--receipt',str(O/'performance-input-receipt.json'),'--output',str(O/'supplemental-verification.json')]
  review.main()
  v=read(O/'supplemental-verification.json')
  assert v['passed'] and v['fits']==50 and len(v['final_replays'])==10
  assert len(state['completed_fits'])==len(set(state['completed_fits']))==50
  assert len(state['reused_successful_fits'])==18
  err=max(x['error'] for x in v['final_replays']);assert err<1e-4
  save(O/'full-verification.json',dict(qualified_full_verification_passed=True,original_exact_local_audit_passed=False,
   fits=50,final_cpu_replays=10,max_model_replay_error=err,verification_sha256=sha(O/'supplemental-verification.json'),
   source_proof_sha256=sha(H/'checks/source-environment-summary-proof.json'),receipt_sha256=sha(H/'checks/full-receipt.json'),
   diagnostics=state,resume_proof_sha256=sha(O/'resume-input-proof.json'),performance_comparison_pending=True))
  state.update(phase='complete',final_cpu_replays=10);save(O/'progress.json',state)
  print('FULL_CPU_AUDIT_PASSED',err,flush=True)
 except BaseException:
  process['failed']=True;save(O/'resume-execution-failure.json',dict(traceback=traceback.format_exc(),state=state));raise
 finally:
  process.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat());save(O/'process-receipt.json',process)
if __name__=='__main__':main()

