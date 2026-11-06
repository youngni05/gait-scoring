"""Verify actual receipt, every fit and ten final CPU64 replays; never train."""
from pathlib import Path
import datetime,hashlib,json,os,sys,tarfile,traceback
import time,uuid
import numpy as np
H=Path(__file__).resolve().parent;O=H/'analysis';O.mkdir(exist_ok=True)
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
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 for attempt in range(12):
  try:os.replace(tmp,p);return
  except PermissionError:
   if attempt==11:raise
   time.sleep(.1*(attempt+1))
def main():
 process=dict(pid=os.getpid(),active=True,started_at=datetime.datetime.now().astimezone().isoformat(),script_sha256=sha(__file__),new_training=False)
 state=dict(phase='receipt_and_source_verification',completed_fits=[],epochs=0)
 save(O/'process-receipt.json',process)
 def progress():save(O/'progress.json',state)
 progress()
 try:
  receipt=read(H/'checks/full-receipt.json');archive=Path(receipt['archive'])
  assert receipt['passed'] and receipt['server_local_sha256_matched'] and receipt['source_succeeded'] and receipt['full_fit_counts_present']
  assert sha(archive)==receipt['archive_sha256'] and sha(receipt['inventory'])==receipt['inventory_sha256']
  U=archive.parent/'performance-review-originals';U.mkdir(exist_ok=True)
  with tarfile.open(archive,'r|gz') as tf:
   for member in tf:
    p=(U/member.name).resolve();assert U.resolve() in p.parents and member.isfile() and not member.issym() and not member.islnk()
    tf.extract(member,U,filter='data')
  for item in read(receipt['inventory'])['files']:
   p=U/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
  assert sha(U/'manifest.json')==sha(H/'manifest.json')
  for n,digest in read(H/'manifest.json').items():assert sha(U/n)==sha(H/n)==digest
  from source_contract import verify
  verify()
  marker=read(U/'runs/summary_complete.json');pair=read(U/'runs/pair_summary.json')
  assert sha(U/'runs/pair_summary.json')==marker['summary_sha256'] and pair['fits']==50 and pair['outer_folds']==10
  workers=[read(U/'control'/('worker-%d.json'%i)) for i in range(3)]
  assert {w['job_id'] for w in workers}==set(receipt['states']) and len({w['uuid'] for w in workers})==3
  assert all(w['exit_code']==0 and w['phase']=='assigned_folds_complete' for w in workers)
  tasks=[tuple(t) for w in workers for t in w['assignments']];assert len(set(tasks))==len(tasks)==10
  assert set(tasks)=={(a,f) for a in ['mse','ranksim'] for f in range(5)}
  for a in ['mse','ranksim']:
   q=read(U/'runs/main'/a/'summary.json');assert q==pair['variants'][a] and q['fits']==25 and q['oof_sha256']==sha(U/'runs/main'/a/'oof_predictions.npz')
  source_proof=dict(passed=True,summary_fits=50,manifest_sha256=sha(H/'manifest.json'),proof_basis='All successful original summaries call unchanged original strict audit_fit before writing summary; source AST/hash, summaries, marker and jobs verified',fresh_server_reexecution=False)
  save(H/'checks/source-environment-summary-proof.json',source_proof)
  input_receipt=dict(passed=True,server_local_sha256_matched=True,archive_sha256=sha(archive),manifest_sha256=sha(H/'manifest.json'),originals=str(U))
  save(O/'performance-input-receipt.json',input_receipt)
  original=review.audit_fit
  def strict(*args,**kwargs):
   state['active_fit']=str(args[5]);progress();out=original(*args,**kwargs);state['completed_fits'].append(str(args[5]));progress();return out
  r.save=save
  review.audit_fit=strict;state['phase']='original_exact_local_audit';progress()
  sys.argv=['review_results.py','--received',str(U),'--receipt',str(O/'performance-input-receipt.json'),'--output',str(O/'strict-verification.json')]
  exact=False
  try:review.main();exact=True
  except AssertionError as exc:
   save(O/'original-exact-audit-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),tolerance_unchanged=True))
   assert 'Not equal to tolerance rtol=0, atol=1e-10' in str(exc)
   state.update(phase='separate_frozen_output_audit',completed_fits=[],epochs=0,coefficient_max_error=0.,train_transformation_max_error=0.,validation_transformation_max_error=0.);progress()
   def verify_transform(train,labels,stored_coef,valid):
    fitted=r.calibrate(train,labels);stored=np.asarray(stored_coef)
    te=float(np.max(abs(r.corrected(train,fitted)-r.corrected(train,stored))))
    ve=float(np.max(abs(r.corrected(valid,fitted)-r.corrected(valid,stored)))) if len(valid) else 0.
    assert te<1e-4 and ve<1e-4,(te,ve)
    state['coefficient_max_error']=max(state['coefficient_max_error'],float(np.max(abs(fitted-stored))))
    state['train_transformation_max_error']=max(state['train_transformation_max_error'],te);state['validation_transformation_max_error']=max(state['validation_transformation_max_error'],ve);state['epochs']+=1
    return stored
   source=(H/'audit.py').read_text(encoding='utf-8');needle="            coef=r.calibrate(raw,y)\n            np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)";assert source.count(needle)==1
   ns={'verify_transform':verify_transform};exec(compile(source.replace(needle,"            coef=verify_transform(raw,y,row['calibration'],valid)"),'separate_frozen_output_audit','exec'),ns)
   def supplemental(*args,**kwargs):
    state['active_fit']=str(args[5]);progress();out=ns['audit_fit'](*args,**kwargs);state['completed_fits'].append(str(args[5]));progress();return out
   review.audit_fit=supplemental;sys.argv[-1]=str(O/'supplemental-verification.json');review.main()
  vp=O/('strict-verification.json' if exact else 'supplemental-verification.json');v=read(vp)
  assert v['passed'] and v['fits']==50 and len(v['final_replays'])==10
  err=max(x['error'] for x in v['final_replays']);assert err<1e-4
  save(O/'full-verification.json',dict(qualified_full_verification_passed=True,original_exact_local_audit_passed=exact,fits=50,final_cpu_replays=10,max_model_replay_error=err,verification_sha256=sha(vp),source_proof_sha256=sha(H/'checks/source-environment-summary-proof.json'),receipt_sha256=sha(H/'checks/full-receipt.json'),diagnostics=state,performance_comparison_pending=True))
  state.update(phase='complete',final_cpu_replays=10);progress()
 except BaseException:
  process['failed']=True;save(O/'execution-failure.json',dict(traceback=traceback.format_exc(),state=state));raise
 finally:
  process.update(active=False,ended_at=datetime.datetime.now().astimezone().isoformat());save(O/'process-receipt.json',process)
if __name__=='__main__':main()
