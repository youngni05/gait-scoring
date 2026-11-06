"""Finish this terminated MAE submission: immutable recovery, full audit, frozen comparison; no training."""
from pathlib import Path
import os,sys,json,time,datetime,hashlib,subprocess,uuid,traceback
H=Path(__file__).resolve().parent;G=H.parents[2];D=Path('D:/gait-research-data/D_P')/H.relative_to(H.parents[3]);A=H/'analysis'
OWNER='01a18b12-4a31-7283-a35d-17f28017111a';JOBS=['35847','35848','35849']
SNAP=G/'control/gpu-watch/snapshot-20261105-112223.json'
PLAN='91cc95d5097e926196891461e796e959c0fb0b2f4c45fb5e76eae2a1ac8c5afc'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for i in range(12):
  try:os.replace(t,p);return
  except PermissionError:
   if i==11:raise
   time.sleep(.1*(i+1))
def ownership():
 reg=read(G/'control/gpu-watch/registry.json');assert reg['thread_id']==OWNER and reg['automation_id']=='gdi-2'
 e=next(e for e in reg['experiments'] if e['id']=='cnn-mae-point-692');assert e['jobs']==JOBS and e['remote']=='/abr/RESEARCH_USER/gdi-cnn-mae-point692-v2-20261104'
 assert sha(H/'protocol.json')==e['plan_sha256']==PLAN and sha(H/'manifest.json')==e['manifest_sha256']
 assert reg['iterative_research']['new_gpu_training_allowed'] is False and reg['closeout_policy']['new_training'] is False
 return reg,e
def phase(name,extra=None):
 state.update(phase=name,updated_at=now(),**(extra or {}));save(A/'continuation-process.json',state)
 reg,e=ownership();reg['iterative_research']['phase']='mae_point_'+name
 if name=='retrieval_running':e['retrieval_state']='full_recovery_running'
 if name=='audit_running':e['performance_review_state']='full_cpu_audit_running'
 if name=='comparison_running':e['performance_review_state']='qualified_full_verified_comparison_running'
 if name=='comparison_ready_for_report':e['performance_review_state']='qualified_full_verified_comparison_ready_unreported'
 save(G/'control/gpu-watch/registry.json',reg)
 rp=G/'control/gpu-watch/iterative-research/round-20261104-mae-point.json';r=read(rp);assert r['jobs']==JOBS;r.update(phase=name,continuation_process=str(A/'continuation-process.json'));save(rp,r)
def child(driver,stage,args=()):
 ownership();log=D/(stage+'.log');err=D/(stage+'.err.log')
 command=[sys.executable,'-u','-B',str(H/driver),*map(str,args)]
 metadata={}
 if stage=='full-receive':
  # Operational SSH wait only. Keep original recovery/archiving/validation source byte-for-byte.
  shim="import sys;import receive_full as m\noriginal=m.helper.command\ndef extended(args,input=None,timeout=1800):\n return original(args,input=input,timeout=7200 if timeout==1800 and args[:len(m.helper.SSH)]==m.helper.SSH else timeout)\nm.helper.command=extended\nm.main()\n"
  command=[sys.executable,'-u','-B','-c',shim,*map(str,args)];metadata=dict(ssh_packaging_timeout_seconds=7200,shim_sha256=hashlib.sha256(shim.encode()).hexdigest())
 with log.open('wb') as out,err.open('wb') as errors:
  p=subprocess.Popen(command,cwd=str(H),stdout=out,stderr=errors,creationflags=subprocess.CREATE_NO_WINDOW)
  receipt=dict(pid=p.pid,active=True,started_at=now(),driver=driver,script_sha256=sha(H/driver),protocol_sha256=sha(H/'protocol.json'),manifest_sha256=sha(H/'manifest.json'),snapshot=str(SNAP),snapshot_sha256=sha(SNAP),stdout=str(log),stderr=str(err),**metadata)
  if (H/'checks/full-receipt.json').exists():receipt['input_receipt_sha256']=sha(H/'checks/full-receipt.json')
  target=H/'checks/full-receive-process.json' if stage=='full-receive' else A/(stage+'-launch-receipt.json')
  save(target,receipt);phase('retrieval_running' if stage=='full-receive' else stage+'_running',dict(child=receipt));code=p.wait()
  receipt.update(active=False,ended_at=now(),exit_code=code);save(target,receipt)
 assert code==0,(driver,code,str(err))
def main():
 global state
 A.mkdir(exist_ok=True);D.mkdir(parents=True,exist_ok=True);ownership()
 state=dict(pid=os.getpid(),active=True,started_at=now(),new_training=False,jobs=JOBS,script_sha256=sha(__file__),protocol_sha256=PLAN,driver_sha256={n:sha(H/n) for n in ['receive_full.py','audit_received.py','compare_completed.py']})
 lock=A/'closeout-controller.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps(state).encode());os.close(fd)
 try:
  snap=read(SNAP);s=snap['experiments']['cnn-mae-point-692'];assert all(s['jobs'][j]==dict(state='COMPLETED',exit_code='0:0') for j in JOBS) and s['completed_fits']==50 and s['completed_outer_folds']==10 and s['summary_exists'] and s['summary_complete']
  phase('closeout_start')
  if not (H/'checks/full-receipt.json').exists():child('receive_full.py','full-receive',['--snapshot',SNAP])
  receipt=read(H/'checks/full-receipt.json');assert receipt['passed'] and receipt['source_succeeded'] and receipt['full_fit_counts_present'] and receipt['all_archived_file_hashes_verified']
  if not (A/'full-verification.json').exists():child('audit_received.py','audit')
  v=read(A/'full-verification.json');assert v['qualified_full_verification_passed'] and v['fits']==50 and v['final_cpu_replays']==10 and v['max_model_replay_error']<1e-4 and v['receipt_sha256']==sha(H/'checks/full-receipt.json')
  if not (H/'ensemble/analysis/comparison.json').exists():child('compare_completed.py','comparison')
  c=read(H/'ensemble/analysis/comparison.json');q=read(H/'ensemble/analysis/verification.json');assert c['passed'] and q['passed'] and q['comparison_sha256']==sha(H/'ensemble/analysis/comparison.json') and c['full_verification_sha256']==sha(A/'full-verification.json')
  phase('comparison_ready_for_report',dict(full_verification_sha256=sha(A/'full-verification.json'),comparison_sha256=sha(H/'ensemble/analysis/comparison.json'),reporting_pending=True))
 except BaseException:
  state.update(failed=True,error=traceback.format_exc());save(A/'continuation-failure.json',state);raise
 finally:
  state.update(active=False,ended_at=now());save(A/'continuation-process.json',state)
  if read(lock)['pid']==os.getpid():lock.unlink()
if __name__=='__main__':main()
