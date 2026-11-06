"""Continue this existing receipt, then gated CPU audit and frozen comparison only."""
from pathlib import Path
import os,sys,json,time,datetime,hashlib,subprocess,ctypes,uuid,traceback
H=Path(__file__).resolve().parent;G=H.parents[2];D=Path('D:/gait-research-data/D_P')/H.relative_to(H.parents[3]);A=H/'analysis'
OWNER='01a18b12-4a31-7283-a35d-17f28017111a';JOBS=['34721','34722','34723']
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
 e=next(e for e in reg['experiments'] if e['id']=='cnn-huber-point-692');assert e['jobs']==JOBS
 assert sha(H/'protocol.json')==e['plan_sha256']=='4a1320fabe90278502cddad57f0dd71b0783a999672d01ebc8b49f06261deee3'
 assert sha(H/'manifest.json')==e['manifest_sha256'];return reg,e
def phase(name,extra=None):
 state.update(phase=name,updated_at=now(),**(extra or {}));save(A/'continuation-process.json',state)
 reg,e=ownership();reg['iterative_research']['phase']='huber_point_'+name
 if name=='audit_running':e['performance_review_state']='full_cpu_audit_running'
 if name=='comparison_running':e['performance_review_state']='qualified_full_verified_comparison_running'
 if name=='comparison_ready_for_report':e['performance_review_state']='qualified_full_verified_comparison_ready_unreported'
 save(G/'control/gpu-watch/registry.json',reg)
 rp=G/'control/gpu-watch/iterative-research/round-20261103-huber-point.json';r=read(rp);assert r['jobs']==JOBS;r.update(phase=name,continuation_process=str(A/'continuation-process.json'));save(rp,r)
def child(driver,stage):
 ownership();log=D/(stage+'.log');err=D/(stage+'.err.log')
 with log.open('wb') as out,err.open('wb') as errors:
  p=subprocess.Popen([sys.executable,'-u','-B',str(H/driver)],cwd=str(H),stdout=out,stderr=errors,creationflags=subprocess.CREATE_NO_WINDOW)
  receipt=dict(pid=p.pid,active=True,started_at=now(),driver=driver,script_sha256=sha(H/driver),protocol_sha256=sha(H/'protocol.json'),manifest_sha256=sha(H/'manifest.json'),input_receipt_sha256=sha(H/'checks/full-receipt.json'),stdout=str(log),stderr=str(err))
  save(A/(stage+'-launch-receipt.json'),receipt);phase(stage+'_running',dict(child=receipt));code=p.wait()
  receipt.update(active=False,ended_at=now(),exit_code=code);save(A/(stage+'-launch-receipt.json'),receipt)
 assert code==0,(driver,code,str(err))
def wait_receiver():
 receipt=read(H/'checks/full-receive-process.json');assert receipt['script_sha256']==sha(H/'receive_full.py') and receipt['manifest_sha256']==sha(H/'manifest.json')
 # Keep a handle to the exact original process so PID reuse cannot prolong a wait.
 k=ctypes.WinDLL('kernel32',use_last_error=True);k.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];k.OpenProcess.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];k.CloseHandle.argtypes=[ctypes.c_void_p]
 handle=k.OpenProcess(0x100000|0x1000,False,receipt['pid'])
 if handle:
  class FT(ctypes.Structure):_fields_=[('lo',ctypes.c_ulong),('hi',ctypes.c_ulong)]
  values=[FT() for _ in range(4)];k.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.POINTER(FT)]*4
  assert k.GetProcessTimes(handle,*[ctypes.byref(v) for v in values])
  created=((values[0].hi<<32)+values[0].lo)/1e7-11644473600
  expected=datetime.datetime.fromisoformat(receipt['started_at'].replace('Z','+00:00')).timestamp();assert abs(created-expected)<.1
  deadline=time.monotonic()+2400
  try:
   while k.WaitForSingleObject(handle,2000)==258:assert time.monotonic()<deadline,'Existing receiver still active; do not duplicate'
  finally:k.CloseHandle(handle)
 full=read(H/'checks/full-receipt.json');assert full['passed'] and full['source_succeeded'] and full['full_fit_counts_present'] and full['all_archived_file_hashes_verified']
 assert sha(full['archive'])==full['archive_sha256'] and sha(full['inventory'])==full['inventory_sha256']
 last=json.loads(Path(receipt['stdout']).read_text(encoding='utf-8').strip().splitlines()[-1]);assert last['passed']
 receipt.update(active=False,passed=True,finished_observed_at=now(),full_receipt_sha256=sha(H/'checks/full-receipt.json'));save(H/'checks/full-receive-process.json',receipt)
state=dict(pid=os.getpid(),active=True,started_at=now(),new_training=False,script_sha256=sha(__file__),protocol_sha256=sha(H/'protocol.json'),driver_sha256={n:sha(H/n) for n in ['audit_received.py','compare_completed.py']})
try:
 A.mkdir(exist_ok=True);ownership();phase('waiting_existing_receiver');wait_receiver()
 if not (A/'full-verification.json').exists():child('audit_received.py','audit')
 v=read(A/'full-verification.json');assert v['qualified_full_verification_passed'] and v['fits']==50 and v['final_cpu_replays']==10 and v['max_model_replay_error']<1e-4
 assert v['receipt_sha256']==sha(H/'checks/full-receipt.json')
 if not (H/'ensemble/analysis/comparison.json').exists():child('compare_completed.py','comparison')
 c=read(H/'ensemble/analysis/comparison.json');q=read(H/'ensemble/analysis/verification.json');assert c['passed'] and q['passed'] and q['comparison_sha256']==sha(H/'ensemble/analysis/comparison.json')
 phase('comparison_ready_for_report',dict(full_verification_sha256=sha(A/'full-verification.json'),comparison_sha256=sha(H/'ensemble/analysis/comparison.json'),reporting_pending=True))
except BaseException:
 state.update(failed=True,error=traceback.format_exc());save(A/'continuation-failure.json',state);raise
finally:
 state.update(active=False,ended_at=now());save(A/'continuation-process.json',state)
