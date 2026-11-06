"""One bounded local continuation for this submission's verified pilot gate."""
from pathlib import Path
import json,hashlib,sys,os,time,datetime,subprocess,traceback,uuid
from deploy import H,D,REMOTE,AUTH
OWNER='01a18b12-4a31-7283-a35d-17f28017111a';REG=H.parents[2]/'control/gpu-watch/registry.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,x):
 p=Path(p);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for i in range(12):
  try:os.replace(t,p);return
  except PermissionError:
   if i==11:raise
   time.sleep(.1*(i+1))
def current():
 r=read(REG);e=next(x for x in r['experiments'] if x['id']=='cnn-mae-point-692')
 assert r['thread_id']==OWNER and r['automation_id']=='gdi-2' and e['remote']==REMOTE and e['jobs']==read(H/'submission.json')['jobs'] and e['manifest_sha256']==sha(H/'manifest.json')
 return e
def query():
 source="from pathlib import Path\nimport json,time\nh=Path(%r)\nprint(json.dumps(dict(server_epoch=time.time(),workers={p.name:json.loads(p.read_text()) for p in (h/'control').glob('worker-*.json')},gate=json.loads((h/'control/approve-main.json').read_text()) if (h/'control/approve-main.json').exists() else None,archive=(h/'runs/pilot-review.tar.gz').exists(),local_approval=(h/'control/approve-main-local.json').exists())))\n"%REMOTE
 p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe',*AUTH,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=source,text=True,encoding='utf-8',capture_output=True,timeout=45)
 assert p.returncode==0,p.stderr;return json.loads(p.stdout)
def main():
 e=current();state=dict(pid=os.getpid(),active=True,started_at=now(),phase='waiting_server_pilot',jobs=e['jobs'],remote=REMOTE,script_sha256=sha(__file__),manifest_sha256=sha(H/'manifest.json'),protocol_sha256=sha(H/'protocol.json'),new_training=False,poll_seconds=30)
 path=H/'checks/pilot-watch-process.json';save(path,state);deadline=time.monotonic()+6*3600
 try:
  while True:
   current();assert time.monotonic()<deadline,'Bounded local pilot wait ended; preserve current jobs and inspect'
   try:q=query()
   except (subprocess.TimeoutExpired,OSError,AssertionError,ValueError) as error:
    state.update(last_connection_error=repr(error),last_checked_at=now());save(path,state);time.sleep(30);continue
   state.update(last_checked_at=now(),workers=q['workers']);save(path,state)
   assert not any(w['phase']=='training_or_summary_failed' for w in q['workers'].values()),'GPU worker failed; no automatic resubmit'
   if q['local_approval']:
    assert (H/'checks/pilot-receipt.json').exists(),'Remote approval not created by this verified local stage';break
   if q['gate'] and q['archive'] and len(q['workers'])==3 and all(w['phase']=='waiting_local_pilot_audit' for w in q['workers'].values()):
    assert q['gate']['passed'] and q['gate']['manifest_sha256']==state['manifest_sha256'] and len({w['uuid'] for w in q['workers'].values()})==3
    state['phase']='local_pilot_audit';save(path,state)
    log=D/'pilot-local-audit.log';err=D/'pilot-local-audit.err.log'
    with log.open('wb') as out,err.open('wb') as errors:
     p=subprocess.Popen([sys.executable,'-u','-B',str(H/'receive_pilot.py')],cwd=H,stdout=out,stderr=errors,env=os.environ.copy())
     receipt=dict(pid=p.pid,active=True,started_at=now(),driver_sha256=sha(H/'receive_pilot.py'),manifest_sha256=state['manifest_sha256'],protocol_sha256=state['protocol_sha256'],jobs=e['jobs'],gate_fingerprint=hashlib.sha256(json.dumps(q['gate'],sort_keys=True).encode()).hexdigest(),stdout=str(log),stderr=str(err));save(H/'checks/pilot-process.json',receipt);code=p.wait();receipt.update(active=False,exit_code=code,ended_at=now());save(H/'checks/pilot-process.json',receipt)
    assert code==0,err.read_text(encoding='utf-8')
    break
   time.sleep(30)
  receipt=read(H/'checks/pilot-receipt.json');budget=read(H/'checks/runtime-estimate.json');assert receipt['passed'] and budget['passed']
  until=time.monotonic()+120
  while True:
   q=query();assert not any(w['phase']=='training_or_summary_failed' for w in q['workers'].values())
   if len(q['workers'])==3 and all(w['phase']=='main' for w in q['workers'].values()):break
   assert time.monotonic()<until,'Approval sent but all-worker main confirmation pending; do not resend';time.sleep(5)
  assert q['local_approval'];q.update(checked_at=now(),passed=True,all_main=True,pilot_receipt_sha256=sha(H/'checks/pilot-receipt.json'),runtime_estimate_sha256=sha(H/'checks/runtime-estimate.json'));save(H/'checks/main-startup.json',q)
  state.update(phase='main_started',pilot_receipt_sha256=q['pilot_receipt_sha256'],reporting_pending=True);print(json.dumps(dict(passed=True,main_started=True,jobs=e['jobs'],workers=q['workers'])),flush=True)
 except BaseException:
  state.update(failed=True,error=traceback.format_exc());save(H/'checks/pilot-watch-failure.json',state);raise
 finally:
  state.update(active=False,ended_at=now());save(path,state)
if __name__=='__main__':main()
