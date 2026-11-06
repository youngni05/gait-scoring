"""Verify both submitted-job pilots, then release ONLY their existing main-training gate.
No sbatch, scancel, new model conditions or other-study writes.
"""
import os,sys,json,hashlib,subprocess,tarfile,datetime,math
from pathlib import Path
H=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/cnn-stable-scale-20261103')
REMOTE='/abr/RESEARCH_USER/cnn-stable-scale-20261103';JOBS=['34737','34738'];OWNER='01a19c6f-49f7-7300-97b0-a303b71ffe91'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST,'/usr/bin/python3.8 -']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def remote(code):
 p=subprocess.run(SSH,input=code,text=True,encoding='utf-8',capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
 return json.loads(p.stdout)
r=read(H/'watch/registry.json');assert r['thread_id']==OWNER and r['automation_id']=='gdi-cnn';e=next(x for x in r['experiments'] if x['id']=='cnn-stable-scale-20261103');assert e['jobs']==JOBS and e['remote']==REMOTE
if (H/'main-release.json').exists() and read(H/'main-release.json').get('main_gate_released'):
 print(json.dumps(dict(status='main_gate_already_released')));sys.exit(0)
probe=remote("""
import pathlib,json,subprocess
root=pathlib.Path(%r);jobs=%r
workers={j:json.loads((root/'control'/('worker-'+j+'.json')).read_text()) if (root/'control'/('worker-'+j+'.json')).exists() else None for j in jobs}
queue=subprocess.check_output(['squeue','-h','-j',','.join(jobs),'-o','%%i|%%T|%%L'],text=True)
print(json.dumps(dict(workers=workers,queue=queue)))
"""%(REMOTE,JOBS))
if not all(probe['workers'][j] and probe['workers'][j]['phase']=='waiting_pilot_review' for j in JOBS):
 print(json.dumps(dict(status='pilots_not_both_ready',**probe)));sys.exit(0)
manifest_sha=sha(H/'bundle/manifest.json')
for w in probe['workers'].values():assert w['manifest_sha256']==manifest_sha
packet=remote("""
import pathlib,json,tarfile,hashlib
root=pathlib.Path(%r);archive=root/'pilot-review.tar.gz'
with tarfile.open(str(archive),'w:gz') as tf:
 for base in ['checks','runs/pilot','control']:
  for p in sorted((root/base).rglob('*')):
   if p.is_file():tf.add(str(p),arcname=str(p.relative_to(root)))
 for p in sorted(root.glob('*')):
  if p.is_file() and p.suffix in ['.py','.json','.slurm']:tf.add(str(p),arcname=p.name)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
"""%REMOTE)
received=B/'received/pilot';received.mkdir(parents=True,exist_ok=True);archive=received/'pilot-review.tar.gz'
subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[HOST+':'+packet['path'],str(archive)],check=True,timeout=120)
assert sha(archive)==packet['sha256']
with tarfile.open(archive) as tf:tf.extractall(received,filter='data')
assert sha(received/'manifest.json')==manifest_sha
checks=[]
for j in JOBS:
 for v in ['baseline','stable_scale']:
  q=received/'checks'/j/v;check=read(q/'check.json');smoke=read(q/'smoke/smoke.json');assert check['original_full124_forward_max_error']==0 and check['all20_person_splits_checked'];assert smoke['passed'] and smoke['resume_max_error']==0;checks.append(check)
pilots={};replays=[]
for v in ['baseline','stable_scale']:
 folder=received/'runs/pilot'/v;pilot=read(folder/'pilot_summary.json');assert pilot['epochs']==3 and not pilot['outer_evaluation_used'];assert read(folder/'replay.json')['passed']
 env=dict(os.environ,GDI_VARIANT=v,GDI_DATA_DIR=str(B/'data'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1')
 with (folder/'local-replay.log').open('w') as log:p=subprocess.run([sys.executable,'-B',str(H/'bundle/pilot_replay.py'),str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT)
 assert p.returncode==0,(folder/'local-replay.log').read_text();replays.append(read(folder/'local-replay.json'));pilots[v]=pilot
# Same-sized fold allocation: a conservative per-worker upper estimate, plus25% and1h.
model_seconds={v:p['rough_max_training_seconds'] for v,p in pilots.items()};worker_seconds={JOBS[0]:sum(model_seconds.values())*.61,JOBS[1]:sum(model_seconds.values())*.41}
def seconds(s):
 days=0
 if '-' in s:day,s=s.split('-');days=int(day)
 parts=list(map(int,s.split(':')))
 if len(parts)==2:parts=[0]+parts
 return days*86400+parts[0]*3600+parts[1]*60+parts[2]
rows={line.split('|')[0]:line.split('|')[1:] for line in probe['queue'].splitlines() if line.strip()}
for job in JOBS:assert rows[job][0]=='RUNNING' and seconds(rows[job][1])>worker_seconds[job]*1.25+3600,(job,rows[job],worker_seconds[job])
verification=dict(passed=True,main_gate_released=False,manifest_sha256=manifest_sha,pilot_archive_sha256=packet['sha256'],checks=4,pilots=pilots,local_replays=replays,model_total_estimated_seconds=model_seconds,worker_estimated_upper_seconds=worker_seconds,reviewed_at=datetime.datetime.now().astimezone().isoformat())
save(H/'gpu-pilot-verification.json',verification)
approval=dict(manifest_sha256=manifest_sha,pilot_archive_sha256=packet['sha256'],verification_sha256=sha(H/'gpu-pilot-verification.json'),authorized_jobs=JOBS,reason='same user-authorized two-arm CNN training; both real pilots independently replayed')
response=remote("""
import json,pathlib,hashlib,subprocess
root=pathlib.Path(%r);approval=json.loads(%r)
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==approval['manifest_sha256']
for job in approval['authorized_jobs']:
 assert subprocess.check_output(['squeue','-h','-j',job,'-o','%%T'],text=True).strip()=='RUNNING'
 assert json.loads((root/'control'/('worker-'+job+'.json')).read_text())['phase']=='waiting_pilot_review'
p=root/'control/approve-main.json';assert not p.exists();tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(approval,indent=2));tmp.replace(p)
print(json.dumps(dict(main_gate_released=True,approval_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
"""%(REMOTE,json.dumps(approval)))
save(H/'main-release.json',dict(**response,verification_sha256=sha(H/'gpu-pilot-verification.json'),manifest_sha256=manifest_sha))
r=read(H/'watch/registry.json');assert r['thread_id']==OWNER
r['monitor_estimate']=dict(source='both current GPU3epoch pilots',source_path=str(H/'gpu-pilot-verification.json'),model_total_estimated_seconds=model_seconds,half_shortest_total_minutes=min(model_seconds.values())/120,interval_minutes=max(10,10*round(min(model_seconds.values())/1200)),new_pilots_pending=False,not_remaining_time=True)
save(H/'watch/registry.json',r)
print(json.dumps(dict(status='both_pilots_verified_existing_main_gate_released',monitor=r['monitor_estimate'],worker_estimated_upper_seconds=worker_seconds)))
