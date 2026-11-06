import os,sys,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;worker=int(sys.argv[1]);job=os.environ['SLURM_JOB_ID'];control=ROOT/'control';control.mkdir(exist_ok=True)
variants=['full124','common124','full90','common90'];myvariants=variants[:2] if worker==0 else variants[2:]
manifest_sha=hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()
def state(phase,**kw):
 p=control/('worker-'+job+'.json');p.write_text(json.dumps(dict(job_id=job,worker=worker,phase=phase,manifest_sha256=manifest_sha,**kw),indent=2))
def call(script,args,variant,log):
 log.parent.mkdir(parents=True,exist_ok=True)
 with log.open('a') as f:
  p=subprocess.run(['/usr/bin/python3.8','-u','-B',str(ROOT/script)]+args,env=dict(os.environ,GDI_VARIANT=variant),stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError('%s %s exit%d; %s'%(variant,script,p.returncode,log))
try:
 state('gpu_preflight')
 for v in variants:call('check_one.py',['--device','cuda','--output',str(ROOT/'checks'/job/v)],v,ROOT/'logs'/('check-'+job+'-'+v+'.log'))
 state('pilot')
 for v in myvariants:
  folder=ROOT/'runs/pilot'/v
  call('run.py',['pilot','--device','cuda','--output',str(folder)],v,ROOT/'logs'/('pilot-'+v+'.log'))
  call('pilot_replay.py',[str(folder)],v,ROOT/'logs'/('pilot-'+v+'.log'))
 state('waiting_pilot_review')
 deadline=time.monotonic()+7200
 while not (control/'approve-main.json').exists():
  if time.monotonic()>deadline:raise TimeoutError('Pilot approval absent after 2 hours')
  time.sleep(10)
 approval=json.loads((control/'approve-main.json').read_text());assert approval['manifest_sha256']==manifest_sha
 tasks=[(v,f) for v in variants for f in range(5)]
 for task,(v,f) in enumerate(tasks):
  if task%2!=worker:continue
  state('training',variant=v,fold=f)
  lock=control/('lock-'+v+'-'+str(f));fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
  try:call('run.py',['train','--device','cuda','--fold',str(f),'--output',str(ROOT/'runs/main'/v)],v,ROOT/'logs'/('train-'+v+'-'+str(f)+'.log'))
  finally:lock.unlink()
 state('assigned_folds_complete')
 # Exactly the worker that sees all outer outputs first obtains the summary lock.
 if all((ROOT/'runs/main'/v/('fold%d'%f)/'complete.json').exists() for v,f in tasks):
  try:fd=os.open(str(control/'summary.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
  except FileExistsError:pass
  else:
   for v in variants:call('run.py',['summarize','--device','cpu','--output',str(ROOT/'runs/main'/v)],v,ROOT/'logs'/('summary-'+v+'.log'))
   (ROOT/'runs/summary_complete.json').write_text(json.dumps(dict(variants=variants,outer_predictions=20,expected_fits=100,manifest_sha256=manifest_sha)))
except BaseException as e:
 state('training_or_summary_failed',error=repr(e));raise
