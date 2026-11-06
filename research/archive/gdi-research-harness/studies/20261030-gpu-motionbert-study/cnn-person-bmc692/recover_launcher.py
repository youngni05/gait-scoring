"""One-time recovery of pre-Python launcher failure; preserve all failed jobs."""
from pathlib import Path
import json,datetime
from deploy import ssh,H,REMOTE
old=json.loads((H/'submission.json').read_text());assert old['jobs']==['33507','33508','33509']
code='''from pathlib import Path
import json,subprocess,hashlib,os
h=Path(%r);os.chdir(h)
acct=subprocess.check_output(['sacct','-n','-P','-j','33507,33508,33509','--format=JobIDRaw,State,ExitCode'],text=True)
for j in ['33507','33508','33509']:assert j+'|FAILED|13:0' in acct
assert not list((h/'control').glob('rank*.lock')) and not (h/'runs').exists()
failed=dict(accounting=acct,logs={})
for job in ['33507','33508','33509']:
 p=h/'logs'/('worker-'+job+'.log');failed['logs'][job]=dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(h/'control/submission-first-attempt.json').write_text((h/'control/submission.json').read_text())
r=dict(jobs=[],rank_jobs={},phase='held_ready',launcher='/usr/bin/env')
for rank in range(3):
 wrap='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 /usr/bin/env BMC_RANK='+str(rank)+' /usr/bin/python3.8 -u -B worker.py'
 args=['sbatch','--hold','--parsable','--account=iucp','--partition=p02','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=12G','--time=24:00:00','--exclude=DIS04,DIS05,DIS06','--job-name=gdi-person-bmc-'+str(rank),'--output=logs/worker-%%j.log','--wrap',wrap]
 p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True);job=p.stdout.strip().split(';')[0];assert job.isdigit()
 r['jobs'].append(job);r['rank_jobs'][str(rank)]=job;(h/'control/submission.json').write_text(json.dumps(r,indent=2))
print(json.dumps(dict(submission=r,failure=failed)))
'''%REMOTE
v=json.loads(ssh(code));history=H/'history/launcher-failure';history.mkdir(parents=True,exist_ok=False)
(history/'submission.json').write_text(json.dumps(old,indent=2)+'\n')
(history/'failure.json').write_text(json.dumps(v['failure'],indent=2)+'\n')
for job,log in v['failure']['logs'].items():
 p=history/(job+'.log');p.write_bytes(log['text'].encode('utf-8'))
 import hashlib
 assert hashlib.sha256(p.read_bytes()).hexdigest()==log['sha256']
new=dict(old,**v['submission']);new['previous_jobs']=old['jobs'];new['submitted_at']=datetime.datetime.now().isoformat()
(H/'submission.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(new))
