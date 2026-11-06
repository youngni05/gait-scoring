"""Safer alternative: upload code only; all synthetic tensors generated remotely."""
from pathlib import Path
import sys,subprocess,json,zipfile,re,hashlib,datetime
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def ssh(cmd):
    p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+['RESEARCH_USER@PRIVATE_HOST',cmd],capture_output=True,text=True,encoding='utf8',timeout=120);assert p.returncode==0,p.stderr;return p.stdout.strip()
R='/abr/RESEARCH_USER/frailpol-temporal-env-20261104'
assert not(H/'environment-submission.json').exists(),'Inspect existing submission, do not duplicate'
archive=B/'synthetic-code-only.zip'
script='''#!/bin/bash
#SBATCH --account=iucp
#SBATCH --partition=p02
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd /abr/RESEARCH_USER/frailpol-temporal-env-20261104
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/usr/bin/python3.8 -u -B env.py
'''
(H/'bundle/submit-env.slurm').write_text(script,encoding='utf8',newline='\n')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name,target in [('train.py','train.py'),('env.py','env.py'),('submit-env.slurm','submit.slurm')]:z.write(H/'bundle'/name,target)
with zipfile.ZipFile(archive) as z:assert set(z.namelist())=={'train.py','env.py','submit.slurm'} and all(z.getinfo(n).file_size<50000 for n in z.namelist())
digest=sha(archive);code="from pathlib import Path;p=Path('"+R+"');p.mkdir();(p/'logs').mkdir();print('created')";ssh('/usr/bin/python3.8 -c "'+code+'"')
p=subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[str(archive),'RESEARCH_USER@PRIVATE_HOST:'+R+'/code-only.zip'],capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr
code="import hashlib,zipfile;from pathlib import Path;p=Path('"+R+"');f=p/'code-only.zip';assert hashlib.sha256(f.read_bytes()).hexdigest()=='"+digest+"';zipfile.ZipFile(f).extractall(p)";ssh('/usr/bin/python3.8 -c "'+code+'"');reply=ssh('cd '+R+' && sbatch --parsable --nodelist=DIS02 --time=00:03:00 --job-name=frailpol-synthetic-env submit.slurm');job=reply.split(';')[0].strip();assert re.fullmatch(r'\d+',job)
record=dict(job=job,remote=R,owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',uploaded_members=['train.py','env.py','submit.slurm'],data_uploaded=False,clinical_data_uploaded=False,archive_sha256=digest,kind='synthetic_environment',requested_seconds=180,estimated_total_seconds=120,allocation_multiplier=1.5)
save(H/'environment-submission.json',record);save(H/'watch/registry.json',dict(thread_id=record['owner_thread'],automation_id='frailpol-gpu',monitor_estimate=dict(status='short_synthetic_pilot_checked_in_active_turn',clinical_training_requires_explicit_transfer_approval=True),experiments=[dict(id='frailpol-synthetic-env-20261104',kind='synthetic_environment',jobs=[job],remote=R,local=str(H),received=str(B/'received/synthetic-environment'),run_root='checks',summary_path='checks/summary.json',marker_path='checks/summary.json',curve_glob='__none__',outer_complete_glob='__none__',review_state='pending',reported_digest=None)]));print(json.dumps(record))
