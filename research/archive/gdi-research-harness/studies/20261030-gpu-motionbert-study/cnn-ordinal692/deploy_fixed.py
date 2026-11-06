"""Preserve failed first pilots; submit verified deterministic implementation once."""
from pathlib import Path
import hashlib,json,subprocess,zipfile,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[3];REMOTE='/abr/RESEARCH_USER/gdi-cnn-ordinal692-v2-20261031'
D=Path('D:/gait-research-data/D_P')/H.relative_to(ROOT);D.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
proof=json.loads((H/'checks/verification.json').read_text());assert proof['passed'] and proof['manifest_sha256']==sha(H/'manifest.json')
assert not (H/'submission-v2.json').exists()
opts=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def ssh(code):return subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True).stdout
code=r'''
from pathlib import Path
import subprocess,json,tarfile,hashlib
root=Path('/abr/RESEARCH_USER/gdi-cnn-ordinal692-20261031')
acct=subprocess.check_output(['sacct','-n','-P','-j','33495,33496','--format=JobIDRaw,State,ExitCode'],text=True)
assert '33495|FAILED|1:0' in acct and '33496|FAILED|1:0' in acct
assert not (root/'runs/main').exists() and not (root/'control/approve-main.json').exists()
p=root/'pilot-failure-review.tar.gz'
with tarfile.open(str(p),'w:gz') as tf:
    for name in ['runs/pilot','control','logs','manifest.json','protocol.json','ordinal.py','worker.py','run.py']:
        tf.add(str(root/name),arcname=name)
print(json.dumps(dict(accounting=acct,archive=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),main_started=False)))
'''
failure=json.loads(ssh(code));dest=D/'pilot-failure-review-v1.tar.gz'
subprocess.run(['scp',*opts,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+failure['archive'],str(dest)],check=True);assert sha(dest)==failure['sha256']
failure['local_archive']=str(dest);save(H/'history/first-attempt/failure_receipt.json',failure)
assert json.loads(ssh("from pathlib import Path\nimport json\nprint(json.dumps(not Path('"+REMOTE+"').exists()))"))
files=json.loads((H/'manifest.json').read_text());source=H.parent/'cnn-record-loss692/bundle/data.npz'
for n,dig in files.items():assert sha(H/n)==dig
archive=D/'gdi-cnn-ordinal692-v2-20261031.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for n in [*files,'manifest.json','README.md','checks/verification.json','checks/baselines.json']:z.write(H/n,'gdi-cnn-ordinal692-v2-20261031/'+n)
    z.write(source,'gdi-cnn-ordinal692-v2-20261031/data.npz')
subprocess.run(['scp',*opts,'-P','10000',str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/gdi-cnn-ordinal692-v2-20261031.zip'],check=True)
code='''from pathlib import Path
import hashlib,json,zipfile,subprocess,os
p=Path('/abr/RESEARCH_USER/gdi-cnn-ordinal692-v2-20261031.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:z.extractall('/abr/RESEARCH_USER')
root=Path(%r);os.chdir(root)
for name,digest in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((root/'data.npz').read_bytes()).hexdigest()==%r
(root/'logs').mkdir();(root/'control').mkdir()
rec=dict(jobs=[],rank_jobs={},phase='held_ready')
for rank in range(2):
    wrap='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 /usr/bin/env SLURM_PROCID='+str(rank)+' /usr/bin/python3.8 -u -B worker.py'
    args=['sbatch','--hold','--parsable','--account=iucp','--partition=p02','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=12G','--time=24:00:00','--exclude=DIS04,DIS05,DIS06','--job-name=gdi-ordinal-v2-'+str(rank),'--output=logs/worker-%%j.log','--wrap',wrap]
    q=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True);job=q.stdout.strip().split(';')[0];assert job.isdigit()
    rec['jobs'].append(job);rec['rank_jobs'][str(rank)]=job;(root/'control/submission.json').write_text(json.dumps(rec,indent=2))
print(json.dumps(rec))
'''%(sha(archive),REMOTE,sha(source))
result=json.loads(ssh(code));result.update(remote=REMOTE,archive=str(archive),archive_sha256=sha(archive),manifest_sha256=sha(H/'manifest.json'),submitted_at=datetime.datetime.now().isoformat(),gpu_per_job=1,total_gpu=2,previous_jobs=['33494','33495','33496'],failure_receipt_sha256=sha(H/'history/first-attempt/failure_receipt.json'))
save(H/'submission-v2.json',result);save(H/'submission.json',result);print(json.dumps(result),flush=True)
