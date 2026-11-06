"""Upload sealed code and submit three HELD jobs; registry must precede release."""
from pathlib import Path
import datetime,hashlib,json,subprocess,zipfile
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
D=Path('D:/gait-research-data/D_P')/H.relative_to(ROOT)
REMOTE='/abr/RESEARCH_USER/gdi-cnn-mae-point692-20261104'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ssh(code):
    return subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe',*AUTH,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True).stdout
def main():
    assert not (H/'submission.json').exists(),'Already submitted; inspect before retry'
    proof=json.loads((D/'checks/cpu-v1/verification.json').read_text())
    assert proof['passed'] and proof['synthetic_nested_fits']==50 and proof['manifest_sha256']==sha(H/'manifest.json')
    (H/'checks').mkdir(exist_ok=True);(H/'checks/cpu-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    files=json.loads((H/'manifest.json').read_text())
    for n,digest in files.items():assert sha(H/n)==digest
    archive=D/'gdi-cnn-mae-point692-20261104.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for n in [*files,'manifest.json','README.md','checks/cpu-verification.json']:z.write(H/n,'gdi-cnn-mae-point692-20261104/'+n)
    # Probe first: never overwrite an existing remote experiment, even without a local receipt.
    ssh("from pathlib import Path\nassert not Path(%r).exists(), 'Existing remote experiment'\n"%REMOTE)
    subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe',*AUTH,'-P','10000',str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/gdi-cnn-mae-point692-20261104.zip'],check=True)
    code='''from pathlib import Path
import hashlib,json,zipfile,subprocess,shutil,os
root=Path(%r);assert not root.exists()
p=Path(str(root)+'.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(p) as z:z.extractall('/abr/RESEARCH_USER')
src=Path('/abr/RESEARCH_USER/gdi-cnn-ordinal692-v2-20261031/data.npz')
expected=json.loads((root/'training.json').read_text())['data_sha256']
assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
shutil.copyfile(str(src),str(root/'data.npz'))
assert hashlib.sha256((root/'data.npz').read_bytes()).hexdigest()==expected
for n,dig in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==dig
os.chdir(root);(root/'logs').mkdir();(root/'control').mkdir()
pre=subprocess.run(['/usr/bin/python3.8','-B','source_contract.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,check=True)
(root/'checks/source-contract-pre-submit.json').write_text(pre.stdout)
record=dict(jobs=[],rank_jobs={},phase='held_ready')
for rank in range(3):
    wrap='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 /usr/bin/env BMC_RANK='+str(rank)+' /usr/bin/python3.8 -u -B worker.py'
    args=['sbatch','--hold','--parsable','--account=iucp','--partition=p02','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=12G','--time=24:00:00','--exclude=DIS04,DIS05,DIS06','--job-name=gdi-mae-point-'+str(rank),'--output=logs/worker-%%j.log','--wrap',wrap]
    q=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    job=q.stdout.strip().split(';')[0];assert job.isdigit()
    record['jobs'].append(job);record['rank_jobs'][str(rank)]=job
    (root/'control/submission.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''%(REMOTE,sha(archive))
    result=json.loads(ssh(code));result.update(remote=REMOTE,archive=str(archive),archive_sha256=sha(archive),manifest_sha256=sha(H/'manifest.json'),submitted_at=datetime.datetime.now().isoformat(),gpu_per_job=1,total_gpus=3)
    (H/'submission.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
