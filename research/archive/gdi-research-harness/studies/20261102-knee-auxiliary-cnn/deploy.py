"""Upload sealed code, verify server inputs, submit two held one-GPU jobs."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P')/HERE.relative_to(ROOT)
REMOTE='/abr/RESEARCH_USER/gdi-knee-auxiliary-cnn-20261102'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
      '-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
SSH='C:/Windows/System32/OpenSSH/ssh.exe'
SCP='C:/Windows/System32/OpenSSH/scp.exe'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ssh(code):
    p=subprocess.run([SSH,*AUTH,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],
                     input=code,text=True,capture_output=True,timeout=120)
    if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
    return p.stdout


def main():
    assert not (HERE/'submission.json').exists(),'Already submitted: inspect existing receipt first'
    proof=read(BIG/'checks/cpu-v1/verification.json')
    assert proof['passed'] and proof['synthetic_nested_fits']==90
    assert proof['manifest_sha256']==sha(HERE/'manifest.json')
    for name,digest in read(HERE/'manifest.json').items():assert sha(HERE/name)==digest
    (HERE/'checks').mkdir(exist_ok=True)
    save(HERE/'checks/cpu-verification.json',proof)
    queue=ssh("import subprocess\nprint(subprocess.check_output(['squeue','-u','RESEARCH_USER','-o','%.18i %.24j %.2t %.10M %.10L %R'],text=True))\nfrom pathlib import Path\nassert not Path("+repr(REMOTE)+").exists(),'Existing experiment: do not overwrite'\n")
    print(queue,flush=True)
    archive=BIG/'gdi-knee-auxiliary-cnn-20261102.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in [*read(HERE/'manifest.json'),'manifest.json','README.md','checks/cpu-verification.json']:
            z.write(HERE/name,Path(REMOTE).name+'/'+name)
    subprocess.run([SCP,*AUTH,'-P','10000',str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/'+archive.name],check=True)
    code='''from pathlib import Path
import hashlib,json,zipfile,shutil,subprocess,os
root=Path(%r);assert not root.exists()
archive=Path(str(root)+'.zip');assert hashlib.sha256(archive.read_bytes()).hexdigest()==%r
with zipfile.ZipFile(archive) as z:z.extractall('/abr/RESEARCH_USER')
expected=json.loads((root/'protocol.json').read_text())['data_sha256']
source=Path('/abr/RESEARCH_USER/gdi-cnn-multiwindow-20261030/data.npz')
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
shutil.copyfile(str(source),str(root/'data.npz'))
assert hashlib.sha256((root/'data.npz').read_bytes()).hexdigest()==expected
for name,digest in json.loads((root/'manifest.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
os.chdir(root);(root/'logs').mkdir();(root/'control').mkdir()
record=dict(jobs=[],rank_jobs={},phase='held_ready')
for rank in range(2):
    wrap='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 /usr/bin/env KNEE_RANK='+str(rank)+' /usr/bin/python3.8 -u -B worker.py'
    args=['sbatch','--hold','--parsable','--account=iucp','--partition=p02','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=12G','--time=24:00:00','--exclude=DIS04,DIS05,DIS06','--job-name=gdi-knee-'+str(rank),'--output=logs/worker-%%j.log','--wrap',wrap]
    p=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    job=p.stdout.strip().split(';')[0];assert job.isdigit()
    record['jobs'].append(job);record['rank_jobs'][str(rank)]=job
    (root/'control/submission.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
'''%(REMOTE,sha(archive))
    result=json.loads(ssh(code))
    result.update(remote=REMOTE,archive=str(archive),archive_sha256=sha(archive),
                  manifest_sha256=sha(HERE/'manifest.json'),submitted_at=datetime.datetime.now().isoformat(),
                  gpu_per_job=1,total_gpus=2,expected_fits=90,expected_outer_predictions=10)
    save(HERE/'submission.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
