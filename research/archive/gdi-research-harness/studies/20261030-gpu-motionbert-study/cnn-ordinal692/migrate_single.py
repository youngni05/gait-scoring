"""Replace never-started 2GPU request by two held 1GPU jobs, then explicit release."""
from pathlib import Path
import json,subprocess,sys,datetime
H=Path(__file__).resolve().parent;stage=sys.argv[1];assert stage in ('prepare','release')
opts=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
code=r'''
import json,pathlib,subprocess,os
root=pathlib.Path('/abr/RESEARCH_USER/gdi-cnn-ordinal692-20261031');os.chdir(root)
def cmd(args):return subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
path=root/'control/migration-single.json';path.parent.mkdir(exist_ok=True)
if STAGE=='prepare':
    assert not path.exists(),'Existing migration; inspect rather than resubmit'
    state=cmd(['squeue','-h','-j','33494','-o','%T|%R']).strip()
    assert state.split('|')[0]=='PENDING' and state.split('|')[1].strip('()')=='AssocMaxGRESPerJob',state
    cmd(['scontrol','hold','33494'])
    assert cmd(['squeue','-h','-j','33494','-o','%T']).strip()=='PENDING'
    assert not (root/'runs').exists() and not list((root/'control').glob('rank*.lock'))
    rec=dict(previous_job='33494',reason='AssocMaxGRESPerJob',jobs=[],rank_jobs={},phase='creating_held')
    path.write_text(json.dumps(rec,indent=2))
    for rank in range(2):
        wrap='export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1; srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 /usr/bin/env SLURM_PROCID=%d /usr/bin/python3.8 -u -B worker.py'%rank
        args=['sbatch','--hold','--parsable','--account=iucp','--partition=p02','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:1','--mem=12G','--time=24:00:00','--exclude=DIS04,DIS05,DIS06','--job-name=gdi-ordinal-'+str(rank),'--output=logs/worker-%j.log','--wrap',wrap]
        job=cmd(args).strip().split(';')[0];assert job.isdigit();rec['jobs'].append(job);rec['rank_jobs'][str(rank)]=job;path.write_text(json.dumps(rec,indent=2))
    cmd(['scancel','33494'])
    state=cmd(['sacct','-n','-P','-j','33494','--format=JobIDRaw,State,ExitCode'])
    assert '33494|CANCELLED' in state,state
    rec.update(phase='held_ready',cancelled_accounting=state);path.write_text(json.dumps(rec,indent=2))
else:
    rec=json.loads(path.read_text());assert rec['phase']=='held_ready'
    for job in rec['jobs']:
        state=cmd(['squeue','-h','-j',job,'-o','%T|%R']).strip();assert state.split('|')[0]=='PENDING' and state.split('|')[1].strip('()').startswith('JobHeld'),state
    for job in rec['jobs']:cmd(['scontrol','release',job])
    rec['phase']='released';path.write_text(json.dumps(rec,indent=2))
print(json.dumps(rec))
'''.replace('STAGE',repr(stage))
p=subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True)
if p.returncode:
    (H/('migration-'+stage+'-failure.log')).write_text(p.stdout+p.stderr);raise RuntimeError(p.stderr)
v=json.loads(p.stdout);(H/'migration-single.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
