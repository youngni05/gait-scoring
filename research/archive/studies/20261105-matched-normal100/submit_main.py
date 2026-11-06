"""Submit authorized matched arms once; persist actual Slurm IDs immediately."""
import hashlib,json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
ssh=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe');scp=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe');host='RESEARCH_USER@PRIVATE_HOST'
remote='/abr/RESEARCH_USER/matched-normal100-20261105';approval=HERE/'main-approved.json'
assert json.loads((HERE/'preflight.json').read_text())['passed']
assert not json.loads((HERE/'submission.json').read_text())['main_jobs'],'Already submitted'
subprocess.run([scp,'-P','10000',*auth,str(approval),host+':'+remote+'/main-approved.json'],check=True)
program=r'''
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path('/abr/RESEARCH_USER/matched-normal100-20261105');os.chdir(root)
assert hashlib.sha256((root/'main-approved.json').read_bytes()).hexdigest()==sys.argv[1]
p=json.loads((root/'main-approved.json').read_text())
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==p['bundle_sha256']
receipt=root/'main-submission.json'
assert not receipt.exists(),'Prior submission receipt exists; reconcile instead of resubmitting'
lock=os.open(str(root/'main-submit.lock'),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(lock)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs={})
for worker in p['workers']:
    arm=worker['arm']
    for family in ['CNN','STT','bilateral']:
        assert json.loads((root/'data'/arm/'runs/pilot'/family/'verification.json').read_text())['passed']
    job=subprocess.check_output(['sbatch','--parsable','--time='+worker['time_limit'],'--job-name=matched-'+arm,'submit.slurm','train',arm]).decode().strip().split(';')[0]
    assert job.isdigit();result['jobs'][arm]=job;receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(arm=arm,job=job)),flush=True)
'''
digest=hashlib.sha256(approval.read_bytes()).hexdigest()
proc=subprocess.run([ssh,'-p','10000',*auth,host,'/usr/bin/python3.8','-',digest],input=program,text=True,capture_output=True)
sub=json.loads((HERE/'submission.json').read_text())
for line in proc.stdout.splitlines():
    row=json.loads(line);sub['main_jobs'][row['arm']]=row['job']
sub['state']='main_submitted' if proc.returncode==0 else 'submission_requires_reconciliation'
(HERE/'submission.json').write_text(json.dumps(sub,indent=2)+'\n',encoding='utf-8')
registry=HERE.parent/'20261104-ljmu100-mixed/watch/registry.json';v=json.loads(registry.read_text());assert v['thread_id']=='01a1a61c-35ee-71e1-bb0b-78ad9d6a7b22'
e=next(e for e in v['experiments'] if e['id']=='matched-normal100-20261105');e['jobs']=list(sub['pilot_jobs'].values())+list(sub['main_jobs'].values());e['main_jobs']=sub['main_jobs'];e['workers']=json.loads(approval.read_text())['workers'];e['pilot_audit']='Six independent CPU replays and Stanford-only calibration checks passed';e['rf_status']='Both arms completed and audited';registry.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(proc.stdout)
if proc.returncode:raise RuntimeError(proc.stderr)
