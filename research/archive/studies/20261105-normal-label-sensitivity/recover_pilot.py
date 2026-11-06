"""Recover a pre-submission line-ending rejection; no fit was launched."""
import json,os,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
program='''import os,json,hashlib,subprocess,datetime
from pathlib import Path
root=Path('/abr/RESEARCH_USER/normal-label-sensitivity-20261105');os.chdir(root)
assert not (root/'submission-pilot.json').exists()
for name,digest in json.loads((root/'manifest.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert not list((root/'runs').glob('*/*/*/fold*'))
active=subprocess.check_output(['squeue','-u','RESEARCH_USER','-h','-o','%j']).decode().splitlines();assert not any(v.startswith('sensitivity-pilot-') for v in active)
original=(root/'submit.slurm').read_bytes();assert b'\\r\\n' in original
normalized=original.replace(b'\\r\\n',b'\\n');(root/'submit-unix.slurm').write_bytes(normalized)
assert normalized.replace(b'\\n',b'\\r\\n')==original
fd=os.open('submit-pilot.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
receipt={'stage':'pilot','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{},'original_script_sha256':hashlib.sha256(original).hexdigest(),'executed_script_sha256':hashlib.sha256(normalized).hexdigest(),'fix':'Only CRLF to LF, original sealed file unchanged'}
for family in ['CNN','STT']:
 job=subprocess.check_output(['sbatch','--parsable','--time=00:30:00','--job-name=sensitivity-pilot-'+family,'submit-unix.slurm','pilot',family]).decode().strip().split(';')[0]
 assert job.isdigit();receipt['jobs'][family]=job;(root/'submission-pilot.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
r=subprocess.run([str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe'),'-p','10000',*auth,'RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8','-'],input=program,text=True,capture_output=True,check=True)
receipt=json.loads(r.stdout);(HERE/'submission-pilot.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
reg=HERE.parents[1]/'studies/20261104-ljmu100-mixed/watch/registry.json';v=json.loads(reg.read_text(encoding='utf-8-sig'));assert v['thread_id']=='01a1a61c-35ee-71e1-bb0b-78ad9d6a7b22'
assert not any(e['id']=='normal-label-sensitivity-20261105' for e in v['experiments'])
v['experiments'].append(dict(id='normal-label-sensitivity-20261105',kind='assumed_label_loss_sensitivity',local=str(HERE),remote='/abr/RESEARCH_USER/normal-label-sensitivity-20261105',large_local='D:/gait-research-data/D_P/normal-label-sensitivity-20261105',jobs=list(receipt['jobs'].values()),pilot_jobs=receipt['jobs'],review_state='pending',reported_digest=None,expected_main_fits=40,expected_pretraining_fits=0,monitoring_mode='Finite foreground snapshots only; no completion automation',preserve_external_allocations=True))
reg.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(receipt,indent=2))
