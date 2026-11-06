"""Finite package/submit/status/collect operations; no polling or background helper."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,tarfile,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];B=HERE/'bundle'
BIG=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105');REMOTE='/abr/RESEARCH_USER/gdi-masked-pretrain-v3-20261105'
REG=ROOT/'studies/20261104-ljmu100-mixed/watch/registry.json';OWNER='01a1a61c-35ee-71e1-bb0b-78ad9d6a7b22'
auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
ssh=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe');scp=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe');host='RESEARCH_USER@PRIVATE_HOST'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def remote(program,*args):
    proc=subprocess.run([ssh,'-p','10000',*auth,host,'/usr/bin/python3.8','-',*map(str,args)],input=program,text=True,capture_output=True)
    if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    return json.loads(proc.stdout)
def register(receipt):
    r=read(REG);assert r['thread_id']==OWNER
    entry=next((e for e in r['experiments'] if e['id']=='gdi-masked-pretrain-v3-20261105'),None)
    if entry is None:
        entry=dict(id='gdi-masked-pretrain-v3-20261105',kind='masked_pretrain_then_gdi',local=str(HERE),remote=REMOTE,large_local=str(BIG),jobs=[],review_state='pending',reported_digest=None,expected_main_fits=60,expected_pretraining_fits=30,monitoring_mode='Finite foreground snapshots only; no job-completion automation',preserve_external_allocations=True);r['experiments'].append(entry)
    entry['jobs']=list(dict.fromkeys(entry['jobs']+list(receipt['jobs'].values())))
    entry[receipt['stage']+'_jobs']=receipt['jobs'];save(REG,r)

ap=argparse.ArgumentParser();ap.add_argument('action',choices=['package','pilot','main','status','collect']);ap.add_argument('--stage',choices=['pilot','main'],default='pilot');a=ap.parse_args()
if a.action=='package':
    assert read(HERE/'preflight.json')['passed'];assert not (B/'manifest.json').exists(),'Sealed bundle must not change'
    # Retain upstream notices distributed with the original model source.
    for family in ['CNN','STT']:
        src=ROOT/'studies/20261105-matched-normal100/bundle'/family/'sources'
        for p in src.rglob('*'):
            if p.is_file() and ('LICENSE' in p.name.upper() or p.name=='provenance.json'):
                dest=B/'licenses'/family/p.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    files=sorted(p for p in B.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest={p.relative_to(B).as_posix():sha(p) for p in files};save(B/'manifest.json',manifest)
    archive=BIG/'gdi-masked-pretrain-v3-20261105.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,p.relative_to(B).as_posix())
        z.write(BIG/'data.npz','data.npz')
    receipt=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(B/'manifest.json'),data_sha256=sha(BIG/'data.npz'),contains='Existing approved deidentified feature data and coordinate observation masks; model/code; no RGB video.');save(HERE/'package.json',receipt);print(json.dumps(receipt,indent=2))
elif a.action=='pilot':
    pkg=read(HERE/'package.json');assert sha(Path(pkg['path']))==pkg['sha256']
    assert not (HERE/'submission-pilot.json').exists(),'Reconcile existing submission instead of duplicating'
    subprocess.run([scp,'-P','10000',*auth,pkg['path'],host+':/abr/RESEARCH_USER/gdi-masked-pretrain-v3-20261105.zip'],check=True)
    program="""import os,sys,json,hashlib,zipfile,subprocess,datetime
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-masked-pretrain-v3-20261105');archive=root.with_suffix('.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==sys.argv[1]
assert not root.exists(),'Existing remote run requires reconciliation'
root.mkdir()
with zipfile.ZipFile(archive) as z:z.extractall(root)
(root/'logs').mkdir();os.chdir(root)
receipt={'stage':'pilot','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
for family in ['CNN','STT']:
 job=subprocess.check_output(['sbatch','--parsable','--time=00:30:00','--job-name=mask-pilot-'+family,'submit.slurm','pilot',family]).decode().strip().split(';')[0]
 assert job.isdigit();receipt['jobs'][family]=job;(root/'submission-pilot.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
"""
    receipt=remote(program,pkg['sha256']);save(HERE/'submission-pilot.json',receipt);register(receipt);print(json.dumps(receipt,indent=2))
elif a.action=='main':
    approved=HERE/'main-approved.json';assert approved.exists();assert not (HERE/'submission-main.json').exists()
    subprocess.run([scp,'-P','10000',*auth,str(approved),host+':'+REMOTE+'/main-approved.json'],check=True)
    program="""import os,sys,json,hashlib,subprocess,datetime
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-masked-pretrain-v3-20261105');os.chdir(root)
assert hashlib.sha256((root/'main-approved.json').read_bytes()).hexdigest()==sys.argv[1]
p=json.loads((root/'main-approved.json').read_text());assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==p['manifest_sha256']
assert not (root/'submission-main.json').exists();fd=os.open('submit-main.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
receipt={'stage':'main','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{}}
for family in ['CNN','STT']:
 for condition in ['A','B_adult','B_child','C','D_adult','D_child']:
  assert (root/'runs/pilot'/family/condition/'fold0/complete.json').exists()
 job=subprocess.check_output(['sbatch','--parsable','--time='+p['workers'][family]['time_limit'],'--job-name=mask-main-'+family,'submit.slurm','train',family]).decode().strip().split(';')[0]
 assert job.isdigit();receipt['jobs'][family]=job;(root/'submission-main.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
"""
    receipt=remote(program,sha(approved));save(HERE/'submission-main.json',receipt);register(receipt);print(json.dumps(receipt,indent=2))
else:
    jobs=[]
    for name in ['pilot','main']:
        path=HERE/('submission-'+name+'.json')
        if path.exists():jobs.extend(read(path)['jobs'].values())
    program="""import sys,json,subprocess,datetime,tarfile,hashlib
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-masked-pretrain-v3-20261105');stage=sys.argv[2]
status={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sacct':subprocess.check_output(['sacct','-j',sys.argv[1],'--format=JobID,State,ExitCode,Elapsed,NodeList','--parsable2','--noheader']).decode(),'fits':[]}
files=[]
for folder in sorted((root/'runs'/stage).glob('*/*/fold*')):
 done=(folder/'complete.json').exists();curve=folder/'curve.json';pc=folder/'pretrain-curve.json'
 status['fits'].append({'family':folder.parent.parent.name,'condition':folder.parent.name,'fold':folder.name,'complete':done,'epochs':len(json.loads(curve.read_text())) if curve.exists() else 0,'pretrain_steps':json.loads(pc.read_text())[-1]['step'] if pc.exists() else 0})
 if done:files.extend(p for p in folder.rglob('*') if p.is_file())
if sys.argv[3]=='collect':
 files+=list((root/'logs').glob('*.log'));files=sorted(set(files));archive=root/('collected-'+stage+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tf:
  for p in files:tf.add(p,arcname=p.relative_to(root).as_posix())
 status['files']={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};status['archive']=str(archive);status['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps(status))
"""
    receipt=remote(program,','.join(jobs),a.stage,a.action);save(HERE/'watch'/('latest-'+a.stage+'.json'),receipt)
    if a.action=='collect':
        dest=BIG/Path(receipt['archive']).name;subprocess.run([scp,'-P','10000',*auth,host+':'+receipt['archive'],str(dest)],check=True)
        assert sha(dest)==receipt['archive_sha256'];target=BIG/'received';target.mkdir(exist_ok=True)
        with tarfile.open(dest) as tf:
            for m in tf.getmembers():assert m.isfile() and (target/m.name).resolve().is_relative_to(target.resolve())
            tf.extractall(target,filter='data')
        for name,digest in receipt['files'].items():assert sha(target/name)==digest
        save(HERE/'watch'/('receipt-'+a.stage+'.json'),receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
