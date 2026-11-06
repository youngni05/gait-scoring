"""One finite snapshot/collection; never starts a watcher or waits for completion."""
import argparse,datetime,hashlib,json,os,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['status','pilot','complete'],default='status');a=ap.parse_args()
auth=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
ssh=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe');scp=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe');host='RESEARCH_USER@PRIVATE_HOST'
sub=json.loads((HERE/'submission.json').read_text());jobs=list(sub['pilot_jobs'].values())+list(sub.get('main_jobs',{}).values())
program=r'''
import datetime,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
root=Path('/abr/RESEARCH_USER/matched-normal100-20261105');scope=sys.argv[1];jobs=sys.argv[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
status=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sacct=subprocess.check_output(['sacct','-j',jobs,'--format=JobID,State,ExitCode,Elapsed,NodeList','--parsable2','--noheader']).decode(),arms={})
files=[]
for arm in ['adult','child']:
    run=root/'data'/arm/'runs';done=[];fits=[]
    for family in ['CNN','STT','bilateral']:
        for f in range(5):
            folder=run/'mixed'/family/('fold%d'%f)
            if (folder/'complete.json').exists():
                done.append(dict(family=family,fold=f))
                if scope=='complete':files.extend(p for p in folder.rglob('*') if p.is_file())
            for path in folder.glob('*/curve.json'):
                try:
                    h=json.loads(path.read_text());i=json.loads((path.parent/'identity.json').read_text());fits.append(dict(family=family,fold=f,fit=path.parent.name,epoch=len(h),epochs=i['epochs'],complete=(path.parent/'fit_complete.json').exists()))
                except (ValueError,FileNotFoundError):pass
        if scope=='pilot':
            for kind in ['pilot','smoke']:
                folder=run/kind/family
                if not (folder/'verification.json').exists():raise RuntimeError('Pilot not complete '+str(folder))
                files.extend(p for p in folder.rglob('*') if p.is_file())
    status['arms'][arm]=dict(completed_folds=done,completed_fits=sum(v['complete'] for v in fits),fits=fits)
if scope!='status':
    files=sorted(set(files));receipt={p.relative_to(root).as_posix():sha(p) for p in files}
    archive=root/('collected-'+scope+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for p in files:tf.add(p,arcname=p.relative_to(root).as_posix())
    status.update(archive=str(archive),archive_sha256=sha(archive),files=receipt)
print(json.dumps(status))
'''
proc=subprocess.run([ssh,'-p','10000',*auth,host,'/usr/bin/python3.8','-',a.scope,','.join(jobs)],input=program,text=True,capture_output=True,check=True)
result=json.loads(proc.stdout);watch=HERE/'watch';watch.mkdir(exist_ok=True)
(watch/'latest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
if a.scope!='status':
    BIG.mkdir(parents=True,exist_ok=True);dest=BIG/Path(result['archive']).name
    subprocess.run([scp,'-P','10000',*auth,host+':'+result['archive'],str(dest)],check=True)
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==result['archive_sha256']
    received=BIG/'received';received.mkdir(exist_ok=True)
    with tarfile.open(dest) as tf:
        for m in tf.getmembers():
            target=(received/m.name).resolve();assert target.is_relative_to(received.resolve()) and m.isfile()
        tf.extractall(received,filter='data')
    for name,digest in result['files'].items():assert hashlib.sha256((received/name).read_bytes()).hexdigest()==digest
    (watch/(a.scope+'-receipt.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(utc=result['utc'],sacct=result['sacct'],arms={k:dict(completed_folds=len(v['completed_folds']),completed_fits=v['completed_fits']) for k,v in result['arms'].items()},received_files=len(result.get('files',{}))),indent=2))
