from pathlib import Path
import subprocess,json,hashlib,tarfile,datetime,shutil
H=Path('D:/Projects/D_P/studies/20261103-cnn-stable-scale');B=Path('D:/gait-research-data/D_P/cnn-stable-scale-20261103/received/final');REMOTE='/abr/RESEARCH_USER/cnn-stable-scale-20261103'
OWNER='01a19c6f-49f7-7300-97b0-a303b71ffe91'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
r=json.loads((H/'watch/registry.json').read_text(encoding='utf8'));assert r['thread_id']==OWNER and r['automation_id']=='gdi-cnn';e=next(x for x in r['experiments'] if x['id']=='cnn-stable-scale-20261103');assert e['jobs']==['34737','34738'] and e['remote']==REMOTE
code='''
import pathlib,json,hashlib,tarfile
root=pathlib.Path(%r)
assert (root/'runs/summary_complete.json').exists()
files=set(p for folder in ['runs','control','checks','logs','original'] for p in (root/folder).rglob('*') if p.is_file())
files.update(p for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.json','.slurm'])
assert len(list((root/'runs/main').glob('*/*/*/fit_complete.json')))==50
inventory={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(files)}
archive=root/'final-originals-review.tar.gz'
with tarfile.open(str(archive),'w:gz') as tf:
 for p in sorted(files):tf.add(str(p),arcname=str(p.relative_to(root)),recursive=False)
print(json.dumps(dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=inventory)))
'''%REMOTE
p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST,'/usr/bin/python3.8 -'],input=code,text=True,encoding='utf8',capture_output=True,timeout=180);assert p.returncode==0,p.stderr;packet=json.loads(p.stdout)
assert shutil.disk_usage(B.parent).free>packet['bytes']+sum(v['bytes'] for v in packet['files'].values())+1024**3
B.mkdir(parents=True,exist_ok=True);save(H/'recovery-remote-inventory.json',packet);archive=B/'final-originals-review.tar.gz'
if not archive.exists() or sha(archive)!=packet['sha256']:
    subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[HOST+':'+packet['path'],str(archive)],check=True,timeout=600)
assert sha(archive)==packet['sha256']
root=B/'original';root.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
    for m in tf.getmembers():assert m.isfile() and root.resolve() in (root/m.name).resolve().parents
    tf.extractall(root,filter='data')
for n,v in packet['files'].items():assert sha(root/n)==v['sha256'],n
receipt=dict(status='recovered_sha256_verified',checked_at=datetime.datetime.now().astimezone().isoformat(),received_root=str(root),archive=str(archive),archive_sha256=packet['sha256'],files=len(packet['files']),bytes=packet['bytes'],uncompressed_bytes=sum(v['bytes'] for v in packet['files'].values()),jobs=e['jobs'],owner_thread=OWNER)
save(H/'recovery.json',receipt);print(json.dumps(receipt),flush=True)
