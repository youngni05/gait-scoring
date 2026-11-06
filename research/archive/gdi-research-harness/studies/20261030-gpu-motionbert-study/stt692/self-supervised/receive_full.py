"""Retrieve completed SSL evidence without touching training or allocation controls."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile
H=Path(__file__).resolve().parent
ROOT=H.parents[4]
DEST=Path('D:/gait-research-data/D_P')/H.relative_to(ROOT)/'received-full-20261031'
REMOTE='/abr/RESEARCH_USER/gdi-stt692-ssl-20261031'
OPTS=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((ROOT/'gdi-research-harness/control/gpu-watch/registry.json').read_text())
assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
assert any(e['id']=='stt-self-supervised-692' and e['remote']==REMOTE for e in reg['experiments'])
assert str(DEST.resolve()).lower().startswith('d:\\gait-research-data\\d_p\\')
assert shutil.disk_usage('D:/').free>2*1024**3
DEST.mkdir(parents=True,exist_ok=True)
code=r'''
import os,pathlib,json,hashlib,tarfile
root=pathlib.Path('/abr/RESEARCH_USER/gdi-stt692-ssl-20261031');os.chdir(root)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((root/'control/worker-ssl.json').read_text())['exit_code']==0
assert (root/'runs/main/summary.json').exists()
files=sorted((root/'runs/main').glob('fold*/*/latest.pt'))+sorted((root/'runs/main').glob('fold*/*/pretrain/latest.pt'))
assert len(files)==50
files+=sorted(p for p in (root/'control').glob('*') if p.is_file())
files+=sorted(p for p in (root/'logs').glob('*') if p.is_file())
files+=sorted(p for p in (root/'checks').rglob('*') if p.is_file())
files+=[root/'runs/summary_complete.json']
archive=root/'runs/checkpoints-review-20261031.tar.gz'
if not archive.exists():
    with tarfile.open(str(archive),'w:gz') as tf:
        for p in files:tf.add(str(p),arcname=str(p.relative_to(root)))
digests={str(p.relative_to(root)):sha(p) for p in files}
for p in [archive,root/'runs/main-review.tar.gz',root/'runs/pilot-reuse-review.tar.gz']:
    assert p.exists();digests[str(p.relative_to(root))]=sha(p)
print(json.dumps(dict(files=digests,summary=json.loads((root/'runs/main/summary.json').read_text()))))
'''
p=subprocess.run(['ssh',*OPTS,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True)
receipt=json.loads(p.stdout)
(DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
for name,digest in receipt['files'].items():
    if not name.endswith('.tar.gz'):continue
    dest=DEST/Path(name).name
    if not dest.exists() or sha(dest)!=digest:
        subprocess.run(['scp',*OPTS,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/'+name,str(dest)],check=True)
    assert sha(dest)==digest
    print(json.dumps(dict(received=dest.name,bytes=dest.stat().st_size,sha256=digest)),flush=True)
for name,target in [('main-review.tar.gz','unpacked'),('checkpoints-review-20261031.tar.gz','checkpoints'),('pilot-reuse-review.tar.gz','pilot')]:
    with tarfile.open(DEST/name) as tf:tf.extractall(DEST/target,filter='data')
for name,digest in receipt['files'].items():
    if not name.endswith('.tar.gz'):assert sha(DEST/'checkpoints'/name)==digest,name
for p in (DEST/'checkpoints/runs/main').rglob('latest.pt'):
    dest=DEST/'unpacked/results'/p.relative_to(DEST/'checkpoints/runs/main')
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
for p in (DEST/'unpacked/bundle').iterdir():assert sha(p)==sha(H/'bundle'/p.name),p.name
proof=dict(passed=True,received=str(DEST),receipt_sha256=sha(DEST/'receipt.json'),files=receipt['files'],source_bundle_unchanged=True)
(H/'analysis').mkdir(exist_ok=True)
(H/'analysis/transfer_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(passed=True,received=str(DEST),receipt_sha256=proof['receipt_sha256'])),flush=True)
