"""Retrieve the completed record-mean condition and run source audit only."""
from pathlib import Path
import hashlib,json,subprocess,shutil
H=Path(__file__).resolve().parent;ROOT=H.parents[3]
D=Path('D:/gait-research-data/D_P')/H.relative_to(ROOT)/'received-record-mean-20261031'
assert str(D.resolve()).lower().startswith('d:\\gait-research-data\\d_p\\') and shutil.disk_usage('D:/').free>2*1024**3
D.mkdir(parents=True,exist_ok=True)
remote='/abr/RESEARCH_USER/gdi-cnn-record-loss692-20261031'
reg=json.loads((ROOT/'gdi-research-harness/control/gpu-watch/registry.json').read_text())
assert any(e['id']=='cnn-record-loss-pair-692' and e['remote']==remote for e in reg['experiments'])
opts=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
code=r'''
from pathlib import Path
import json,hashlib,tarfile,os,sys
root=Path('/abr/RESEARCH_USER/gdi-cnn-record-loss692-20261031');os.chdir(root);sys.path.insert(0,str(root))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((root/'control/worker-pair.json').read_text())['exit_code']==0
files=sorted((root/'runs/main/record_mean').glob('fold*/*/latest.pt'));assert len(files)==25
files+=sorted(p for p in (root/'control').glob('*') if p.is_file())
files+=sorted(p for p in (root/'logs').glob('*') if p.is_file())
files+=[root/'runs/pair_summary.json',root/'runs/summary_complete.json']
archive=root/'runs/checkpoints-record_mean-20261031.tar.gz'
if not archive.exists():
    with tarfile.open(str(archive),'w:gz') as tf:
        for p in files:tf.add(str(p),arcname=str(p.relative_to(root)))
digests={str(p.relative_to(root)):sha(p) for p in files}
for p in [archive,root/'runs/main/record_mean-review.tar.gz']:digests[str(p.relative_to(root))]=sha(p)
print(json.dumps(dict(files=digests,summary=json.loads((root/'runs/main/record_mean/summary.json').read_text()))),flush=True)
'''
p=subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True)
receipt=json.loads(p.stdout);path=D/'receipt-record_mean-20261031.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,digest in receipt['files'].items():
    if not name.endswith('.tar.gz'):continue
    dest=D/Path(name).name
    if not dest.exists() or sha(dest)!=digest:subprocess.run(['scp',*opts,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+remote+'/'+name,str(dest)],check=True)
    assert sha(dest)==digest
    print(json.dumps(dict(received=dest.name,sha256=digest)),flush=True)
audit_code=r'''
import os,sys,json,numpy as np
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-cnn-record-loss692-20261031');os.chdir(root);sys.path.insert(0,str(root))
import run as r, audit
d=r.data();c=r.config();c.update(loss_variant='record_mean',name='cnn_record_mean_packed_records');fits=[]
for f in range(5):
    for j in [i for i in range(5) if i!=f]+[None]:
        path=root/'runs/main/record_mean'/('fold%d'%f)/('refit' if j is None else 'inner%d'%j)
        epochs=r.load(path/'fit_complete.json')['epochs']
        audit.audit_fit(r,d,c,f,j,path,epochs,'outer_refit' if j is None else 'inner_epoch_selection')
        fits.append(dict(fold=f,inner=j,passed=True,epochs=epochs,curve_sha256=r.sha(path/'curve.json'),checkpoint_sha256=r.sha(path/'latest.pt')))
print(json.dumps(dict(passed=True,original_audit_fit_unchanged=True,numpy=np.__version__,fits=fits,summary_sha256=r.sha(root/'runs/main/record_mean/summary.json'))))
'''
p=subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.8 -'],input=audit_code,text=True,capture_output=True,check=True)
source=H/'analysis/record_mean/source_audit.json';source.parent.mkdir(parents=True,exist_ok=True);source.write_text(json.dumps(json.loads(p.stdout),indent=2)+'\n')
proof=dict(received=str(D),receipt_sha256=sha(path),source_audit=str(source),source_audit_sha256=sha(source))
(source.parent/'receipt_pointer.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
