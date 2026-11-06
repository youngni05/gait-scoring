"""Retrieve the immutable pilot evidence and independently replay all seven fits."""
from pathlib import Path
import hashlib,json,os,subprocess,tarfile,shutil
H=Path(__file__).resolve().parent
sub=json.loads((H/'submission.json').read_text())
remote=sub['remote']
dest=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-ordinal692/received-pilot-v2')
dest.mkdir(parents=True,exist_ok=True)
opts=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
code='''from pathlib import Path
import hashlib,json,tarfile
p=Path(%r);gate=json.loads((p/'control/approve-main.json').read_text());assert gate['passed']
assert len(gate['pilot_summaries'])==7
for name,digest in gate['pilot_summaries'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
archive=p/'runs/pilot-review-v2.tar.gz'
with tarfile.open(str(archive),'w:gz') as tf:
    for name in ['runs/pilot','control/approve-main.json','manifest.json','protocol.json']:
        tf.add(str(p/name),arcname=name)
print(json.dumps(dict(archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),gate=gate,workers=[json.loads(f.read_text()) for f in sorted((p/'control').glob('worker-*.json'))])))
'''%remote
res=subprocess.run(['ssh',*opts,'-p','10000','RESEARCH_USER@PRIVATE_HOST','/usr/bin/python3.8 -'],input=code,text=True,capture_output=True,check=True)
receipt=json.loads(res.stdout)
archive=dest/'pilot-review-v2.tar.gz'
subprocess.run(['scp',*opts,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+receipt['archive'],str(archive)],check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(archive)==receipt['sha256']
unpacked=dest/'unpacked';unpacked.mkdir(exist_ok=True)
with tarfile.open(archive) as tf:
    for member in tf.getmembers():
        target=(unpacked/member.name).resolve()
        assert unpacked.resolve() in target.parents and not member.issym() and not member.islnk()
    tf.extractall(unpacked)
assert sha(unpacked/'manifest.json')==sha(H/'manifest.json')==receipt['gate']['manifest_sha256']
os.environ['GDI_ORDINAL_DATA']=str(H.parent/'cnn-record-loss692/bundle/data.npz')
import run as r
d=r.data();c=r.read(H/'protocol.json');checks=[]
for name,digest in receipt['gate']['pilot_summaries'].items():
    p=unpacked/name;assert sha(p)==digest
    q=r.read(p);assert q['passed'] and q['epochs']==3 and not q['outer_evaluation_used']
    assert q['train_people']==416 and q['validation_people']==138
    history=r.audit(d,c,q['group'],0,1,q['lam'],3,p.parent/'fit')
    # replay() writes its report, so run it against a separate working copy.
    working=dest/'local-replay'/q['group']/('lambda'+str(q['lam']))/'fit'
    shutil.copytree(p.parent/'fit',working)
    replay=r.replay(d,c,q['group'],0,1,q['lam'],3,working,'cpu')
    checks.append(dict(group=q['group'],lam=q['lam'],epoch_count=len(history),local_cpu=replay,
        server_gpu=q['gpu'],server_cpu=q['cpu'],peak_allocated_bytes=q['peak_allocated_bytes'],
        mean_epoch_seconds=sum(x['seconds'] for x in history)/len(history),total_seconds=q['seconds']))
    print(json.dumps(checks[-1]),flush=True)
assert len(checks)==7 and len({p['uuid'] for p in receipt['gate']['peers']})==2
receipt.update(local_archive=str(archive),server_local_sha256_matched=True,passed=True)
r.save(H/'checks/pilot-v2-receipt.json',receipt)
r.save(H/'checks/pilot-v2-verification.json',dict(passed=True,pilots=7,epochs=21,train_people=416,validation_people=138,
    outer_people=138,outer_evaluation_used=False,checks=checks,manifest_sha256=sha(H/'manifest.json'),
    receipt_sha256=sha(H/'checks/pilot-v2-receipt.json'),local_optimizer_steps=0,protected_final_opened=False))
print('PILOT_RETRIEVAL_AND_LOCAL_VERIFICATION_FINISHED',flush=True)
