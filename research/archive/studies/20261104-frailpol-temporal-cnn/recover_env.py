"""Recover only synthetic hardware checks and independently replay locally."""
from pathlib import Path
import sys,json,subprocess,tarfile,importlib.util,hashlib
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')
def ssh(cmd):
    p=subprocess.run(['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+['RESEARCH_USER@PRIVATE_HOST',cmd],capture_output=True,text=True,encoding='utf8',timeout=120);assert p.returncode==0,p.stderr;return p.stdout.strip()
import numpy as np,torch
r=json.loads((H/'environment-submission.json').read_text());remote=r['remote'];dest=B/'received/synthetic-environment';dest.mkdir(parents=True,exist_ok=True);archive=dest/'environment-review.tar.gz'
digest=json.loads(ssh('cat '+remote+'/environment-review.sha256.json'))['sha256']
p=subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+['RESEARCH_USER@PRIVATE_HOST:'+remote+'/environment-review.tar.gz',str(archive)],capture_output=True,text=True,timeout=120);assert p.returncode==0,p.stderr;assert sha(archive)==digest
root=dest/'original';root.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
    for member in t.getmembers():
        target=(root/member.name).resolve();assert target==root.resolve() or root.resolve() in target.parents;assert not member.issym() and not member.islnk()
    t.extractall(root,filter='data')
summary=json.loads((root/'checks/summary.json').read_text());assert summary['data_origin'].startswith('All input tensors') and len(summary['runs'])==4;assert sha(root/'train.py')==summary['training_library_sha256']==sha(H/'bundle/train.py');assert sha(root/'env.py')==summary['env_code_sha256']
spec=importlib.util.spec_from_file_location('synthetic_verified_train',root/'train.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);torch.set_num_threads(4);checks=[]
for row in summary['runs']:
    c=row['condition'];pt=root/'checks'/(c+'.pt');assert sha(pt)==row['checkpoint_sha256'];data=np.load(root/'checks'/(c+'.npz'));m=module.Model(c,row['nfeatures']).double();m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True));m.eval()
    with torch.no_grad():replay=m(torch.from_numpy(data['x']).double(),torch.from_numpy(data['f']).double()).numpy()
    delta=float(np.max(np.abs(replay-data['prediction'])));assert np.allclose(replay,data['prediction'],atol=1e-6,rtol=1e-6);checks.append(dict(condition=c,checkpoint_sha256=sha(pt),cpu64_max_delta=delta,seconds_per100synthetic_train_steps=row['seconds']))
result=dict(status='passed',job=r['job'],data_origin='SYNTHETIC_ONLY, generated on server; no human signals or clinical labels transferred.',server_local_archive_sha256=digest,models_replayed=4,checks=checks,environment=summary['environment'],limitation='Hardware/code/save/replay and rough timing only, no TUG/MediaPipe prediction performance. Real-data export and training remain pending explicit approval.')
save(H/'environment-verification.json',result);print(json.dumps(result,indent=2))
