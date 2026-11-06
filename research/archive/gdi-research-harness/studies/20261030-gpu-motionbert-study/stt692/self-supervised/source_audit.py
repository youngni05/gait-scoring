"""Run unchanged auditors in original Python/NumPy; no optimizer or GPU action."""
from pathlib import Path
import json,subprocess,hashlib
H=Path(__file__).resolve().parent
code=r'''
import os,sys,json,hashlib
from pathlib import Path
root=Path('/abr/RESEARCH_USER/gdi-stt692-ssl-20261031');os.chdir(root);sys.path.insert(0,str(root))
import run as r
import numpy as np
d=r.data();c=r.load(root/'protocol.json');fits=[]
for f in range(5):
    for j in [i for i in range(5) if i!=f]+[None]:
        path=root/'runs/main'/('fold%d'%f)/('refit' if j is None else 'inner%d'%j)
        epochs=r.load(path/'fit_complete.json')['epochs']
        h,n=r.audit_fit(d,c,f,j,epochs,path,'outer_refit' if j is None else 'inner_epoch_selection')
        fits.append(dict(fold=f,inner=j,passed=True,epochs=len(h),steps=n,checkpoint_sha256=r.sha(path/'latest.pt'),curve_sha256=r.sha(path/'curve.json')))
print(json.dumps(dict(passed=True,original_audit_fit_unchanged=True,python=sys.version,numpy=np.__version__,fits=fits,summary_sha256=r.sha(root/'runs/main/summary.json'),run_sha256=r.sha(root/'run.py'),pretrain_sha256=r.sha(root/'pretrain.py'))))
'''
args=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub','-p','10000','RESEARCH_USER@PRIVATE_HOST','env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.8 -']
p=subprocess.run(args,input=code,text=True,capture_output=True,timeout=900)
out=H/'analysis/source_audit.json'
if p.returncode:
    (H/'analysis/source_audit_failure.log').write_text(p.stderr+p.stdout)
    raise RuntimeError(p.stderr)
v=json.loads(p.stdout);assert v['passed'] and len(v['fits'])==25
out.write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(passed=True,fits=25,sha256=hashlib.sha256(out.read_bytes()).hexdigest())),flush=True)
