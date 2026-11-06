"""One allocated GPU: verify -> pilot -> replay gate -> full nested training -> audit/export."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parent
def status(phase, **extra):
    p=ROOT/'control'/('worker-'+os.environ.get('SLURM_JOB_ID','local')+'.json')
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(phase=phase,time=time.time(),**extra),indent=2),encoding='utf-8')
def call(script,args,timeout):
    subprocess.run([sys.executable,'-B',str(ROOT/script)]+args,cwd=str(ROOT),check=True,timeout=timeout)
def main():
    try:
        status('checks')
        call('run.py',['check','--device','cpu','--output','checks'],180)
        call('audit.py',['weights','--output','checks'],300)
        call('run.py',['smoke','--device','cuda','--output','checks/smoke'],180)
        status('pilot')
        call('run.py',['pilot','--device','cuda','--output','runs/pilot'],1800)
        call('audit.py',['pilot','--device','cuda','--output','runs/pilot'],600)
        status('training')
        call('run.py',['train','--device','cuda','--output','runs/main'],61000)
        status('auditing')
        call('audit.py',['final','--device','cuda','--output','runs/main'],3600)
        status('exporting')
        archive=ROOT/'runs/main-review.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for p in sorted((ROOT/'runs/main').rglob('*')):
                if p.is_file() and (p.suffix in ('.json','.npz','.csv') or (p.name=='latest.pt' and p.parent.name=='refit')):
                    tf.add(p,arcname='results/'+p.relative_to(ROOT/'runs/main').as_posix())
            for folder in ('checks','runs/pilot'):
                for p in sorted((ROOT/folder).rglob('*')):
                    if p.is_file():tf.add(p,arcname=p.relative_to(ROOT).as_posix())
            for p in ROOT.iterdir():
                if p.suffix in ('.py','.json'):tf.add(p,arcname='bundle/'+p.name)
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        (ROOT/'runs/summary_complete.json').write_text(json.dumps({'sha256':digest,'archive':str(archive)}),encoding='utf-8')
        status('complete',archive_sha256=digest)
    except Exception as exc:
        status('failed',error=str(exc));raise
if __name__=='__main__':main()
