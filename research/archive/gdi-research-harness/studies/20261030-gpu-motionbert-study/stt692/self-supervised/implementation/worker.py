"""One GPU, pilot plus GPU/CPU audit gate, then all five folds."""
from pathlib import Path
import json,os,subprocess,sys,time,tarfile,traceback
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(str(t),str(p))
def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(*args,**kw):subprocess.run([sys.executable,'-u','-B']+list(args),cwd=ROOT,check=True,**kw)
def main():
    job=os.environ['SLURM_JOB_ID'];state=dict(job_id=job,phase='checking',folds=list(range(5)),reused_allocation=os.getenv('SSL_REUSE_ALLOCATION')=='1')
    dest=ROOT/'control/worker-ssl.json';save(dest,state)
    try:
        execute('capacity.py','--min-free','0',timeout=60)
        execute('run.py','check','--device','cuda','--output','checks/gpu',timeout=300)
        execute('verify_ssl.py','--device','cuda','--output','checks/ssl-gpu',timeout=600)
        execute('verify.py','--device','cuda','--output','checks/synthetic-gpu',timeout=900)
        state.update(phase='pilot');save(dest,state)
        execute('run.py','pilot','--device','cuda','--output','runs/pilot',timeout=1800)
        execute('review_pilot.py','--device','cuda','--input','runs/pilot','--output','checks/replay-gpu',timeout=600)
        execute('review_pilot.py','--device','cpu','--input','runs/pilot','--output','checks/replay-cpu',timeout=900)
        with tarfile.open(ROOT/'runs/pilot-review.tar.gz','w:gz') as tf:
            for parent in ['runs/pilot','checks']:
                for p in (ROOT/parent).rglob('*'):
                    if p.is_file():tf.add(str(p),arcname=p.relative_to(ROOT).as_posix())
            tf.add(str(ROOT/'manifest.json'),arcname='manifest.json')
        state.update(phase='pilot_verified',archive_sha256=sha(ROOT/'runs/pilot-review.tar.gz'));save(dest,state)
        pilot=load(ROOT/'runs/pilot/pilot_summary.json');gatepath=ROOT/'control/approve-main.json'
        checks={name:load(ROOT/('checks/'+name+'/verification.json')) for name in ['ssl-gpu','synthetic-gpu','replay-gpu','replay-cpu']}
        assert all(v['passed'] for v in checks.values())
        save(gatepath,dict(approved=True,bundle_sha256=sha(ROOT/'manifest.json'),pilot_summary_sha256=sha(ROOT/'runs/pilot/pilot_summary.json'),
            pilot_checkpoint_sha256=pilot['checkpoint_sha256'],ssl_checkpoint_sha256=pilot['ssl_checkpoint_sha256'],
            verification_sha256={name:sha(ROOT/('checks/'+name+'/verification.json')) for name in checks},
            approval_basis='User authorized main training after pilot passes; automatic fixed GPU and CPU audit gate'))
        gate=load(gatepath);pilot=load(ROOT/'runs/pilot/pilot_summary.json')
        assert gate['approved'] is True and gate['bundle_sha256']==sha(ROOT/'manifest.json')
        assert gate['pilot_summary_sha256']==sha(ROOT/'runs/pilot/pilot_summary.json')
        assert gate['pilot_checkpoint_sha256']==pilot['checkpoint_sha256']==sha(ROOT/'runs/pilot/fit/latest.pt')
        assert gate['ssl_checkpoint_sha256']==pilot['ssl_checkpoint_sha256']==sha(ROOT/'runs/pilot/fit/pretrain/latest.pt')
        state.update(phase='training',gate=gate);save(dest,state)
        execute('run.py','train','--device','cuda','--output','runs/main')
        save(ROOT/'runs/summary_complete.json',dict(summary_sha256=sha(ROOT/'runs/main/summary.json'),archive_sha256=sha(ROOT/'runs/main-review.tar.gz')))
        state.update(phase='assigned_folds_complete',exit_code=0);save(dest,state)
    except Exception as e:
        state.update(phase='training_or_summary_failed',exit_code=1,error=repr(e));save(dest,state);raise
if __name__=='__main__':main()
