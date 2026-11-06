"""Reuse held allocation once, audit existing pilot, then run original full training."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,tarfile,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run as r
def execute(*args,**kw):subprocess.run([sys.executable,'-u','-B']+list(args),cwd=ROOT,check=True,**kw)
def main():
    assert os.environ['SLURM_JOB_ID']=='32844'
    with (ROOT/'control/reuse-execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hold=r.load(Path('/abr/RESEARCH_USER/gdi-stt692-multiwindow-20261031/control/worker-primary.json'))
        assert hold['phase']=='holding_gpu_after_training' and hold['training_exit_code']==0
        assert not (ROOT/'runs/main').exists(),'Do not restart an already started full run'
        state=dict(job_id='32844',step_id=os.environ.get('SLURM_STEP_ID'),previous_job_id='32897',reused_allocation=True,phase='checking',folds=list(range(5)))
        dest=ROOT/'control/worker-ssl.json';r.save(dest,state)
        try:
            execute('run.py','check','--device','cuda','--output','checks/reuse-gpu',timeout=300)
            execute('review_pilot.py','--device','cuda','--input','runs/pilot','--output','checks/replay-reuse-gpu',timeout=600)
            execute('control/replay64.py',timeout=900)
            check=r.load(ROOT/'checks/replay-reuse-cpu64/verification.json');assert check['passed']
            gate=dict(approved=True,bundle_sha256=r.sha(ROOT/'manifest.json'),
                pilot_summary_sha256=r.sha(ROOT/'runs/pilot/pilot_summary.json'),
                pilot_checkpoint_sha256=r.sha(ROOT/'runs/pilot/fit/latest.pt'),
                ssl_checkpoint_sha256=r.sha(ROOT/'runs/pilot/fit/pretrain/latest.pt'),
                gpu_replay_sha256=r.sha(ROOT/'checks/replay-reuse-gpu/verification.json'),
                cpu64_replay_sha256=r.sha(ROOT/'checks/replay-reuse-cpu64/verification.json'),
                original_cpu32_failure_sha256=r.sha(ROOT/'checks/replay-cpu/verification.json'),
                unchanged_tolerance=1e-4,approval_basis='User authorized reuse and pilot to full; existing STT float64 reference procedure; original float32 failure retained')
            r.save(ROOT/'control/approve-main-reuse.json',gate)
            with tarfile.open(ROOT/'runs/pilot-reuse-review.tar.gz','w:gz') as tf:
                for parent in ['runs/pilot','checks','control']:
                    for p in (ROOT/parent).rglob('*'):
                        if p.is_file() and p.suffix!='.lock':tf.add(str(p),arcname=p.relative_to(ROOT).as_posix())
                tf.add(str(ROOT/'manifest.json'),arcname='manifest.json')
            state.update(phase='training',gate=gate,pilot_archive_sha256=r.sha(ROOT/'runs/pilot-reuse-review.tar.gz'));r.save(dest,state)
            execute('run.py','train','--device','cuda','--output','runs/main',timeout=42000)
            r.save(ROOT/'runs/summary_complete.json',dict(summary_sha256=r.sha(ROOT/'runs/main/summary.json'),archive_sha256=r.sha(ROOT/'runs/main-review.tar.gz')))
            state.update(phase='assigned_folds_complete',exit_code=0);r.save(dest,state)
        except Exception as error:
            state.update(phase='training_or_summary_failed',exit_code=1,error=repr(error));r.save(dest,state);raise
if __name__=='__main__':main()
