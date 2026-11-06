"""One GPU: checks and pilot, pinned approval, all folds; no allocation holding."""
import os,sys,time,json,subprocess
from pathlib import Path
import run as r
ROOT=Path(__file__).resolve().parent
def execute(*args):subprocess.run([sys.executable,'-u','-B']+list(args),cwd=ROOT,check=True)
def main():
    job=os.environ['SLURM_JOB_ID'];state=dict(job_id=job,phase='pilot',folds=list(range(5)));dest=ROOT/'control/worker-bilateral.json';r.save(dest,state)
    try:
        execute('run.py','check','--device','cuda','--output','checks/gpu-'+job)
        for name,folder in [('verify_structure.py','structure'),('verify.py','synthetic'),('verify_nested.py','nested'),('verify_windows.py','windows')]:
            args=[name,'--output','checks/'+folder+'-'+job]
            if folder!='windows':args+=['--device','cuda']
            execute(*args)
        execute('run.py','pilot','--device','cuda','--output','runs/pilot')
        state['phase']='waiting_pilot_review';r.save(dest,state)
        gate=ROOT/'control/approve-main.json';deadline=time.time()+2700
        while not gate.exists():
            if time.time()>deadline:raise RuntimeError('Pilot approval timeout; no main training')
            time.sleep(5)
        g=r.load(gate);p=r.load(ROOT/'runs/pilot/pilot_summary.json')
        r.require(g['approved'] and g['bundle_sha256']==r.sha(ROOT/'manifest.json') and g['pilot_summary_sha256']==r.sha(ROOT/'runs/pilot/pilot_summary.json') and g['pilot_checkpoint_sha256']==p['checkpoint_sha256']==r.sha(ROOT/'runs/pilot/fit/latest.pt'),'Pilot gate mismatch')
        state['phase']='training';r.save(dest,state)
        execute('run.py','train','--device','cuda','--output','runs/main')
        r.save(ROOT/'runs/summary_complete.json',dict(summary_sha256=r.sha(ROOT/'runs/main/summary.json'),archive_sha256=r.sha(ROOT/'runs/main-review.tar.gz')))
        state.update(phase='assigned_folds_complete',training_exit_code=0);r.save(dest,state)
    except Exception as error:
        state.update(phase='training_or_summary_failed',training_exit_code=getattr(error,'returncode',1),error=str(error));r.save(dest,state);raise
if __name__=='__main__':main()
