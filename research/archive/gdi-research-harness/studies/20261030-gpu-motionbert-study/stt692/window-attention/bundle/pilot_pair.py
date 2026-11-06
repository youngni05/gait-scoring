"""Sequential two-arm GPU pilot only. Does not start full training or reserve GPUs."""
from pathlib import Path
import sys,os,subprocess,tarfile,json
import run as r
ROOT=Path(__file__).resolve().parent
def execute(*args):subprocess.run([sys.executable,'-u','-B']+list(args),cwd=ROOT,check=True)
def main():
    r.require(bool(os.getenv('SLURM_JOB_ID')),'Run within an explicitly allocated Slurm GPU job')
    state=dict(job_id=os.environ['SLURM_JOB_ID'],phase='pilot',full_training_started=False)
    r.save(ROOT/'control/worker-pilot.json',state)
    try:
        execute('verify_pooling.py','--device','cuda','--output','checks/gpu-pooling')
        for mode in ['mean','attention']:
            execute('run.py','check','--variant',mode,'--device','cuda','--output','checks/'+mode)
            execute('verify.py','--variant',mode,'--device','cuda','--output','checks/synthetic-'+mode)
            execute('run.py','pilot','--variant',mode,'--device','cuda','--output','runs/pilot/'+mode)
            execute('review_pilot.py','--variant',mode,'--device','cuda','--input','runs/pilot/'+mode,'--output','checks/replay-'+mode)
        summaries={v:r.load(ROOT/'runs/pilot'/v/'pilot_summary.json') for v in ['mean','attention']}
        r.require(summaries['mean']['optimizer_steps']==summaries['attention']['optimizer_steps'],'Unequal pilot updates')
        r.save(ROOT/'runs/pilot/pair_summary.json',dict(pilot_only=True,full_training_started=False,passed=True,variants=summaries,
            primary_comparison_pending=True,note='Three epochs test operation, timing and memory; do not choose a model or claim OOF performance from pilot MAE.'))
        with tarfile.open(ROOT/'runs/pilot-review.tar.gz','w:gz') as tf:
            for name in ['runs/pilot','checks','manifest.json','protocol.json']:tf.add(str(ROOT/name),arcname=name)
        state.update(phase='pilot_complete',exit_code=0);r.save(ROOT/'control/worker-pilot.json',state)
        print('POOLING_PAIR_PILOT_FINISHED',flush=True)
    except Exception as error:
        state.update(phase='pilot_failed',exit_code=getattr(error,'returncode',1),error=str(error));r.save(ROOT/'control/worker-pilot.json',state);raise
if __name__=='__main__':main()
