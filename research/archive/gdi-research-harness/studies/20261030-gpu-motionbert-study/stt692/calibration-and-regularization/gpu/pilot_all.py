"""Explicit future pilot command, four sequential arms on one allocated GPU."""
from pathlib import Path
import subprocess,sys,json,os
H=Path(__file__).resolve().parent
if __name__=='__main__':
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('GPU pilot requires a Slurm allocation')
    summaries={}
    for variant in ('baseline','person','decay','person_decay'):
        subprocess.run([sys.executable,'-u','-B',str(H/'run.py'),'check','--variant',variant,'--device','cuda','--output','checks/gpu'],cwd=H,check=True)
        subprocess.run([sys.executable,'-u','-B',str(H/'run.py'),'pilot','--variant',variant,'--device','cuda','--output','runs/pilot'],cwd=H,check=True)
        summaries[variant]=json.loads((H/'runs/pilot'/variant/'pilot_summary.json').read_text())
    (H/'runs/pilot/factorial_summary.json').write_text(json.dumps({'pilot_only':True,'full_training_started':False,'variants':summaries},indent=2)+'\n')
    print('FACTORIAL_PILOT_FINISHED',flush=True)
