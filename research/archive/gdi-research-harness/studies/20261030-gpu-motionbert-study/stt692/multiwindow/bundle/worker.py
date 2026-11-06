"""Two disjoint fold workers; explicit pilot gate; one bounded GPU hold."""
from pathlib import Path
import argparse,datetime,json,os,re,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');os.replace(str(temp),str(p))
def sha(p):
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(*args,timeout=None):
    subprocess.run([sys.executable,'-u','-B']+list(args),cwd=ROOT,check=True,timeout=timeout)
def validate_gate(gate):
    g=load(gate);p=load(ROOT/'runs/pilot/pilot_summary.json')
    if not(g.get('approved') is True and p['passed'] and p['pilot_only'] and not p['outer_evaluation_used']
        and g['bundle_sha256']==sha(ROOT/'manifest.json') and g['pilot_summary_sha256']==sha(ROOT/'runs/pilot/pilot_summary.json')
        and g['pilot_checkpoint_sha256']==p['checkpoint_sha256']==sha(ROOT/'runs/pilot/fit/latest.pt')):
        raise ValueError('Pilot approval gate mismatch')
    return g
def try_finalize():
    # flock automatically releases on process death; outputs are not shared until
    # all five complete.json files exist. Do not overwrite other worker state.
    import fcntl
    out=ROOT/'runs/main'
    with (out/'summary.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        if not all((out/('fold%d/complete.json'%f)).exists() for f in range(5)):return False
        if not (ROOT/'runs/summary_complete.json').exists():
            execute('run.py','summarize','--device','cpu','--output','runs/main')
            save(ROOT/'runs/summary_complete.json',dict(summary_sha256=sha(out/'summary.json'),archive_sha256=sha(ROOT/'runs/main-review.tar.gz')))
        return True
def hold_until(end_epoch,release_file,state,training_code):
    state.update(phase='holding_gpu_after_training',training_exit_code=training_code,
        hold_until=datetime.datetime.fromtimestamp(end_epoch).isoformat(),release_file=str(release_file))
    save(ROOT/'control'/('worker-%s.json'%state['role']),state)
    print(json.dumps(state),flush=True)
    while time.time()<end_epoch-5 and not release_file.exists():time.sleep(5)
    state.update(phase='releasing_gpu',release_reason='requested' if release_file.exists() else 'allocation_end')
    save(ROOT/'control'/('worker-%s.json'%state['role']),state)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['primary','secondary']);a=ap.parse_args()
    job=os.environ['SLURM_JOB_ID']
    line=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    def field(name):return re.search(r'\b'+name+r'=([^ ]+)',line).group(1)
    start=datetime.datetime.fromisoformat(field('StartTime')).timestamp()
    end=datetime.datetime.fromisoformat(field('EndTime')).timestamp()
    if end-start>86401:raise ValueError('Allocation exceeds approved 24 hours')
    folds=[0,1,2] if a.role=='primary' else [3,4]
    state=dict(role=a.role,job_id=job,folds=folds,started_at=field('StartTime'),allocation_end=field('EndTime'),phase='checking')
    dest=ROOT/'control'/('worker-%s.json'%a.role);save(dest,state)
    execute('capacity.py','--min-free','1')
    execute('run.py','check','--device','cuda','--output','checks/gpu-'+job,timeout=300)
    if a.role=='primary':
        execute('verify.py','--device','cuda','--output','checks/synthetic-'+job,timeout=600)
        execute('verify_nested.py','--device','cuda','--output','checks/nested-'+job,timeout=900)
        execute('verify_windows.py','--output','checks/windows-'+job,timeout=300)
        execute('run.py','pilot','--device','cuda','--output','runs/pilot',timeout=900)
        state.update(phase='pilot_complete_waiting_for_verified_gate');save(dest,state)
        gate=ROOT/'control/approve-main.json'
        until=min(time.time()+2700,end-60)
        while not gate.exists() and time.time()<until:time.sleep(5)
        if not gate.exists():raise TimeoutError('No verified pilot gate within 45 minutes')
    gate=validate_gate(ROOT/'control/approve-main.json')
    execute('capacity.py','--min-free','1')
    state.update(phase='training',gate=gate);save(dest,state)
    code=0
    try:
        execute('run.py','train','--device','cuda','--output','runs/main','--folds',*[str(f) for f in folds])
        state.update(phase='assigned_folds_complete');save(dest,state)
        try_finalize()
    except subprocess.CalledProcessError as error:
        code=error.returncode;state.update(phase='training_or_summary_failed',exit_code=code);save(dest,state)
        print(json.dumps(state),flush=True)
    if a.role=='primary':
        hold_until(end,ROOT/'control'/('release-'+job),state,code)
    raise SystemExit(code)
if __name__=='__main__':main()
