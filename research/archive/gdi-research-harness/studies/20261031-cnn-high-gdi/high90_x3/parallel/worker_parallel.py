"""Two single-GPU workers partition existing outer folds; frozen training code."""
import argparse, hashlib, json, os, subprocess, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENTS = {'primary': [0, 1, 2], 'secondary': [3, 4]}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text())

def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(temp), str(p))

def claim(p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        f.write(os.environ.get('SLURM_JOB_ID', 'test')+'\n')

def verify_sources(root):
    plan = load(root/'control/parallel-plan.json')
    assert plan['assignments'] == ASSIGNMENTS
    assert sorted(sum(ASSIGNMENTS.values(), [])) == list(range(5))
    for name, digest in plan['source_sha256'].items():
        assert sha(root/name) == digest, name
    return plan

def validate_gate(root):
    gate = load(root/'control/parallel-gate.json')
    assert gate['passed'] and gate['plan_sha256'] == sha(root/'control/parallel-plan.json')
    verify_sources(root)
    for name, digest in gate['pilot_sha256'].items():
        assert sha(root/name) == digest, name
    assert load(root/'runs/pilot/pilot_replay_cuda.json')['passed']
    assert load(root/'runs/pilot/pilot_summary.json')['epochs'] == 3

def export(root):
    out = root/'runs/main'
    summary = load(out/'summary.json')
    assert summary['fits'] == 25 and len(summary['fold_metrics']) == 5
    assert summary['metrics'][0]['people'] == 692 and summary['metrics'][0]['records'] == 1061
    assert len(list(out.glob('fold*/complete.json'))) == 5
    assert load(out/'final_replay_cuda.json')['passed']
    archive = root/'runs/main-review.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and (p.suffix in ('.json','.npz','.csv') or (p.name=='latest.pt' and p.parent.name=='refit')):
                tf.add(p, arcname='results/'+p.relative_to(out).as_posix())
        for folder in ('checks', 'runs/pilot', 'control'):
            for p in sorted((root/folder).rglob('*')):
                if p.is_file(): tf.add(p, arcname=p.relative_to(root).as_posix())
        for p in root.iterdir():
            if p.suffix in ('.py','.json'): tf.add(p, arcname='bundle/'+p.name)
    save(root/'runs/summary_complete.json', {'sha256':sha(archive),'archive':str(archive)})

def execute(root, role, call, status, wait_for):
    verify_sources(root)
    claim(root/('control/parallel-'+role+'.claim'))
    status('checks')
    call('run.py',['check','--device','cpu','--output','checks/parallel-'+role],300)
    call('run.py',['smoke','--device','cuda','--output','checks/parallel-'+role+'/smoke'],600)
    if role == 'primary':
        call('audit64.py',['weights','--output','checks'],300)
        status('pilot')
        call('run.py',['pilot','--device','cuda','--output','runs/pilot'],1800)
        call('audit.py',['pilot','--device','cuda','--output','runs/pilot'],600)
        assert load(root/'runs/pilot/pilot_replay_cuda.json')['passed']
        names = ['runs/pilot/pilot/latest.pt','runs/pilot/pilot_summary.json',
                 'runs/pilot/pilot_replay_cuda.json','checks/weight_check.json']
        save(root/'control/parallel-gate.json', {'passed':True,
            'plan_sha256':sha(root/'control/parallel-plan.json'),
            'pilot_sha256':{n:sha(root/n) for n in names}})
    else:
        status('waiting_for_pilot_gate')
        wait_for('control/parallel-gate.json', 'primary')
    validate_gate(root)
    for fold in ASSIGNMENTS[role]:
        verify_sources(root)
        claim(root/('control/fold%d.claim'%fold))
        status('training', current_fold=fold)
        call('run.py',['train','--device','cuda','--output','runs/main','--fold',str(fold)],60000)
        assert (root/('runs/main/fold%d/complete.json'%fold)).exists()
    save(root/('control/parallel-'+role+'-done.json'), {'folds':ASSIGNMENTS[role]})
    if role == 'secondary':
        status('assigned_folds_complete'); return
    status('waiting_for_secondary')
    wait_for('control/parallel-secondary-done.json','secondary')
    assert load(root/'control/parallel-secondary-done.json')['folds'] == ASSIGNMENTS['secondary']
    assert len(list((root/'runs/main').glob('fold*/complete.json'))) == 5
    status('auditing')
    call('audit.py',['final','--device','cuda','--output','runs/main'],3600)
    verify_sources(root)
    export(root)
    status('assigned_folds_complete', archive_sha256=sha(root/'runs/main-review.tar.gz'))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('role',choices=ASSIGNMENTS)
    role=parser.parse_args().role
    jobs=load(ROOT/'control/parallel-jobs.json')
    job=os.environ['SLURM_JOB_ID']
    assert jobs[role] == job and sys.executable == '/usr/bin/python3.8'
    deadline=time.monotonic()+63000
    def status(phase, **extra):
        save(ROOT/('control/worker-parallel-'+role+'.json'), dict(job_id=job,
            role=role, phase=phase, folds=ASSIGNMENTS[role], time=time.time(), **extra))
    def call(script,args,timeout):
        left=deadline-time.monotonic()
        if left<=0: raise TimeoutError('Worker budget expired')
        subprocess.run([sys.executable,'-u','-B',str(ROOT/script)]+args,
                       cwd=str(ROOT),check=True,timeout=min(timeout,left))
    def wait_for(name, peer):
        while not (ROOT/name).exists():
            p=ROOT/('control/worker-parallel-'+peer+'.json')
            if p.exists() and load(p)['phase']=='training_or_summary_failed':
                raise RuntimeError('Peer failed: '+peer)
            state=subprocess.check_output(['sacct','-nP','-j',jobs[peer],'--format=JobIDRaw,State'],text=True)
            rows=[s.split('|') for s in state.splitlines()]
            for row in rows:
                if row[0]==jobs[peer] and row[1].split()[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED'):
                    if not (ROOT/name).exists(): raise RuntimeError('Peer ended before required output: '+str(row))
            if time.monotonic()>deadline: raise TimeoutError('Peer wait expired')
            time.sleep(20)
    try:
        execute(ROOT,role,call,status,wait_for)
    except BaseException as exc:
        status('training_or_summary_failed',error=repr(exc)); raise

if __name__=='__main__': main()
