"""One allocation: both pilot audits, then two immutable CNN training runs in order."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('window_control','record_mean')
MANIFEST='128b153bd8cacbeda749d6574167bb2e2c7237cf888912be2204f08ca616074b'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(str(tmp),str(p))
def verify_bundle():
    assert sha(ROOT/'manifest.json')==MANIFEST
    for name,digest in read(ROOT/'manifest.json').items():assert sha(ROOT/name)==digest
def gate():
    verify_bundle();proof={};pilots={}
    synthetic=read(ROOT/'checks/gpu/verification.json')
    assert synthetic['passed'] and synthetic['bundle_sha256']==MANIFEST
    assert synthetic['real_data_optimizer_steps']==0
    for variant in VARIANTS:
        parent=ROOT/'runs/pilot'/variant;pilot=read(parent/'pilot_summary.json')
        assert pilot['pilot_only'] and pilot['epochs']==3 and pilot['train_people']==416 and pilot['validation_people']==138
        assert not pilot['outer_evaluation_used']
        checkpoint=parent/'pilot/latest.pt';digest=sha(checkpoint)
        done=read(parent/'pilot/fit_complete.json')
        assert done['checkpoint_sha256']==digest and done['epochs']==3 and done['optimizer_steps']==954
        for device in ('gpu','cpu'):
            path=ROOT/('checks/pilot-%s-%s.json'%(variant,device));report=read(path)
            assert report['passed'] and report['variant']==variant and report['checkpoint_sha256']==digest
            assert report['device']==('cuda' if device=='gpu' else 'cpu') and report['tolerance']==1e-4
            assert report['max_corrected_prediction_error']<1e-4 and not report['outer_evaluation_used']
            proof[str(path.relative_to(ROOT))]=sha(path)
        for path in (checkpoint,parent/'pilot_summary.json',parent/'pilot/fit_complete.json'):
            proof[str(path.relative_to(ROOT))]=sha(path)
        pilots[variant]=pilot
    proof['checks/gpu/verification.json']=sha(ROOT/'checks/gpu/verification.json')
    return dict(approved=True,bundle_sha256=MANIFEST,proof_sha256=proof,pilots=pilots,
        approval_basis='User authorized both main runs sequentially only after both fixed pilot audits pass')
def validate_gate(g):
    assert g['approved'] and g['bundle_sha256']==MANIFEST
    verify_bundle()
    for name,digest in g['proof_sha256'].items():assert sha(ROOT/name)==digest
def archive_pilot(name='pilot-review.tar.gz'):
    p=ROOT/'runs'/name;p.parent.mkdir(exist_ok=True)
    with tarfile.open(p,'w:gz') as tf:
        for directory in ('runs/pilot','checks','control'):
            for item in sorted((ROOT/directory).rglob('*')):
                if item.is_file() and item.suffix!='.lock':tf.add(str(item),arcname=item.relative_to(ROOT).as_posix())
        tf.add(str(ROOT/'manifest.json'),arcname='manifest.json')
    return sha(p)
def execute(args,timeout):
    subprocess.run(args,cwd=str(ROOT),check=True,timeout=timeout)
def main():
    job=os.environ['SLURM_JOB_ID'];(ROOT/'control').mkdir(exist_ok=True)
    with (ROOT/'control/execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any((ROOT/'runs/main').glob('*/fold*')), 'Existing main outputs: no automatic restart'
        state=dict(job_id=job,phase='checking',kind='full_training_pair',variants=list(VARIANTS),expected_fits=50,expected_outer_folds=10,
            started_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),completed_variants=[])
        path=ROOT/'control/worker-pair.json';save(path,state)
        try:
            verify_bundle();state.update(phase='pilot');save(path,state)
            execute(['bash','launch.sh','pilot'],4500)
            approval=gate();save(ROOT/'control/approve-main-pair.json',approval)
            state.update(phase='pilot_verified',pilot_archive_sha256=archive_pilot());save(path,state)
            summaries={}
            for variant in VARIANTS:
                validate_gate(approval);state.update(phase='training',active_variant=variant);save(path,state)
                execute(['bash','launch.sh','train',variant],28000)
                out=ROOT/'runs/main'/variant;s=read(out/'summary.json')
                assert s['fits']==25 and len(s['fold_metrics'])==5 and s['loss_variant']==variant
                assert s['metrics'][0]['people']==692 and s['metrics'][0]['records']==1061
                assert sha(out/'oof_predictions.npz')==s['oof_sha256']
                summaries[variant]=dict(summary_sha256=sha(out/'summary.json'),archive_sha256=sha(out.parent/(variant+'-review.tar.gz')),metrics=s['metrics'],fits=25)
                save(ROOT/'runs'/('complete-'+variant+'.json'),summaries[variant])
                state['completed_variants'].append(variant);save(path,state)
            save(ROOT/'runs/pair_summary.json',dict(variants=summaries,fits=50,outer_folds=10,bundle_sha256=MANIFEST))
            save(ROOT/'runs/summary_complete.json',dict(summary_sha256=sha(ROOT/'runs/pair_summary.json'),variants=summaries))
            state.update(phase='assigned_folds_complete',exit_code=0);save(path,state)
        except Exception as exc:
            state.update(phase='training_or_summary_failed',exit_code=1,error=repr(exc));save(path,state)
            if not (ROOT/'control/approve-main-pair.json').exists():archive_pilot('pilot-failed-review.tar.gz')
            raise
if __name__=='__main__':main()
