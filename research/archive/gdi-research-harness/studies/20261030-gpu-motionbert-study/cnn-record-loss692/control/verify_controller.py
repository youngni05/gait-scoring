"""Exercise gating and sequential orchestration with synthetic filesystem fixtures only."""
from pathlib import Path
import ast,importlib.util,json,os,sys,types
H=Path(__file__).resolve().parents[1]
if os.name=='nt':sys.modules['fcntl']=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
spec=importlib.util.spec_from_file_location('worker_pair',H/'control/worker_pair.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
original_verify=w.verify_bundle
def fixtures():
    w.save(w.ROOT/'manifest.json',{})
    w.save(w.ROOT/'checks/gpu/verification.json',dict(passed=True,bundle_sha256=w.MANIFEST,real_data_optimizer_steps=0))
    for variant in w.VARIANTS:
        parent=w.ROOT/'runs/pilot'/variant;(parent/'pilot').mkdir(parents=True,exist_ok=True)
        (parent/'pilot/latest.pt').write_bytes(b'synthetic checkpoint')
        digest=w.sha(parent/'pilot/latest.pt')
        w.save(parent/'pilot_summary.json',dict(pilot_only=True,epochs=3,train_people=416,validation_people=138,outer_evaluation_used=False))
        w.save(parent/'pilot/fit_complete.json',dict(checkpoint_sha256=digest,epochs=3,optimizer_steps=954))
        for device in ('cpu','gpu'):
            w.save(w.ROOT/('checks/pilot-%s-%s.json'%(variant,device)),dict(passed=True,variant=variant,device='cuda' if device=='gpu' else device,checkpoint_sha256=digest,tolerance=1e-4,max_corrected_prediction_error=0.,outer_evaluation_used=False))
def main():
    ast.parse((H/'control/worker_pair.py').read_text(),feature_version=(3,8))
    original_root=w.ROOT;w.ROOT=H/'bundle';original_verify();w.ROOT=original_root
    root=H/'checks/controller';root.mkdir(exist_ok=True)
    # Never reuse potentially stale test outputs.
    import time
    root=root/str(time.time_ns());root.mkdir()
    cases=[]
    w.ROOT=root/'gate';fixtures();w.verify_bundle=lambda:None
    gate=w.gate();w.validate_gate(gate);cases.append('valid_gate')
    p=w.ROOT/'checks/pilot-record_mean-cpu.json';old=p.read_bytes();v=w.read(p);v['max_corrected_prediction_error']=0.001;w.save(p,v)
    try:w.gate()
    except AssertionError:cases.append('failed_replay_blocks_main')
    else:raise AssertionError('Bad replay accepted')
    p.write_bytes(old);(w.ROOT/'runs/pilot/window_control/pilot/latest.pt').write_bytes(b'corrupt')
    try:w.validate_gate(gate)
    except AssertionError:cases.append('changed_checkpoint_blocks_main')
    else:raise AssertionError('Changed checkpoint accepted')
    os.environ['SLURM_JOB_ID']='synthetic_only'
    for failure in (False,True):
        w.ROOT=root/('failure' if failure else 'success');w.ROOT.mkdir();calls=[]
        def execute(args,timeout):
            calls.append(args[2:])
            if args[2]=='pilot':
                fixtures()
                if failure:raise RuntimeError('synthetic pilot failure')
            else:
                variant=args[3];out=w.ROOT/'runs/main'/variant;out.mkdir(parents=True)
                (out/'oof_predictions.npz').write_bytes(b'synthetic')
                (out.parent/(variant+'-review.tar.gz')).write_bytes(b'synthetic archive')
                w.save(out/'summary.json',dict(fits=25,fold_metrics=[{}]*5,loss_variant=variant,metrics=[dict(people=692,records=1061)],oof_sha256=w.sha(out/'oof_predictions.npz')))
        w.execute=execute
        try:w.main()
        except RuntimeError:
            assert failure and calls==[['pilot']]
            assert w.read(w.ROOT/'control/worker-pair.json')['phase']=='training_or_summary_failed'
            cases.append('pilot_exception_stops_all_training')
        else:
            assert not failure and calls==[['pilot'],['train','window_control'],['train','record_mean']]
            assert w.read(w.ROOT/'runs/pair_summary.json')['fits']==50
            cases.append('both_pilots_before_two_sequential_main_runs')
    report=dict(passed=True,cases=cases,synthetic_only=True,remote_mutations=0,worker_sha256=w.sha(H/'control/worker_pair.py'))
    w.save(H/'control/verification.json',report);print(json.dumps(report))
if __name__=='__main__':main()
