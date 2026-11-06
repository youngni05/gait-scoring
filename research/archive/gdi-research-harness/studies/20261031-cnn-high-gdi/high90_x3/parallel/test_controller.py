"""CPU fixtures test orchestration only; no research data or optimizer calls."""
import importlib.util, json, tempfile, pathlib
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('parallel_controller',HERE/'worker_parallel.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

def fixture(root):
    w.save(root/'control/parallel-plan.json',{'assignments':w.ASSIGNMENTS,'source_sha256':{}})
    return [],[]

def pilot(root):
    w.save(root/'runs/pilot/pilot_summary.json',{'epochs':3})
    w.save(root/'runs/pilot/pilot_replay_cuda.json',{'passed':True})
    w.save(root/'checks/weight_check.json',{'passed':True})
    p=root/'runs/pilot/pilot/latest.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')

def callbacks(root,calls,states):
    def call(script,args,timeout):
        calls.append((script,args))
        if args[0]=='pilot': pilot(root)
        if args[0]=='train':
            fold=int(args[args.index('--fold')+1]); w.save(root/('runs/main/fold%d/complete.json'%fold),{'fixture':True})
        if args[0]=='final':
            w.save(root/'runs/main/summary.json',{'fits':25,'fold_metrics':[{}]*5,'metrics':[{'people':692,'records':1061}]})
            w.save(root/'runs/main/final_replay_cuda.json',{'passed':True})
    def status(phase,**extra): states.append(phase)
    def wait_for(name,peer):
        assert peer=='secondary'
        for fold in (3,4):w.save(root/('runs/main/fold%d/complete.json'%fold),{})
        w.save(root/name,{'folds':[3,4]})
    return call,status,wait_for

passed=[]
with tempfile.TemporaryDirectory(dir=str(HERE), prefix="fixture-") as tmp:
    root=pathlib.Path(tmp);calls,states=fixture(root)
    w.execute(root,'primary',*callbacks(root,calls,states))
    assert [int(a[a.index('--fold')+1]) for _,a in calls if a[0]=='train']==[0,1,2]
    assert [a[0] for _,a in calls].index('pilot')<[a[0] for _,a in calls].index('train')
    assert (root/'runs/summary_complete.json').exists()
    assert states[-1]=='assigned_folds_complete'
    passed.append('primary_pilot_gate_3_folds_single_final_export')
    try:w.claim(root/'control/parallel-primary.claim')
    except FileExistsError:passed.append('duplicate_role_rejected')
    else:raise AssertionError('duplicate role accepted')
    for fold in (0,1,2):
        try:w.claim(root/('control/fold%d.claim'%fold))
        except FileExistsError:pass
        else:raise AssertionError('duplicate fold accepted')
    passed.append('duplicate_folds_rejected')
    (root/'runs/pilot/pilot/latest.pt').write_bytes(b'tampered')
    try:w.validate_gate(root)
    except AssertionError:passed.append('changed_pilot_checkpoint_rejected')
    else:raise AssertionError('tampering accepted')
with tempfile.TemporaryDirectory(dir=str(HERE), prefix="fixture-") as tmp:
    root=pathlib.Path(tmp);calls,states=fixture(root);pilot(root)
    w.save(root/'control/parallel-gate.json',{'passed':True,'plan_sha256':w.sha(root/'control/parallel-plan.json'),
        'pilot_sha256':{'runs/pilot/pilot/latest.pt':w.sha(root/'runs/pilot/pilot/latest.pt')}})
    call,status,_=callbacks(root,calls,states)
    w.execute(root,'secondary',call,status,lambda n,p:None)
    assert [int(a[a.index('--fold')+1]) for _,a in calls if a[0]=='train']==[3,4]
    assert not any(a[0] in ('pilot','final') for _,a in calls)
    assert not (root/'runs/summary_complete.json').exists()
    passed.append('secondary_only_2_folds_no_export')
with tempfile.TemporaryDirectory(dir=str(HERE), prefix="fixture-") as tmp:
    root=pathlib.Path(tmp);calls,states=fixture(root)
    w.save(root/'control/parallel-gate.json',{'passed':False})
    call,status,_=callbacks(root,calls,states)
    try:w.execute(root,'secondary',call,status,lambda n,p:None)
    except AssertionError:pass
    else:raise AssertionError('failed pilot accepted')
    assert not any(a[0]=='train' for _,a in calls)
    passed.append('failed_pilot_blocks_main')
with tempfile.TemporaryDirectory(dir=str(HERE), prefix="fixture-") as tmp:
    root=pathlib.Path(tmp);calls,states=fixture(root)
    call,status,_=callbacks(root,calls,states)
    def incomplete(name,peer):w.save(root/name,{'folds':[3,4]})
    try:w.execute(root,'primary',call,status,incomplete)
    except AssertionError:pass
    else:raise AssertionError('missing outer folds accepted')
    assert not any(a[0]=='final' for _,a in calls)
    passed.append('missing_outer_fold_blocks_final')
w.save(HERE/'controller-verification.json',{'passed':True,'tests':passed,'real_data_optimizer_steps':0})
print(json.dumps({'passed':True,'tests':passed}))
