"""Three one-GPU jobs; shared verified pilot gate and exclusive outer-fold claims."""
from pathlib import Path
import datetime,os,subprocess,sys,time,tarfile,traceback
import torch
import run as r

def execute(args):subprocess.run([sys.executable,'-u','-B']+args,cwd=str(r.ROOT),check=True)
def tasks():return [(v,f) for f in range(5) for v in ['mse','ranksim']]

def main():
    root=r.ROOT;os.chdir(root);rank=int(os.environ['BMC_RANK']);assert rank in range(3)
    control=root/'control';control.mkdir(exist_ok=True);(control/('rank%d.lock'%rank)).mkdir()
    state=control/('worker-%d.json'%rank)
    info=dict(job_id=os.environ['SLURM_JOB_ID'],rank=rank,phase='checking',started_at=datetime.datetime.now().isoformat(),started_epoch=time.time(),assignments=[x for i,x in enumerate(tasks()) if i%3==rank])
    r.save(state,info)
    try:
        r.verify_manifest();r.setup(123,'cuda');assert torch.cuda.device_count()==1
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={line.split(',')[1].strip() for line in rows if line.split(',')[0].strip()==str(os.getpid())}
        assert len(uuids)==1
        info.update(uuid=next(iter(uuids)),hostname=os.uname().nodename,pid=os.getpid());r.save(state,info)
        execute(['verify.py','--device','cuda','--output','checks/gpu-rank%d'%rank])
        info['phase']='pilot';r.save(state,info)
        if rank<2:
            variant=['mse','ranksim'][rank];dest='runs/pilot/'+variant
            execute(['run.py','pilot','--device','cuda','--variant',variant,'--output',dest])
            for device in ['cuda','cpu']:
                execute(['review.py','--device',device,'--variant',variant,'--folder',dest+'/pilot','--output',dest+'/replay-'+device+'.json'])
        info['phase']='waiting_pilot_gate';r.save(state,info)
        def wait_for(condition,seconds=82000):
            start=time.monotonic()
            while not condition():
                assert not any(r.load(p).get('phase')=='training_or_summary_failed' for p in control.glob('worker-*.json')),'Partner failed'
                if time.monotonic()-start>seconds:raise TimeoutError('Worker synchronization timeout')
                time.sleep(5)
        proofs=[root/'checks'/('gpu-rank%d'%i)/'verification.json' for i in range(3)]
        proofs += [root/'runs/pilot'/v/('replay-'+device+'.json') for v in ['mse','ranksim'] for device in ['cuda','cpu']]
        wait_for(lambda:all(p.exists() for p in proofs))
        peers=[r.load(control/('worker-%d.json'%i)) for i in range(3)]
        assert len({p['uuid'] for p in peers})==3
        for p in proofs:assert r.load(p)['passed']
        for variant in ['mse','ranksim']:
            dest=root/'runs/pilot'/variant
            q=r.load(dest/'pilot_summary.json')
            assert q['epochs']==3 and q['train_people']==416 and q['validation_people']==138 and not q['outer_evaluation_used']
            for device in ['cpu','cuda']:
                p=r.load(dest/('replay-'+device+'.json'))
                assert p['checkpoint_sha256']==r.sha(dest/'pilot/latest.pt') and p['tolerance']==1e-4
                assert max(p['max_raw_error'],p['max_corrected_error'])<1e-4
        r.verify_manifest()
        gate=dict(passed=True,proofs={str(p.relative_to(root)):r.sha(p) for p in proofs},manifest_sha256=r.sha(root/'manifest.json'),workers=peers)
        if rank==0:
            r.save(control/'approve-main.json',gate)
            with tarfile.open(str(root/'runs/pilot-review.tar.gz'),'w:gz') as tf:
                for n in ['runs/pilot','checks','control','manifest.json']:tf.add(str(root/n),arcname=n)
        wait_for(lambda:(control/'approve-main.json').exists())
        assert r.load(control/'approve-main.json')['proofs']==gate['proofs']
        info['phase']='waiting_local_pilot_audit';r.save(state,info)
        wait_for(lambda:(control/'approve-main-local.json').exists(),seconds=21600)
        local=r.load(control/'approve-main-local.json')
        assert local['passed'] and local['manifest_sha256']==gate['manifest_sha256'] and local['proofs']==gate['proofs']
        assert local['local_report_sha256']
        info['phase']='main';r.save(state,info)
        for i,(variant,fold) in enumerate(tasks()):
            if i%3!=rank:continue
            (control/('%s-fold%d.lock'%(variant,fold))).mkdir()
            info.update(active_variant=variant,active_fold=fold);r.save(state,info)
            execute(['run.py','train','--device','cuda','--variant',variant,'--fold',str(fold),'--output','runs/main/'+variant])
        info['phase']='assigned_folds_complete';r.save(state,info)
        if rank==0:
            wait_for(lambda:all((root/'runs/main'/v/('fold%d'%f)/'complete.json').exists() for v,f in tasks()))
            summaries={}
            for variant in ['mse','ranksim']:
                execute(['run.py','summarize','--device','cpu','--variant',variant,'--output','runs/main/'+variant])
                q=r.load(root/'runs/main'/variant/'summary.json');assert q['fits']==25
                summaries[variant]=q
            r.save(root/'runs/pair_summary.json',dict(variants=summaries,fits=50,outer_folds=10))
            with tarfile.open(str(root/'runs/main-review.tar.gz'),'w:gz') as tf:
                for n in ['runs/main','runs/pilot','runs/pair_summary.json','checks','control','manifest.json']:
                    tf.add(str(root/n),arcname=n)
            r.save(root/'runs/summary_complete.json',dict(summary_sha256=r.sha(root/'runs/pair_summary.json'),archive_sha256=r.sha(root/'runs/main-review.tar.gz')))
        info.update(phase='assigned_folds_complete',exit_code=0);r.save(state,info)
        print('MAE_POINT_WORKER_FINISHED',rank,flush=True)
    except BaseException as exc:
        info.update(phase='training_or_summary_failed',exit_code=1,error=repr(exc));r.save(state,info);traceback.print_exc();raise
if __name__=='__main__':main()
