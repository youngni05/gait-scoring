"""Two one-GPU workers, with synthetic checks and a local-review pilot gate."""
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
import torch
import run as r


def execute(args):
    subprocess.run([sys.executable,'-u','-B']+args,cwd=str(r.ROOT),check=True)


def tasks():
    return [(v,f) for f in range(5) for v in (['gdi_only','knee_auxiliary'] if f%2==0 else ['knee_auxiliary','gdi_only'])]


def main():
    root=r.ROOT
    os.chdir(root)
    rank=int(os.environ['KNEE_RANK'])
    assert rank in [0,1]
    control=root/'control'
    control.mkdir(exist_ok=True)
    (control/('rank%d.lock'%rank)).mkdir()
    state=control/('worker-%d.json'%rank)
    info=dict(job_id=os.environ['SLURM_JOB_ID'],rank=rank,phase='checking',
              started_at=datetime.datetime.now().isoformat(),assignments=[task for i,task in enumerate(tasks()) if i%2==rank])
    r.b.save(state,info)

    def wait_for(predicate):
        start=time.monotonic()
        while not predicate():
            for path in control.glob('worker-*.json'):
                assert r.b.load(path).get('phase')!='training_or_summary_failed','Peer worker failed'
            if time.monotonic()-start>82000:raise TimeoutError('Peer/local pilot gate timed out')
            time.sleep(5)

    try:
        r.verify_manifest()
        r.b.setup(23,'cuda')
        assert torch.cuda.device_count()==1
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={line.split(',')[1].strip() for line in lines if line.split(',')[0].strip()==str(os.getpid())}
        assert len(uuids)==1
        info.update(uuid=next(iter(uuids)),hostname=os.uname().nodename,pid=os.getpid())
        r.b.save(state,info)
        execute(['verify.py','--device','cuda','--output','checks/gpu-rank%d'%rank])
        info['phase']='pilot' if rank==0 else 'waiting_local_pilot_review'
        r.b.save(state,info)
        if rank==0:
            execute(['run.py','pilot','--device','cuda','--output','runs/pilot'])
            for tag in ['weight-0p0','weight-1p0']:
                for device in ['cpu','cuda']:
                    execute(['review.py','--folder','runs/pilot/'+tag,'--device',device,
                             '--output','runs/pilot/'+tag+'/replay-'+device+'.json'])
            wait_for(lambda:(root/'checks/gpu-rank1/verification.json').exists())
            peers=[r.b.load(control/('worker-%d.json'%i)) for i in range(2)]
            assert len({p['uuid'] for p in peers})==2
            for i in range(2):assert r.b.load(root/('checks/gpu-rank%d/verification.json'%i))['passed']
            for tag in ['weight-0p0','weight-1p0']:
                for device in ['cpu','cuda']:assert r.b.load(root/'runs/pilot'/tag/('replay-'+device+'.json'))['passed']
            with tarfile.open(str(root/'runs/pilot-review.tar.gz'),'w:gz') as tf:
                for name in ['runs/pilot','checks','manifest.json']:
                    tf.add(str(root/name),arcname=name)
            r.b.save(control/'pilot_ready.json',dict(passed=True,manifest_sha256=r.b.sha(root/'manifest.json'),
                       archive_sha256=r.b.sha(root/'runs/pilot-review.tar.gz'),workers=peers))
        info['phase']='waiting_local_pilot_review'
        r.b.save(state,info)
        wait_for(lambda:(control/'approve-main.json').exists())
        assert r.b.load(control/'approve-main.json')['manifest_sha256']==r.b.sha(root/'manifest.json')
        assert r.b.load(control/'approve-main.json')['local_pilot_review_passed']
        info['phase']='main'
        r.b.save(state,info)
        for i,(variant,fold) in enumerate(tasks()):
            if i%2!=rank:continue
            (control/('%s-fold%d.lock'%(variant,fold))).mkdir()
            info.update(active_variant=variant,active_fold=fold)
            r.b.save(state,info)
            execute(['run.py','train','--variant',variant,'--fold',str(fold),'--device','cuda','--output','runs/main'])
        info['phase']='assigned_folds_complete'
        r.b.save(state,info)
        if rank==0:
            wait_for(lambda:all((root/'runs/main'/v/('fold%d'%f)/'complete.json').exists() for v,f in tasks()))
            execute(['run.py','summarize','--device','cpu','--output','runs/main'])
            execute(['review.py','--all','--folder','runs/main','--device','cpu',
                     '--output','runs/full-audit.json'])
            with tarfile.open(str(root/'runs/main-review.tar.gz'),'w:gz') as tf:
                for name in ['runs/main','runs/pilot','runs/full-audit.json','checks','control','manifest.json','protocol.json']:
                    tf.add(str(root/name),arcname=name)
            r.b.save(root/'runs/summary_complete.json',dict(summary_sha256=r.b.sha(root/'runs/main/summary.json'),
                        archive_sha256=r.b.sha(root/'runs/main-review.tar.gz')))
        info.update(phase='assigned_folds_complete',exit_code=0)
        r.b.save(state,info)
    except BaseException as exc:
        info.update(phase='training_or_summary_failed',exit_code=1,error=repr(exc))
        r.b.save(state,info)
        traceback.print_exc()
        raise


if __name__=='__main__':main()
