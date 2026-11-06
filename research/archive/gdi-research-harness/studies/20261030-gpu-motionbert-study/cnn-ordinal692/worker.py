"""Two independent GPU workers in one allocation; global pilot gate, unique tasks."""
import os,time,traceback,datetime,subprocess,tarfile
from pathlib import Path
import run as r
import torch
def assignments(c):return [(g,f) for f in range(5) for g in c['groups']]
def pilot_specs(c):return [(g,l) for g in c['groups'] for l in (c['joint_lambdas'] if g.startswith('joint') else [1.0])]
def main():
    rank=int(os.environ.get('SLURM_PROCID',os.environ.get('ORDINAL_RANK','-1')));assert rank in (0,1)
    root=r.H;os.chdir(root);c=r.read(root/'protocol.json');d=r.data();job=os.environ['SLURM_JOB_ID']
    control=root/'control';control.mkdir(exist_ok=True);state=control/('worker-%d.json'%rank)
    lock=control/('rank%d.lock'%rank);lock.mkdir() # Never silently retry an existing worker.
    info=dict(job_id=job,rank=rank,phase='pilot',started_at=datetime.datetime.now().isoformat())
    try:
        r.setup(c['seed'],'cuda');assert torch.cuda.device_count()==1,'Each worker must see exactly one assigned GPU'
        prop=torch.cuda.get_device_properties(0)
        # CUDA_VISIBLE_DEVICES can be "0" for both cgroup-isolated workers.
        # Resolve this process's physical GPU via nvidia-smi instead.
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={line.split(',')[1].strip() for line in rows if line.split(',')[0].strip()==str(os.getpid())}
        assert len(uuids)==1,('GPU process identity unavailable',os.getpid(),rows)
        info.update(visible=os.environ.get('CUDA_VISIBLE_DEVICES'),uuid=next(iter(uuids)),gpu=prop.name,hostname=os.uname().nodename,pid=os.getpid())
        r.save(state,info)
        for i,(group,lam) in enumerate(pilot_specs(c)):
            if i%2!=rank:continue
            out=root/'runs/pilot'/group/('lambda'+str(lam))/'fit'
            before=time.monotonic();torch.cuda.reset_peak_memory_stats()
            r.fit(d,c,group,0,1,lam,c['pilot_epochs'],out,'cuda')
            gpu=r.replay(d,c,group,0,1,lam,c['pilot_epochs'],out,'cuda')
            cpu=r.replay(d,c,group,0,1,lam,c['pilot_epochs'],out,'cpu')
            r.save(out.parent/'pilot_summary.json',dict(passed=True,pilot_only=True,group=group,lam=lam,epochs=c['pilot_epochs'],
                outer_evaluation_used=False,train_people=len(set(d['person_id'][r.split(d,0,1)[0]])),validation_people=len(set(d['person_id'][r.split(d,0,1)[1]])),
                gpu=gpu,cpu=cpu,seconds=time.monotonic()-before,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del gpu,cpu;torch.cuda.empty_cache()
        info['phase']='waiting_pilot_gate';r.save(state,info)
        def wait_for(condition,seconds=3600):
            start=time.monotonic()
            while not condition():
                if any(r.read(p).get('phase')=='failed' for p in control.glob('worker-*.json')):raise RuntimeError('Partner worker failed')
                if time.monotonic()-start>seconds:raise TimeoutError('Partner timeout')
                time.sleep(3)
        pilot_paths=[root/'runs/pilot'/g/('lambda'+str(l))/'pilot_summary.json' for g,l in pilot_specs(c)]
        wait_for(lambda:all(p.exists() for p in pilot_paths))
        peers=[r.read(control/('worker-%d.json'%i)) for i in range(2)]
        assert peers[0]['uuid']!=peers[1]['uuid'],'Duplicate physical GPU'
        assert all(r.read(p)['passed'] for p in pilot_paths)
        # All seven frozen pilot checkpoints and original code must still match.
        for p in pilot_paths:
            q=r.read(p);assert r.sha(p.parent/'fit/latest.pt')==q['gpu']['checkpoint_sha256']==q['cpu']['checkpoint_sha256']
        r.data()
        gate=dict(passed=True,pilot_summaries={str(p.relative_to(root)):r.sha(p) for p in pilot_paths},manifest_sha256=r.sha(root/'manifest.json'),peers=peers)
        if rank==0:r.save(control/'approve-main.json',gate)
        wait_for(lambda:(control/'approve-main.json').exists())
        saved=r.read(control/'approve-main.json');assert saved['pilot_summaries']==gate['pilot_summaries']
        info['phase']='main';r.save(state,info)
        for i,(group,f) in enumerate(assignments(c)):
            if i%2!=rank:continue
            claim=control/('%s-fold%d.lock'%(group,f));claim.mkdir()
            info.update(active_group=group,active_fold=f);r.save(state,info)
            r.train_outer(d,c,group,f,root/'runs/main','cuda')
        info['phase']='worker_folds_complete';r.save(state,info)
        if rank==0:
            wait_for(lambda:all((root/'runs/main'/g/('fold%d'%f)/'complete.json').exists() for g,f in assignments(c)),24*3600)
            r.summarize(d,c,root/'runs/main')
            archive=root/'runs/main-review.tar.gz'
            with tarfile.open(str(archive),'w:gz') as tf:
                for p in sorted((root/'runs/main').rglob('*')):
                    if p.is_file() and p.suffix in ('.json','.npz'):tf.add(str(p),arcname='results/'+str(p.relative_to(root/'runs/main')))
            r.save(root/'runs/summary_complete.json',dict(summary_sha256=r.sha(root/'runs/main/summary.json'),archive_sha256=r.sha(archive)))
        info['phase']='assigned_folds_complete';info['exit_code']=0;r.save(state,info)
        print('ORDINAL_WORKER_FINISHED',rank,flush=True)
    except BaseException as e:
        info.update(phase='failed',exit_code=1,error=repr(e));r.save(state,info);traceback.print_exc();raise
if __name__=='__main__':main()
