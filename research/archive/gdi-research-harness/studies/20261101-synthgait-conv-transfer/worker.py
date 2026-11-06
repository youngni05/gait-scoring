"""One GPU, verified pilot, explicit local audit gate, then fixed matched full comparison."""
from pathlib import Path
import os,sys,time,datetime,subprocess,tarfile,traceback
import torch
import run as r
import stem

def execute(args):subprocess.run([sys.executable,'-u','-B']+args,cwd=str(r.ROOT),check=True)
def archive(path,names):
    with tarfile.open(str(path),'w:gz') as tf:
        for name in names:
            if (r.ROOT/name).exists():tf.add(str(r.ROOT/name),arcname=name)
def main():
    os.chdir(r.ROOT);control=r.ROOT/'control';control.mkdir(exist_ok=True);(control/'worker.lock').mkdir()
    state=control/'worker-primary.json';info={'job_id':os.environ['SLURM_JOB_ID'],'phase':'checking',
        'started_at':datetime.datetime.now().isoformat(),'pid':os.getpid(),'hostname':os.uname().nodename}
    r.save(state,info)
    try:
        r.verify_manifest();r.setup(123,'cuda');assert torch.cuda.device_count()==1
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
        uuids={s.split(',')[1].strip() for s in rows if s.split(',')[0].strip()==str(os.getpid())};assert len(uuids)==1
        info['gpu_uuid']=next(iter(uuids));r.save(state,info)
        execute(['selfcheck.py','--device','cuda','--output','checks/gpu-selfcheck'])
        info['phase']='pilot';r.save(state,info);seed=r.config()['seed']+1
        execute(['pretrain_cli.py','--stage','pilot','--seed',str(seed),'--device','cuda'])
        for variant in ['scratch','synth']:
            dest='runs/pilot/'+variant
            execute(['run.py','pilot','--variant',variant,'--device','cuda','--output',dest])
            for device in ['cuda','cpu']:
                execute(['review.py','--variant',variant,'--device',device,'--folder',dest+'/pilot','--output',dest+'/replay-'+device+'.json'])
        proofs=[r.ROOT/'checks/gpu-selfcheck/verification.json']
        proofs += [r.ROOT/'runs/pilot'/v/('replay-'+d+'.json') for v in ['scratch','synth'] for d in ['cuda','cpu']]
        proofs += [r.ROOT/'runs/pretrain/pilot'/('seed%d'%seed)/('replay-'+d+'.json') for d in ['cuda','cpu']]
        for p in proofs:assert r.load(p)['passed']
        ckpaths=[r.ROOT/'runs/pilot'/v/'pilot/latest.pt' for v in ['scratch','synth']]
        ckpaths += [r.ROOT/'runs/pretrain/pilot'/('seed%d'%seed)/'selected.pt']
        gate={'passed':True,'manifest_sha256':r.sha(r.ROOT/'manifest.json'),
            'proofs':{str(p.relative_to(r.ROOT)):r.sha(p) for p in proofs},
            'checkpoints':{str(p.relative_to(r.ROOT)):r.sha(p) for p in ckpaths}}
        r.save(control/'server-pilot-passed.json',gate);info['phase']='waiting_local_pilot_audit';r.save(state,info)
        archive(r.ROOT/'runs/pilot-review.tar.gz',['runs/pilot','runs/pretrain/pilot','checks','control','manifest.json','protocol.json'])
        started=time.monotonic()
        while not (control/'approve-main-local.json').exists():
            if time.monotonic()-started>3600:raise TimeoutError('Local pilot audit approval not received within1hour')
            time.sleep(5)
        local=r.load(control/'approve-main-local.json');assert local['passed'] and local['manifest_sha256']==gate['manifest_sha256']
        assert local['checkpoints']==gate['checkpoints'];assert local['local_report_sha256']
        r.verify_manifest();info['phase']='main';r.save(state,info)
        for f in range(5):
            for variant in ['scratch','synth']:
                (control/('%s-fold%d.lock'%(variant,f))).mkdir()
                info.update(active_variant=variant,active_fold=f);r.save(state,info)
                if variant=='synth':
                    for j in [k for k in range(5) if k!=f]+[50]:
                        execute(['pretrain_cli.py','--stage','main','--seed',str(r.config()['seed']+f*100+j),'--device','cuda'])
                execute(['run.py','train','--variant',variant,'--device','cuda','--fold',str(f),'--output','runs/main/'+variant])
        info['phase']='summarizing';r.save(state,info);summaries={}
        for variant in ['scratch','synth']:
            execute(['run.py','summarize','--variant',variant,'--device','cpu','--output','runs/main/'+variant])
            q=r.load(r.ROOT/'runs/main'/variant/'summary.json');assert q['fits']==25;summaries[variant]=q
        external=list((r.ROOT/'runs/pretrain/main').glob('seed*/complete.json'));assert len(external)==25
        r.save(r.ROOT/'runs/pair_summary.json',{'variants':summaries,'GDI_fits':50,'outer_folds':10,'external_pretraining_fits':25,
            'source_movements':18,'independent_source_people_known':False})
        archive(r.ROOT/'runs/main-review.tar.gz',['runs/main','runs/pretrain','runs/pilot','runs/pair_summary.json','checks','control','manifest.json','protocol.json'])
        r.save(r.ROOT/'runs/summary_complete.json',{'summary_sha256':r.sha(r.ROOT/'runs/pair_summary.json'),
            'archive_sha256':r.sha(r.ROOT/'runs/main-review.tar.gz')})
        info.update(phase='assigned_folds_complete',exit_code=0);r.save(state,info)
    except BaseException as exc:
        info.update(phase='training_or_summary_failed',error=repr(exc),exit_code=1);r.save(state,info)
        if (r.ROOT/'runs').exists():archive(r.ROOT/'runs/failure-review.tar.gz',['runs/pilot','runs/pretrain/pilot','checks','control','manifest.json'])
        traceback.print_exc();raise
if __name__=='__main__':main()
