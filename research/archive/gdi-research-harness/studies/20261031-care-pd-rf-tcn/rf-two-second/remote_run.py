"""Own CPU jobs only, with hashed transfer and separate pilot/main."""
import sys,json,tarfile,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REMOTE='/abr/RESEARCH_USER/care-pd-rf2s-20261031'
sys.path.insert(0,str(BASE));from remote import command,AUTH,HOST
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2)+'\n')
def main():
    mode=sys.argv[1];prefix='import pathlib,json,hashlib,tarfile,subprocess\np=pathlib.Path('+repr(REMOTE)+')\n'
    if mode=='pilot':
        assert not (HERE/'submission.json').exists()
        with tarfile.open(HERE/'bundle.tar.gz','w:gz') as t:
            for name in ['experiment.py','cpu_runner.py','study.py','protocol.json','input.npz']:t.add(HERE/name,arcname=name)
        digest=sha(HERE/'bundle.tar.gz');subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[str(HERE/'bundle.tar.gz'),HOST+':'+REMOTE+'.tar.gz'],check=True)
        code=prefix+"z=p.with_suffix('.tar.gz');assert hashlib.sha256(z.read_bytes()).hexdigest()=="+repr(digest)+"\nassert not p.exists();p.mkdir();tarfile.open(z).extractall(p);(p/'logs').mkdir()\nr=subprocess.check_output(['sbatch','--job-name=care-rf2s','--account=iucp','--partition=p02','--qos=normal','--exclude=DIS04,DIS05,DIS06','--cpus-per-task=2','--mem=8G','--time=00:15:00','--output=logs/%j.log','--wrap=/usr/bin/python3 -u cpu_runner.py pilot'],cwd=p,text=True);print(json.dumps(dict(job=r.strip().split()[-1])))"
        result=json.loads(command(code));save('submission.json',dict(remote=REMOTE,pilot_job=result['job'],status='pilot_submitted',bundle_sha256=digest))
        rp=BASE/'watch/registry.json';reg=json.loads(rp.read_text());assert reg['thread_id']=='01a18b70-7da2-7430-a223-d4b314940cb3';assert not any(r['id']=='care-pd-rf2s' for r in reg['experiments'])
        reg['experiments'].append(dict(id='care-pd-rf2s',kind='rf_retraining',local=str(HERE),remote=REMOTE,jobs=[result['job']],pilot_jobs=[result['job']],preserve_allocations=['32844'],expected_fits=270,expected_final_models=90,expected_outer_folds=6,review_state='pending',reported_digest=None,run_root='main/fits',marker_path='main/complete.json',summary_path='main/complete.json',audit='Read rf-two-second protocol and submission. CPU RF only. Pilot1model is not full result. Main90final models+180inner fits: full matched,first2,center2 x6folds x5seeds. remote_run.py collect; analyze.py; verify.py. Same110people2914records and MoVi14people56clips. No labels for MoVi. Do not submit training or alter other jobs.'));rp.write_text(json.dumps(reg,indent=2)+'\n');print(json.dumps(result))
    elif mode=='main':
        sub=json.loads((HERE/'submission.json').read_text());assert 'main_job' not in sub
        code=prefix+"d=json.loads((p/'pilot/complete.json').read_text());assert d['passed'] and d['final_models']==1\ns=subprocess.check_output(['sacct','-j',"+repr(sub['pilot_job'])+",'--format=JobID,State,ExitCode','-n','-P'],text=True);assert "+repr(sub['pilot_job']+'|COMPLETED|0:0')+" in s\nassert not (p/'main').exists()\nr=subprocess.check_output(['sbatch','--job-name=care-rf2s','--account=iucp','--partition=p02','--qos=normal','--exclude=DIS04,DIS05,DIS06','--cpus-per-task=12','--mem=16G','--time=01:00:00','--output=logs/%j.log','--wrap=/usr/bin/python3 -u cpu_runner.py main'],cwd=p,text=True);print(json.dumps(dict(job=r.strip().split()[-1])))"
        result=json.loads(command(code));sub.update(main_job=result['job'],status='main_submitted');save('submission.json',sub);rp=BASE/'watch/registry.json';reg=json.loads(rp.read_text());assert reg['thread_id']=='01a18b70-7da2-7430-a223-d4b314940cb3';next(r for r in reg['experiments'] if r['id']=='care-pd-rf2s')['jobs']=[result['job']];rp.write_text(json.dumps(reg,indent=2)+'\n');print(json.dumps(result))
    elif mode=='status':
        sub=json.loads((HERE/'submission.json').read_text());jobs=[sub[k] for k in ['pilot_job','main_job'] if k in sub]
        code=prefix+"print(subprocess.check_output(['sacct','-j',"+repr(','.join(jobs))+",'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,NodeList','-P'],text=True))\nfor mode in ['pilot','main']:\n print(mode,'models',len(list((p/mode/'fits').glob('*/*.json'))))\n f=p/mode/'complete.json'\n if f.exists():print(f.read_text())\nfor f in (p/'logs').glob('*'):print(f.name,f.read_text()[-400:])\n"
        result=command(code);(HERE/'status.txt').write_text(result);print(result)
    elif mode=='collect':
        sub=json.loads((HERE/'submission.json').read_text())
        code=prefix+"d=json.loads((p/'main/complete.json').read_text());assert d['passed'] and d['final_models']==90\ns=subprocess.check_output(['sacct','-j',"+repr(sub['main_job'])+",'--format=JobID,State,ExitCode,Elapsed','-P','-n'],text=True);assert "+repr(sub['main_job']+'|COMPLETED|0:0')+" in s\n(p/'slurm-final.txt').write_text(s)\nfiles=[f for root in ['main','pilot','logs'] for f in (p/root).rglob('*') if f.is_file()]+[p/n for n in ['experiment.py','cpu_runner.py','study.py','protocol.json','input.npz','slurm-final.txt']]\nz=p/'result.tar.gz'\nwith tarfile.open(z,'w:gz') as t:\n for f in files:t.add(f,arcname=str(f.relative_to(p)))\nprint(json.dumps(dict(archive_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})))"
        result=json.loads(command(code,timeout=300));dest=HERE/'result.tar.gz';subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[HOST+':'+REMOTE+'/result.tar.gz',str(dest)],check=True);assert sha(dest)==result['archive_sha256'];out=HERE/'received';out.mkdir(exist_ok=True)
        with tarfile.open(dest) as t:
            assert all((out/m.name).resolve().is_relative_to(out.resolve()) for m in t.getmembers());t.extractall(out,filter='data')
        for name,digest in result['files'].items():assert sha(out/name)==digest
        for name in ['experiment.py','cpu_runner.py','study.py','protocol.json','input.npz']:assert sha(out/name)==sha(HERE/name)
        result['passed']=True;save('collection-verification.json',result);sub['status']='collected_hash_verified';save('submission.json',sub);print('COLLECTED_VERIFIED',len(result['files']))
    else:raise ValueError(mode)
if __name__=='__main__':main()
