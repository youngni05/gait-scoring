"""Dedicated transport and Slurm submission; never touches other experiments."""
import sys,json,subprocess,hashlib,tarfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];REMOTE='/abr/RESEARCH_USER/care-pd-straight-20261031'
sys.path.insert(0,str(BASE));from remote import command,AUTH,HOST
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    mode=sys.argv[1];prefix='import pathlib,json,subprocess,hashlib,tarfile\np=pathlib.Path('+repr(REMOTE)+')\n'
    if mode in ['pilot','resume-pilot']:
        assert not (HERE/'submission.json').exists()
        bundle=HERE/'bundle.tar.gz';digest=sha(bundle)
        if mode=='pilot':subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[str(bundle),HOST+':'+REMOTE+'.tar.gz'],check=True)
        source=prefix+"z=p.with_suffix('.tar.gz')\nassert hashlib.sha256(z.read_bytes()).hexdigest()=="+repr(digest)+"\nassert not p.exists()\np.mkdir()\ntarfile.open(z).extractall(p)\n(p/'logs').mkdir()\nr=subprocess.check_output(['sbatch','--exclude=DIS04,DIS05,DIS06','--time=00:10:00','submit.slurm','pilot'],cwd=p,text=True)\nprint(json.dumps(dict(job=r.strip().split()[-1],bundle_sha256="+repr(digest)+")))"
        if mode=='resume-pilot':
            fixed=(HERE/'submit.slurm').read_bytes().replace(b'\r\n',b'\n');(HERE/'submit.slurm').write_bytes(fixed)
            source=prefix+"assert len(list(p.glob('clips/*/*.jpg')))==3360\nassert not (p/'pilot').exists()\ns=subprocess.check_output(['sacct','-u','RESEARCH_USER','--name=care-straight','--format=JobID','-P','-n'],text=True);assert not s.strip()\nf=p/'submit.slurm';f.write_bytes(f.read_bytes().replace(b'\\r\\n',b'\\n'));assert hashlib.sha256(f.read_bytes()).hexdigest()=="+repr(sha(HERE/'submit.slurm'))+"\nr=subprocess.check_output(['sbatch','--exclude=DIS04,DIS05,DIS06','--time=00:10:00','submit.slurm','pilot'],cwd=p,text=True)\nprint(json.dumps(dict(job=r.strip().split()[-1],bundle_sha256="+repr(digest)+")))"
        result=json.loads(command(source,timeout=120));submission=dict(remote=REMOTE,pilot_job=result['job'],status='pilot_submitted',bundle_sha256=digest,timestamp=datetime.now(timezone.utc).isoformat(),transport_note='Extraction outlived SSH response timeout; original sbatch rejected CRLF before creating any job. LF conversion only, verified no prior job, then submitted.' if mode=='resume-pilot' else None);save(HERE/'submission.json',submission)
        regpath=BASE/'watch/registry.json';reg=json.loads(regpath.read_text());assert reg['thread_id']=='01a18b70-7da2-7430-a223-d4b314940cb3' and reg['automation_id']=='care-pd'
        assert not any(r['id']=='care-pd-straight' for r in reg['experiments'])
        reg['experiments'].append(dict(id='care-pd-straight',kind='video_inference',local=str(HERE),remote=REMOTE,jobs=[result['job']],pilot_jobs=[result['job']],preserve_allocations=['32844'],expected_fits=0,expected_outer_folds=6,expected_final_models=0,expected_outputs=56,expected_frames=3360,review_state='pending',reported_digest=None,run_root='main',marker_path='main/complete.json',summary_path='main/progress.json',audit='Read straight-followup/protocol.json and submission.json. Pilot is16frames, not final. Main must exit0 with56clips3360frames. remote_run.py collect verifies archive/file/code hashes; analyze.py care and movi, then verify.py. Frozen RF128 and698,30models each.14people paired straight/turn2sec PG1/PG2; CARE matching50frames25Hz with held-out folds. No MoVi gait labels. No automatic submissions; preserve all other jobs.'))
        save(regpath,reg);print(json.dumps(result))
    elif mode=='status':
        sub=json.loads((HERE/'submission.json').read_text());jobs=[sub[k] for k in ['pilot_job','main_job'] if k in sub]
        src=prefix+"print(subprocess.check_output(['sacct','-j',"+repr(','.join(jobs))+",'--format=JobID,State,ExitCode,Elapsed,NodeList','-P'],text=True))\nfor mode in ['pilot','main']:\n for name in ['complete.json','progress.json']:\n  f=p/mode/name\n  if f.exists():\n   d=json.loads(f.read_text());print(mode,name,json.dumps({k:d[k] for k in ['passed','frames','completed','expected','repeat_max_error','peak_gpu_gib'] if k in d}))\nfor f in (p/'logs').glob('*.log'):print(f.name,f.read_text()[-500:])"
        result=command(src);(HERE/'status.txt').write_text(result);print(result)
    elif mode=='main':
        sub=json.loads((HERE/'submission.json').read_text());assert 'main_job' not in sub
        source=prefix+"d=json.loads((p/'pilot/complete.json').read_text());assert d['passed'] and d['frames']==16 and d['repeat_max_error']<=1e-6\ns=subprocess.check_output(['sacct','-j',"+repr(sub['pilot_job'])+",'--format=JobID,State,ExitCode','-P','-n'],text=True);assert "+repr(sub['pilot_job']+'|COMPLETED|0:0')+" in s\nassert not (p/'main').exists()\nr=subprocess.check_output(['sbatch','--exclude=DIS04,DIS05,DIS06','--time=00:30:00','submit.slurm','main'],cwd=p,text=True)\nprint(json.dumps(dict(job=r.strip().split()[-1])))"
        result=json.loads(command(source));sub.update(main_job=result['job'],status='main_submitted');save(HERE/'submission.json',sub)
        regpath=BASE/'watch/registry.json';reg=json.loads(regpath.read_text());assert reg['thread_id']=='01a18b70-7da2-7430-a223-d4b314940cb3';entry=next(r for r in reg['experiments'] if r['id']=='care-pd-straight');entry['jobs']=[result['job']];save(regpath,reg);print(json.dumps(result))
    elif mode=='collect':
        sub=json.loads((HERE/'submission.json').read_text())
        src=prefix+"d=json.loads((p/'main/complete.json').read_text());assert d['passed'] and d['frames']==3360 and len(d['outputs'])==56\ns=subprocess.check_output(['sacct','-j',"+repr(sub['main_job'])+",'--format=JobID,State,ExitCode,Elapsed,NodeList','-P','-n'],text=True);assert "+repr(sub['main_job']+'|COMPLETED|0:0')+" in s\n(p/'slurm-final.txt').write_text(s)\nfiles=[f for root in ['main','pilot','logs'] for f in (p/root).rglob('*') if f.is_file()]+[p/n for n in ['infer.py','protocol.json','clips.json','submit.slurm','slurm-final.txt']]\nz=p/'result.tar.gz'\nwith tarfile.open(z,'w:gz') as t:\n for f in files:t.add(f,arcname=str(f.relative_to(p)))\nprint(json.dumps(dict(archive_sha256=hashlib.sha256(z.read_bytes()).hexdigest(),files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})))"
        result=json.loads(command(src,timeout=120));dest=HERE/'result.tar.gz';subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[HOST+':'+REMOTE+'/result.tar.gz',str(dest)],check=True);assert sha(dest)==result['archive_sha256']
        out=HERE/'received';out.mkdir(exist_ok=True)
        with tarfile.open(dest) as t:
            assert all((out/m.name).resolve().is_relative_to(out.resolve()) for m in t.getmembers());t.extractall(out,filter='data')
        for name,digest in result['files'].items():assert sha(out/name)==digest
        for name in ['infer.py','protocol.json','clips.json','submit.slurm']:assert sha(HERE/name)==sha(out/name)
        result['passed']=True;save(HERE/'collection-verification.json',result);sub['status']='collected_hash_verified';save(HERE/'submission.json',sub);print(json.dumps(dict(passed=True,files=len(result['files']),archive_sha256=result['archive_sha256'])))
    else:raise ValueError(mode)
if __name__=='__main__':main()
