"""Own-experiment submission/retrieval only. Never cancels jobs or touches other chats."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
DATA=Path('D:/gait-research-data/D_P/sppb-sample-size-cnn-20261101')
BUNDLE=DATA/'bundle'
REMOTE='/abr/RESEARCH_USER/sppb-sample-size-cnn-20261101'
OWNER='01a1929b-43f3-7bc1-b921-b17bc551d7fe'
REGISTRY=HERE.parent/'20261101-healthgait-transfer/watch/registry.json'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+['RESEARCH_USER@PRIVATE_HOST']
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def run(cmd,timeout=120,input=None):
    p=subprocess.run(cmd,input=input,capture_output=True,text=True,encoding='utf8',timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-4000:] or p.stdout[-4000:])
    return p.stdout.strip()
def ssh(command,timeout=120):return run(SSH+[command],timeout)
def remote_python(source,timeout=120):return run(SSH+['/usr/bin/python3.8 -'],timeout,source)
def now():return datetime.datetime.now().astimezone().isoformat()
def environment():
    env=dict(os.environ)
    env['PYTHONPATH']=';'.join([str(WORKSPACE/'outputs/motionbert-smoke-20261015/deps'),str(WORKSPACE/'outputs/gdi-ridge-experiment-20261016/deps')])
    return env


def registry():
    p=REGISTRY
    r=load(p)
    assert r['thread_id']==OWNER and r['automation_id']==load(HERE/'control/campaign.json')['automation_id']
    return r


def prepare_registry():
    assert REGISTRY.exists(), 'Reuse the existing owning thread registry'
    registry()


def upload():
    receipt=HERE/'upload.json'
    if receipt.exists():
        r=load(receipt)
        assert ssh('sha256sum '+REMOTE+'/manifest.json').split()[0]==r['manifest_sha256']
        return r
    for n,h in load(BUNDLE/'manifest.json').items():assert sha(BUNDLE/n)==h,n
    archive=DATA/'sppb-sample-size-cnn-20261101.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in BUNDLE.iterdir():
            if p.is_file():z.write(p,'sppb-sample-size-cnn-20261101/'+p.name)
    digest=sha(archive)
    run(SCP+[str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/sppb-sample-size-cnn-20261101.zip'],timeout=1200)
    assert ssh('sha256sum /abr/RESEARCH_USER/sppb-sample-size-cnn-20261101.zip').split()[0]==digest
    source="""import pathlib,zipfile,json
p=pathlib.Path('/abr/RESEARCH_USER/sppb-sample-size-cnn-20261101')
assert not p.exists(), 'Refuse overwrite of an existing remote experiment'
with zipfile.ZipFile('/abr/RESEARCH_USER/sppb-sample-size-cnn-20261101.zip') as z:z.extractall('/abr/RESEARCH_USER')
(p/'logs').mkdir()
print(json.dumps({'created':str(p)}))
"""
    remote_python(source)
    r=dict(uploaded_at=now(),local_archive=str(archive),archive_sha256=digest,remote=REMOTE,manifest_sha256=sha(BUNDLE/'manifest.json'))
    save(receipt,r);return r


def job_status(jobs):
    return ssh('sacct -n -P -j '+','.join(jobs)+' --format=JobIDRaw,State,ExitCode,Elapsed,NodeList')


def completed(jobs):
    lines=job_status(jobs).splitlines();states={v.split('|')[0]:v.split('|')[1:3] for v in lines}
    if not all(j in states for j in jobs):return False
    bad={j:states[j] for j in jobs if states[j][0].split()[0] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')}
    if bad:raise RuntimeError('Training failed: '+json.dumps(bad))
    return all(states[j]==['COMPLETED','0:0'] for j in jobs)


def submissions():
    return load(HERE/'submission.json') if (HERE/'submission.json').exists() else {'stages':{}}


def receive(stage):
    name=stage+'-review.tar.gz'
    digest=ssh('cd '+REMOTE+' && tar -czf runs/'+name+' *.py launch.sh submit.slurm MODEL_LICENSE protocol.json plan.json manifest.json target.npz reference-cnn.csv reference-rf.csv runs/'+stage+' control logs && sha256sum runs/'+name,timeout=1200).split()[0]
    dest=DATA/'received'/stage;dest.mkdir(parents=True,exist_ok=True)
    archive=dest/name;run(SCP+['RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/runs/'+name,str(archive)],timeout=1200)
    assert sha(archive)==digest
    extracted=dest/'originals';extracted.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            p=(extracted/member.name).resolve()
            assert extracted.resolve() in p.parents and not(member.issym() or member.islnk())
            if member.isdir():p.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                p.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
            else:raise ValueError('Unsupported tar member')
    return extracted,dict(server_archive_sha256=digest,local_archive_sha256=sha(archive),archive=str(archive),originals=str(extracted),verified_at=now())
def verification(stage,pilot=False):
    out=HERE/'analysis'/stage
    if (out/'verification.json').exists():
        value=load(out/'verification.json');assert value['passed'];return value
    extracted,receipt=receive(stage)
    cmd=[sys.executable,'-X','utf8','-B',str(HERE/'verify.py'),'--root',str(extracted),'--out',str(out)]
    if pilot:cmd.append('--pilot')
    p=subprocess.run(cmd,env=environment(),capture_output=True,text=True,encoding='utf8',timeout=2400)
    if p.returncode:raise RuntimeError(p.stderr[-5000:]+p.stdout[-2000:])
    save(out/'receipt.json',receipt);return load(out/'verification.json')
def pilot_review():
    s=submissions();assert 'pilot' in s['stages']
    if not completed(s['stages']['pilot']['jobs']):return {'state':'pilot_pending'}
    result=json.loads(ssh('cat '+REMOTE+'/runs/pilot/summary.json'))
    assert result['passed'] and result['cuda'] and len(result['fits'])==6
    assert all(f['optimizer_steps']>0 and f['reload_max_error']<=1e-6 for f in result['fits'])
    independent=verification('pilot',True);assert independent['passed']
    save(HERE/'pilot-verification.json',dict(passed=True,result=result,independent=independent,verified_at=now()))
    return dict(state='pilot_verified',**independent)
def submit(stage,round_no):
    assert round_no==0 and stage in ('pilot','train')
    prepare_registry();s=submissions();key='pilot' if stage=='pilot' else 'round0'
    count=1 if stage=='pilot' else 2
    if key in s['stages'] and len(s['stages'][key]['workers'])==count:return dict(already_submitted=True,**s['stages'][key])
    upload()
    if stage=='pilot':commands=[('pilot','sbatch --parsable --job-name=sppb-lc-pilot --time=00:30:00 submit.slurm pilot')]
    else:
        assert load(HERE/'pilot-verification.json')['passed']
        commands=[('fold012','sbatch --parsable --job-name=sppb-lc-a --time=12:00:00 submit.slurm train 0 0,1,2'),('fold34','sbatch --parsable --job-name=sppb-lc-b --time=12:00:00 submit.slurm train 0 3,4')]
    rec=s['stages'].setdefault(key,dict(submitted_at=now(),jobs=[],workers={},remote=REMOTE))
    for name,cmd in commands:
        if name in rec['workers']:continue
        token=key+'-'+name
        code="""import os,pathlib,json,subprocess
os.chdir(%r)
result=pathlib.Path('control/submitted-'+%r+'.json')
result.parent.mkdir(exist_ok=True)
if result.exists():print(result.read_text())
else:
 with result.with_suffix('.lock').open('x') as f:f.write('submission intent; inspect scheduler before retry')
 p=subprocess.run(%r,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 job=p.stdout.strip().split(';')[0]
 assert job.isdigit(),p.stdout
 v={'job_id':job,'token':%r};result.write_text(json.dumps(v));print(json.dumps(v))
"""%(REMOTE,token,cmd.split(),token)
        response=json.loads(remote_python(code));job=response['job_id'];assert job.isdigit()
        rec['workers'][name]=job
        if job not in rec['jobs']:rec['jobs'].append(job)
        save(HERE/'submission.json',s)
        r=registry();entry=next((e for e in r['experiments'] if e['id']=='lc-'+key),None)
        if entry is None:
            entry=dict(id='lc-'+key,kind='pilot' if stage=='pilot' else 'full_training',local='gdi-research-harness/studies/20261101-sppb-sample-size-cnn',remote=REMOTE,jobs=[],review_state='pending',reported_digest=None,preserve_jobs=['32844'],summary_path='runs/'+key+'/summary.json',run_root='runs/'+key,curve_glob='**/curve.json',outer_complete_glob='fold*/complete.json',received=str(DATA/'received'/key),audit='Pilot6fits;main1800CNNfits=1350inner+450final,5outerfolds50personcases. Separate local RF1050fits150final required before campaign complete.')
            if stage=='train':entry.update(marker_path='runs/round0/summary_complete.json',expected_outer_folds=5,expected_inner_fits=1350,expected_final_models=450,expected_total_fits=1800)
            r['experiments'].append(entry)
        if job not in entry['jobs']:entry['jobs'].append(job)
        save(REGISTRY,r)
    campaign=load(HERE/'control/campaign.json');campaign.update(state=key+'_submitted',updated_at=now());save(HERE/'control/campaign.json',campaign)
    return dict(stage=key,**rec)
def rf_status():
    p=HERE/'control/rf-job.json'
    if not p.exists():return dict(state='not_started')
    receipt=load(p);progress=load(DATA/'rf/progress.json') if (DATA/'rf/progress.json').exists() else {}
    pid=int(receipt['pid'])
    raw=run(['powershell','-NoProfile','-Command',"Get-CimInstance Win32_Process -Filter 'ProcessId=%d' | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"%pid])
    process=json.loads(raw) if raw else None
    alive=bool(process and str(HERE/'rf.py').replace('\\','/').lower() in str(process['CommandLine']).replace('\\','/').lower())
    return dict(state=progress.get('state','starting') if alive or progress.get('state')=='complete' else 'stopped',alive=alive,receipt=receipt,progress=progress)
def rf_start(resume=False):
    status=rf_status()
    if status['state']=='complete' or status.get('alive'):return status
    if status['state']!='not_started':
        assert resume,'RF stopped; inspect stderr/progress then explicitly rf-resume to reuse verified fits'
        lock=DATA/'rf/running.lock'
        if lock.exists():lock.unlink()
    assert load(HERE/'selfcheck.json')['passed']
    root=DATA/'rf';root.mkdir(exist_ok=True);log=root/'stdout.log';err=root/'stderr.log'
    with log.open('ab') as stdout,err.open('ab') as stderr:
        proc=subprocess.Popen([sys.executable,'-X','utf8','-B',str(HERE/'rf.py')],env=environment(),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW,cwd=str(HERE))
    receipt=dict(pid=proc.pid,started_at=now(),script_sha256=sha(HERE/'rf.py'),plan_sha256=sha(HERE/'plan.json'),stdout=str(log),stderr=str(err))
    save(HERE/'control/rf-job.json',receipt);return receipt
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['upload','submit','pilot-review','status','retrieve','rf-start','rf-status','rf-resume']);ap.add_argument('--stage',choices=['pilot','train']);ap.add_argument('--round',type=int,default=0);ap.add_argument('--owner-thread',required=True)
    a=ap.parse_args();assert a.owner_thread==OWNER;registry()
    if a.action=='upload':result=upload()
    elif a.action=='submit':result=submit(a.stage,a.round)
    elif a.action=='pilot-review':result=pilot_review()
    elif a.action=='retrieve':
        assert a.round==0
        jobs=submissions()['stages']['round0']['jobs']
        result=verification('round0') if completed(jobs) else dict(state='training_pending',jobs=jobs)
    elif a.action=='rf-status':result=rf_status()
    elif a.action in ('rf-start','rf-resume'):result=rf_start(a.action=='rf-resume')
    else:
        s=submissions();jobs=[j for e in s['stages'].values() for j in e['jobs']]
        result=dict(submissions=s,accounting=job_status(jobs) if jobs else '',queue=ssh('squeue -h -u RESEARCH_USER -o "%i|%T|%j|%R"'))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
