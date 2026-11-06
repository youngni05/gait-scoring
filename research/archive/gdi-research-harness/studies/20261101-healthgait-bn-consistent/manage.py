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
DATA=Path('D:/gait-research-data/D_P/healthgait-bn-consistent-20261101')
BUNDLE=DATA/'bundle'
REMOTE='/abr/RESEARCH_USER/healthgait-bn-consistent-20261101'
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
    assert r['thread_id']==OWNER and r['automation_id']=='healthgait'
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
    archive=DATA/'healthgait-bn-consistent-20261101.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in BUNDLE.iterdir():
            if p.is_file():z.write(p,'healthgait-bn-consistent-20261101/'+p.name)
    digest=sha(archive)
    run(SCP+[str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/healthgait-bn-consistent-20261101.zip'],timeout=1200)
    assert ssh('sha256sum /abr/RESEARCH_USER/healthgait-bn-consistent-20261101.zip').split()[0]==digest
    source="""import pathlib,zipfile,json
p=pathlib.Path('/abr/RESEARCH_USER/healthgait-bn-consistent-20261101')
assert not p.exists(), 'Refuse overwrite of an existing remote experiment'
with zipfile.ZipFile('/abr/RESEARCH_USER/healthgait-bn-consistent-20261101.zip') as z:z.extractall('/abr/RESEARCH_USER')
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


def pilot_review():
    s=submissions();assert 'pilot' in s['stages']
    if not completed(s['stages']['pilot']['jobs']):return {'state':'pilot_pending'}
    result=json.loads(ssh('cat '+REMOTE+'/runs/pilot/summary.json'))
    assert result['passed'] and result['cuda'] and result['source']['optimizer_steps']>0
    assert set(result['targets'])=={'scratch','gdi','health'}
    assert all(v['optimizer_steps']>0 and v['reload_max_error']<=1e-6 for v in result['targets'].values())
    assert result['source']['reload_max_error']<=1e-6
    p=subprocess.run([sys.executable,'-B',str(HERE/'verify_pilot.py')],env=environment(),capture_output=True,text=True,encoding='utf8',timeout=600)
    if p.returncode:raise RuntimeError(p.stderr[-5000:]+p.stdout[-2000:])
    independent=load(HERE/'pilot-cpu-verification.json');assert independent['passed']
    save(HERE/'pilot-verification.json',dict(passed=True,verified_at=now(),result=result,independent=independent,scope='Actual CUDA source and three target fits, independent CPU BN and selected-validation replay; not final performance'))
    return {'state':'pilot_verified',**result}


def submit(stage,round_no):
    assert round_no==0, 'Only one fixed BN comparison is authorized'
    prepare_registry();r=registry();s=submissions()
    key=stage if stage in ('pilot','source') else 'round%d'%round_no
    expected_workers=2 if stage=='train' else 1
    if key in s['stages'] and len(s['stages'][key]['workers'])==expected_workers:
        return {'already_submitted':True,**s['stages'][key]}
    upload()
    if stage=='pilot':
        commands=[('pilot','sbatch --parsable --job-name=hgbn-pilot --time=00:30:00 submit.slurm pilot')]
    elif stage=='source':
        assert (HERE/'pilot-verification.json').exists()
        commands=[('source','sbatch --parsable --job-name=hgbn-source --time=04:00:00 submit.slurm source')]
    else:
        assert 'source' in s['stages'] and completed(s['stages']['source']['jobs'])
        assert load(HERE/'source-verification.json')['passed'], 'Source model must be independently verified before transfer'
        assert json.loads(ssh('cat '+REMOTE+'/runs/source/refit/fit_complete.json'))['optimizer_steps']>0
        if round_no:
            assert (HERE/('analysis/round%d/verification.json'%(round_no-1))).exists()
            assert (HERE/('improvement-r%d.json'%round_no)).exists(),'Write findings and next change before submission'
        commands=[('fold012','sbatch --parsable --job-name=hgbn-r%d-a --time=24:00:00 submit.slurm train %d 0,1,2'%(round_no,round_no)),('fold34','sbatch --parsable --job-name=hgbn-r%d-b --time=24:00:00 submit.slurm train %d 3,4'%(round_no,round_no))]
    # Record each real job immediately, before submitting another worker.
    rec=s['stages'].setdefault(key,dict(submitted_at=now(),jobs=[],workers={},remote=REMOTE))
    for name,cmd in commands:
        if name in rec['workers']:continue
        token=key+'-'+name
        code="""import os,pathlib,json,subprocess
os.chdir(%r)
result=pathlib.Path('control/submitted-'+%r+'.json')
result.parent.mkdir(exist_ok=True)
lock=result.with_suffix('.lock')
if result.exists():
 print(result.read_text())
else:
 with lock.open('x') as f:f.write('submission intent; inspect scheduler if result absent')
 p=subprocess.run(%r,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 if p.returncode:raise RuntimeError(p.stderr)
 job=p.stdout.strip().split(';')[0]
 assert job.isdigit(),p.stdout
 value={'job_id':job,'token':%r}
 result.write_text(json.dumps(value))
 print(json.dumps(value))
""" % (REMOTE,token,cmd.split(),token)
        response=json.loads(remote_python(code));job=response['job_id']
        assert job.isdigit(), 'Ambiguous submission; inspect server before retry'
        if job not in rec['jobs']:rec['jobs'].append(job)
        rec['workers'][name]=job;save(HERE/'submission.json',s)
        # Re-read latest own registry so another heartbeat update is not lost.
        r=registry();entry=next((e for e in r['experiments'] if e['id']=='bn-'+key),None)
        if entry is None:
            entry=dict(id='bn-'+key,kind='pilot' if stage=='pilot' else ('source_pretraining' if stage=='source' else 'full_training'),local='gdi-research-harness/studies/20261101-healthgait-bn-consistent',remote=REMOTE,jobs=[],review_state='pending',reported_digest=None,preserve_jobs=['32844'],summary_path='runs/pilot/summary.json' if stage=='pilot' else ('runs/source/summary.json' if stage=='source' else 'runs/round%d/summary.json'%round_no),run_root='runs/'+key,curve_glob='**/curve.json',outer_complete_glob='fold*/complete.json',audit='Use protocol.json and verify.py; expected full target 5fold*3arms*3seeds final45 models plus nested fits; source pretraining separate',received=str(DATA/'received'/key))
            if stage=='train':entry['marker_path']='runs/round%d/summary_complete.json'%round_no
            r['experiments'].append(entry)
        if job not in entry['jobs']:entry['jobs'].append(job)
        save(REGISTRY,r)
    save(HERE/'submission.json',s)
    campaign=load(HERE/'control/campaign.json')
    campaign.update(state=key+'_submitted',current_round=round_no,updated_at=now())
    save(HERE/'control/campaign.json',campaign)
    return {'stage':key,**rec}


def retrieve(round_no):
    key='round%d'%round_no;s=submissions();assert key in s['stages']
    if not completed(s['stages'][key]['jobs']):return {'state':'training_pending','jobs':s['stages'][key]['jobs']}
    if (HERE/('analysis/'+key+'/verification.json')).exists():return {'state':'already_verified','path':str(HERE/('analysis/'+key+'/verification.json'))}
    name=key+'-review.tar.gz'
    digest=ssh('cd '+REMOTE+' && tar -czf runs/'+name+' *.py launch.sh submit.slurm MODEL_LICENSE protocol.json manifest.json source.npz target.npz gdi.pt runs/source runs/'+key+' control logs && sha256sum runs/'+name,timeout=1200).split()[0]
    dest=DATA/'received'/key;dest.mkdir(parents=True,exist_ok=True)
    archive=dest/name;run(SCP+['RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/runs/'+name,str(archive)],timeout=1200)
    assert sha(archive)==digest
    extracted=dest/'originals';extracted.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf:
            p=(extracted/member.name).resolve()
            assert extracted.resolve() in p.parents and not(member.issym() or member.islnk()),'Unsafe tar member'
            if member.isdir():p.mkdir(parents=True,exist_ok=True)
            elif member.isfile():
                p.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
            else:raise ValueError('Unsupported tar type')
    out=HERE/'analysis'/key
    p=subprocess.run([sys.executable,'-B',str(HERE/'verify.py'),'--root',str(extracted),'--round',str(round_no),'--out',str(out)],env=environment(),capture_output=True,text=True,encoding='utf8',timeout=1800)
    if p.returncode:raise RuntimeError(p.stderr[-5000:]+p.stdout[-2000:])
    save(out/'receipt.json',dict(server_archive_sha256=digest,local_archive_sha256=sha(archive),archive=str(archive),originals=str(extracted),verified_at=now()))
    return {'state':'verified','verification':load(out/'verification.json')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['upload','submit','pilot-review','status','retrieve']);ap.add_argument('--stage',choices=['pilot','source','train']);ap.add_argument('--round',type=int,default=0);ap.add_argument('--owner-thread',required=True)
    a=ap.parse_args();assert a.owner_thread==OWNER
    if a.action=='upload':result=upload()
    elif a.action=='submit':result=submit(a.stage,a.round)
    elif a.action=='pilot-review':result=pilot_review()
    elif a.action=='retrieve':result=retrieve(a.round)
    else:
        s=submissions();jobs=[j for e in s['stages'].values() for j in e['jobs']]
        result=dict(submissions=s,accounting=job_status(jobs) if jobs else '',queue=ssh('squeue -h -u RESEARCH_USER -o "%i|%T|%j|%R"'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
