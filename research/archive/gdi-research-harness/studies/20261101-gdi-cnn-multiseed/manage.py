"""Deploy, submit held jobs, receive and approve only this chat's experiment."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,shlex,subprocess,sys,tarfile
H=Path(__file__).resolve().parent;WORK=H.parents[2]
OWNER='01a19527-6510-7a70-874d-3e58f91b1383';AUTOMATION='gdi-cnn'
REMOTE='/abr/RESEARCH_USER/gdi-cnn-multiseed-20261101'
DATA=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-gdi-cnn-multiseed')
INPUT=H.parent/'20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
PY=Path('C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
SEEDS=(20261101,20261102)
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+['RESEARCH_USER@PRIVATE_HOST']
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');q.replace(p)
def command(args,source=None,timeout=120):
    r=subprocess.run([str(x) for x in args],input=source,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if r.returncode:raise RuntimeError(r.stderr+'\n'+r.stdout)
    return r.stdout
def remote(source):return command(SSH+['/usr/bin/python3.8 -'],"ROOT="+repr(REMOTE)+'\n'+source)
def local_env():
    e=dict(os.environ);e['GDI_DATA_PATH']=str(INPUT);e['PYTHONPATH']=';'.join(str(WORK/p) for p in ['outputs/motionbert-smoke-20261015/deps','outputs/gdi-ridge-experiment-20261016/deps']);return e
def status():
    q=remote('''from pathlib import Path
import json,subprocess
root=Path(ROOT);out={'queue':subprocess.check_output(['squeue','-h','-u','RESEARCH_USER','-o','%i|%T|%j|%M|%L|%R'],text=True),'experiments':{}}
for seed in [20261101,20261102]:
 p=root/('seed%d'%seed);s=p/'control/worker-primary.json'
 out['experiments'][str(seed)]={'worker':json.loads(s.read_text()) if s.exists() else None,'fits':len(list((p/'runs/main').rglob('fit_complete.json'))),'outer_folds':len(list((p/'runs/main').glob('fold*/complete.json'))),'complete':(p/'runs/summary_complete.json').exists(),'pilot_ready':(p/'runs/pilot/summary.json').exists(),'main_gate':(p/'control/approve-main.json').exists()}
print(json.dumps(out))
''');v=json.loads(q);v['checked_at']=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat();save(H/'watch/status-latest.json',v);print(json.dumps(v));return v
def deploy():
    archive=H/'bundle.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for seed in SEEDS:
            b=H/('seed%d'%seed)
            for p in sorted(b.iterdir()):
                if p.is_file():tf.add(p,arcname='seed%d/'%seed+p.name)
        tf.add(H/'protocol.json',arcname='protocol.json')
    digest=sha(archive)
    r=remote("from pathlib import Path\nimport json\np=Path(ROOT);print(json.dumps({'exists':p.exists(),'submitted':list(str(x) for x in p.glob('seed*/control/submission.json'))}))")
    assert not json.loads(r)['exists'],'Existing remote folder: inspect, do not redeploy'
    command(SCP+[str(archive),'RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/gdi-cnn-multiseed-20261101.tar.gz'])
    source='''from pathlib import Path
import hashlib,json,tarfile,shutil
root=Path(ROOT);assert not root.exists();archive=root.with_suffix('.tar.gz');assert hashlib.sha256(archive.read_bytes()).hexdigest()==DIGEST
root.mkdir()
with tarfile.open(str(archive)) as tf:
 for m in tf.getmembers():assert not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in Path(m.name).parts
 tf.extractall(str(root))
data=Path('/abr/RESEARCH_USER/gdi-cnn-multiwindow-20261030/data.npz');assert hashlib.sha256(data.read_bytes()).hexdigest()==INPUTSHA
shutil.copyfile(str(data),str(root/'data.npz'))
for seed in [20261101,20261102]:
 p=root/('seed%d'%seed)
 for name,digest in json.loads((p/'manifest.json').read_text()).items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
 (p/'logs').mkdir();(p/'control').mkdir()
print(json.dumps({'deployed':str(root),'data_sha256':INPUTSHA,'archive_sha256':DIGEST}))
'''
    out=json.loads(remote('DIGEST='+repr(digest)+'\nINPUTSHA='+repr(sha(INPUT))+'\n'+source));save(H/'deployment.json',out);print(json.dumps(out))
def submit():
    assert read(H/'deployment.json')['data_sha256']==sha(INPUT)
    entries=[]
    for seed in SEEDS:
        proof=read(H/('checks/local-verification-seed%d.json'%seed));assert proof['passed'] and proof['manifest_sha256']==sha(H/('seed%d/manifest.json'%seed))
        code='''from pathlib import Path
import json,subprocess
p=Path(ROOT)/('seed%d'%SEED);record=p/'control/submission.json'
if record.exists():print(record.read_text())
else:
 name='gdi-seed%d'%SEED
 existing=subprocess.check_output(['squeue','-h','-u','RESEARCH_USER','-n',name,'-o','%i'],text=True).strip();assert not existing,'Ambiguous existing submission: '+existing
 args=['sbatch','--hold','--parsable','--job-name='+name,'--account=iucp','--partition=p02','--gres=gpu:1','--cpus-per-task=4','--mem=12G','--time=08:00:00','--exclude=DIS04,DIS05,DIS06','--chdir='+str(p),'--output='+str(p/'logs/worker-%j.log'),'--wrap=export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; exec /usr/bin/python3.8 -u -B support.py worker']
 result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert result.returncode==0,result.stderr
 job=result.stdout.strip().split(';')[0];assert job.isdigit(),result.stdout
 v={'job_id':job,'seed':SEED,'held':True,'remote':str(p),'args':args};record.write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
        v=json.loads(remote('SEED='+repr(seed)+'\n'+code));entries.append(v);save(H/'submission.json',dict(owner_thread=OWNER,automation_id=AUTOMATION,jobs=entries))
    reg=dict(version=1,thread_id=OWNER,automation_id=AUTOMATION,experiments=[dict(id='cnn-seed%d'%v['seed'],kind='full_training',local=str((H/('seed%d'%v['seed'])).relative_to(WORK)).replace('\\','/'),remote=v['remote'],jobs=[v['job_id']],expected_fits=25,expected_outer_folds=5,preserve_allocations=[],review_state='pending',reported_digest=None,audit='Use README/manage.py fetch pilot/full, review per seed and compare.py. Verify every epoch, original batch order, inner-only epoch selection, training-only normalization/OLS and five final CPU64 replays. Never weaken 1e-10 coefficient/metric or 1e-4 inference tolerance. Full 50 fits and 10 outer folds plus predeclared three-seed equal average required before final reporting.',baselines={'best_blend':6.515816569178735}) for v in entries])
    save(H/'watch/registry.json',reg);print(json.dumps(dict(jobs=entries,registry=str(H/'watch/registry.json'))))
def release():
    r=read(H/'watch/registry.json');assert r['thread_id']==OWNER and r['automation_id']==AUTOMATION
    for e in r['experiments']:
        job=e['jobs'][0];print(command(SSH+['scontrol release '+shlex.quote(job)]).strip())
    save(H/'release.json',dict(jobs=[e['jobs'][0] for e in r['experiments']],released_at=datetime.datetime.now().isoformat()));status()
def fetch(kind,seeds=SEEDS):
    receipt_path=H/('checks/%s-receipt.json'%kind)
    outputs=read(receipt_path)['seeds'] if receipt_path.exists() else []
    for seed in seeds:
        source='''from pathlib import Path
import hashlib,json,tarfile
p=Path(ROOT)/('seed%d'%SEED);assert (p/('runs/pilot/summary.json' if KIND=='pilot' else 'runs/summary_complete.json')).exists()
out=p/'recovery';out.mkdir(exist_ok=True);archive=out/(KIND+'.tar.gz')
if not archive.exists():
 names=list(json.loads((p/'manifest.json').read_text()).keys())+['manifest.json','runs/pilot','checks','control']
 if KIND=='full':names+=['runs/main','runs/summary_complete.json']
 with tarfile.open(str(archive),'w:gz',compresslevel=1) as tf:
  for name in names:tf.add(str(p/name),arcname=name)
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size}))
'''
        proof=json.loads(remote('SEED='+repr(seed)+'\nKIND='+repr(kind)+'\n'+source));dest=DATA/('received/%s/seed%d'%(kind,seed));dest.mkdir(parents=True,exist_ok=True);archive=dest/(kind+'.tar.gz')
        if not archive.exists() or sha(archive)!=proof['sha256']:command(SCP+['RESEARCH_USER@PRIVATE_HOST:'+proof['archive'],str(archive)],timeout=300)
        assert sha(archive)==proof['sha256']
        extracted=dest/'unpacked'
        if not extracted.exists():
            extracted.mkdir()
            with tarfile.open(archive) as tf:
                for m in tf.getmembers():assert not m.issym() and not m.islnk() and (extracted/m.name).resolve().is_relative_to(extracted.resolve())
                tf.extractall(extracted,filter='data')
        for name,digest in read(extracted/'manifest.json').items():assert sha(extracted/name)==digest
        assert sha(extracted/'manifest.json')==sha(H/('seed%d/manifest.json'%seed))
        out=H/('checks/%s-review-seed%d.json'%(kind,seed));out.parent.mkdir(exist_ok=True)
        review=subprocess.run([str(PY),'-B','support.py','pilot-review' if kind=='pilot' else 'review','--out',str(out)],cwd=str(extracted),env=local_env(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=H/('checks/%s-local-review-seed%d.log'%(kind,seed));log.write_text(review.stdout,encoding='utf-8')
        if review.returncode:
            failure=H/('checks/local-strict-failure-seed%d.json'%seed)
            if not failure.exists():save(failure,dict(passed=False,exit_code=review.returncode,log_path=str(log),log_sha256=sha(log),tolerance_changed=False))
            # Keep the local failure. The original coefficients/metrics tolerance is
            # rerun unchanged in the source numeric environment; it is never patched.
            assert kind=='full' and "assert_allclose(coef,row['calibration']" in review.stdout and 'atol=1e-10' in review.stdout,review.stdout[-4000:]
            from source_audit import collect,finish_local_replay
            source_proof=collect(seed)
            # Verify every extracted byte still matches the SHA-matched archive.
            file_count=0
            with tarfile.open(archive,'r|gz') as tf:
                for member in tf:
                    if member.isfile():
                        expected=hashlib.sha256(tf.extractfile(member).read()).hexdigest();assert sha(extracted/member.name)==expected,member.name;file_count+=1
            local_proof=finish_local_replay(seed,extracted)
            save(out,dict(passed=True,scope='Original unchanged full strict audit passed in source environment plus supplemental local CPU64 replay; local original strict failure remains failed.',local_original_strict_passed=False,source_original_strict_passed=True,local_cpu64_replay_passed=True,source_review_path=str(source_proof),source_review_sha256=sha(source_proof),local_replay_path=str(local_proof),local_replay_sha256=sha(local_proof),local_strict_failure_path=str(failure),local_strict_failure_sha256=sha(failure),archive_extracted_files_rehashed=file_count,tolerance_changed=False))
        proof.update(passed=read(out)['passed'],extracted=str(extracted),server_local_sha256_matched=True,seed=seed,review=str(out),review_sha256=sha(out))
        outputs=sorted([v for v in outputs if v['seed']!=seed]+[proof],key=lambda v:v['seed'])
        save(receipt_path,dict(passed=len(outputs)==len(SEEDS) and all(v['passed'] for v in outputs),seeds=outputs))
    print(json.dumps(outputs))
def approve():
    receipt=read(H/'checks/pilot-receipt.json');assert receipt['passed'] and len(receipt['seeds'])==2
    states=status();uuids=[v['worker']['gpu_uuid'] for v in states['experiments'].values()];assert len(set(uuids))==2
    for v in receipt['seeds']:
        seed=v['seed'];proof=read(v['review']);p=Path(v['extracted']);summary=read(p/'runs/pilot/summary.json')
        assert proof['passed'] and proof['maximum_prediction_error']<1e-4 and summary['train_people']==416 and summary['validation_people']==138 and summary['epochs']==3 and not summary['outer_evaluation_used']
        gpu=read(p/'checks/gpu/verification.json');assert gpu['passed'] and gpu['device']=='cuda' and gpu['manifest_sha256']==sha(p/'manifest.json')
        for name in ['replay-gpu64.json','replay-cpu64.json']:
            check=read(p/'runs/pilot'/name);assert check['passed'] and check['maximum_prediction_error']<1e-4 and check['checkpoint_sha256']==proof['checkpoint_sha256']
        gate=dict(passed=True,manifest_sha256=sha(p/'manifest.json'),checkpoint_sha256=proof['checkpoint_sha256'],local_review_sha256=sha(v['review']))
        code='''from pathlib import Path
import hashlib,json
p=Path(ROOT)/('seed%d'%SEED);assert hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest()==GATE['manifest_sha256'];assert hashlib.sha256((p/'runs/pilot/fit/latest.pt').read_bytes()).hexdigest()==GATE['checkpoint_sha256'];q=p/'control/approve-main.json';assert not q.exists() or json.loads(q.read_text())==GATE;q.write_text(json.dumps(GATE,indent=2));print(json.dumps({'seed':SEED,'main_approved':True}))
'''
        print(remote('SEED='+repr(seed)+'\nGATE='+repr(gate)+'\n'+code))
    save(H/'main-approval.json',dict(passed=True,pilot_receipt_sha256=sha(H/'checks/pilot-receipt.json'),approved_seeds=list(SEEDS)))
def compare():
    subprocess.run([str(PY),'-B',str(H/'compare.py')],cwd=str(H),env=local_env(),check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['status','deploy','submit','release','fetch','approve','compare']);p.add_argument('--kind',choices=['pilot','full']);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args()
    if a.command=='fetch':assert a.kind;fetch(a.kind,SEEDS if a.seed is None else [a.seed])
    else:globals()[a.command]()
