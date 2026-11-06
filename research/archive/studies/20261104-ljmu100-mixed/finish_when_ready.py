"""Audit retrieved folds during this foreground task and assemble the requested comparison."""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
SSH=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe');SCP=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe')
OPTS=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
HOST='RESEARCH_USER@PRIVATE_HOST';REMOTE='/abr/RESEARCH_USER/ljmu100-mixed-20261104'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def ssh(command):return subprocess.check_output([SSH,'-p','10000',*OPTS,HOST,command],timeout=60,text=True)
env=dict(os.environ,PYTHONPATH='D:/Projects/D_P/outputs/motionbert-smoke-20261015/deps')
audited=set()
while len(audited)<15:
    for family in ['CNN','STT','bilateral']:
        for fold in range(5):
            key=family+'-fold'+str(fold);receipt=HERE/'watch'/(key+'-receipt.json');audit=HERE/'audits'/(family+'-'+str(fold)+'.json')
            if key in audited or not receipt.exists():continue
            rec=read(receipt);assert sha(Path(rec['archive']))==rec['sha256']
            if not audit.exists():subprocess.run([sys.executable,'-u',str(HERE/'audit_results.py'),'--family',family,'--folds',str(fold)],env=env,check=True)
            assert read(audit)['passed'];audited.add(key)
            print(json.dumps({'event':'fold_audited','family':family,'fold':fold,'completed':len(audited)}),flush=True)
    snap=read(HERE/'watch/latest.json')
    states=[line.split('|')[1] for line in snap['sacct'].splitlines() if line.split('|')[0] in ['35866','35867']]
    if any(s not in ['RUNNING','PENDING','COMPLETED','COMPLETING','CONFIGURING'] for s in states):raise RuntimeError('Main job requires attention: '+str(states))
    if len(audited)<15:time.sleep(60)

while True:
    states=ssh('sacct -j 35866,35867,35869 --format=JobID,State,ExitCode --parsable2 --noheader')
    rows=[v.split('|') for v in states.splitlines() if v.split('|')[0] in ['35866','35867','35869']]
    if len(rows)==3 and all(v[1]=='COMPLETED' and v[2]=='0:0' for v in rows):break
    if any(v[1] not in ['RUNNING','PENDING','COMPLETED','COMPLETING','CONFIGURING'] for v in rows):raise RuntimeError(states)
    time.sleep(60)
digest=ssh('cd '+REMOTE+' && tar -czf final-verification.tar.gz verification-data/runs logs/35866.log logs/35867.log logs/35869.log && sha256sum final-verification.tar.gz').split()[0]
target=BIG/'received/final-verification.tar.gz'
subprocess.run([SCP,'-P','10000',*OPTS,HOST+':'+REMOTE+'/final-verification.tar.gz',str(target)],check=True,timeout=120)
assert sha(target)==digest
subprocess.run(['tar','-xzf',str(target),'-C',str(BIG/'received')],check=True)
checks=[]
for family in ['CNN','STT','bilateral']:
    for f in range(5):
        path=BIG/'received/verification-data/runs/baseline'/family/f'fold{f}'/'predictions.npz'
        assert sha(path)==read(path.parent/'complete.json')['prediction_sha256']
        with np.load(path) as rerun,np.load(BIG/'received/data/runs/baseline'/family/f'fold{f}'/'predictions.npz') as first:
            assert np.array_equal(rerun['row_index'],first['row_index'])
            errors={k:float(np.max(np.abs(rerun[k]-first[k]))) for k in ['refit','refit_raw','bag']}
            assert max(errors.values())<.001,(family,f,errors)
            checks.append({'family':family,'fold':f,'errors':errors})
save(HERE/'baseline-gpu-verified.json',{'passed':True,'job':'35869','folds':checks,'archive_sha256':digest,
     'cross_device_exception_preserved':'CPU replay CNN fold3 bag max error0.0012155359 exceeds0.001. Same-environment replay does not erase this limitation.'})
assert read(HERE/'audits/RF-0_1_2_3_4.json')['passed']
subprocess.run([sys.executable,str(HERE/'compare.py')],check=True)
save(HERE/'completion-verification.json',{'passed':True,'mixed_neural_fits':75,'mixed_neural_folds':15,'RF_fits':170,
     'baseline_checkpoints':54,'data_sha256':sha(BIG/'data.npz'),'comparison_sha256':sha(HERE/'comparison.json'),
     'slurm':states,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'baseline_cross_device_exception':'CNN fold3 bag; GPU rerun passes separately; CPU maximum difference0.0012155359 retained in audit.'})
print('ALL_RESULTS_VERIFIED_AND_COMPARED',flush=True)
