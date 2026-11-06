"""One snapshot and retrieval of already-completed outputs; no polling or background service."""
import datetime,hashlib,json,os,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
REMOTE='/abr/RESEARCH_USER/ljmu100-mixed-20261104';HOST='RESEARCH_USER@PRIVATE_HOST'
SSH=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe');SCP=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe')
OPTS=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def ssh(command):return subprocess.check_output([SSH,'-p','10000',*OPTS,HOST,command],text=True,timeout=120)
def retrieve(archive,paths):
    digest=ssh('cd '+REMOTE+' && tar -czf '+archive+' '+paths+' && sha256sum '+archive).split()[0]
    target=BIG/'received'/archive
    subprocess.run([SCP,'-P','10000',*OPTS,HOST+':'+REMOTE+'/'+archive,str(target)],check=True,timeout=240)
    assert sha(target)==digest
    with tarfile.open(target) as tf:tf.extractall(BIG/'received',filter='data')
    return dict(archive=str(target),sha256=digest,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
snap=json.loads(ssh('/usr/bin/python3.8 '+REMOTE+'/remote_status.py'))
snap['related_job_accounting']=ssh('sacct -j 35847,35848,35861,35866,35867,35869 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList --parsable2 --noheader')
save(HERE/'watch/latest.json',snap)
for item in snap['completed_folds']:
    family=item['family'];fold=item['fold'];assert family in ['CNN','STT','bilateral'] and fold in range(5)
    tag=family+'-fold'+str(fold);record=HERE/'watch'/(tag+'-receipt.json')
    if record.exists():
        old=json.loads(record.read_text());p=Path(old['archive'])
        if p.exists() and sha(p)==old['sha256']:continue
    receipt=retrieve(tag+'-results.tar.gz','data/runs/mixed/'+family+'/fold'+str(fold))
    save(record,dict(receipt,**item));print('RECEIVED',tag,flush=True)
replay=[r.split('|') for r in snap['related_job_accounting'].splitlines() if r.startswith('35869|')]
if replay and replay[0][2:4]==['COMPLETED','0:0'] and not (HERE/'watch/baseline-gpu-receipt.json').exists():
    receipt=retrieve('baseline-gpu-replay-results.tar.gz','verification-data/runs logs/35869.log')
    save(HERE/'watch/baseline-gpu-receipt.json',receipt);print('RECEIVED baseline_gpu_replay',flush=True)
print(json.dumps({'utc':snap['utc'],'completed_fits':snap['completed_fits'],'completed_folds':len(snap['completed_folds']),
                  'sacct':snap['sacct'],'related_job_accounting':snap['related_job_accounting']}),flush=True)
