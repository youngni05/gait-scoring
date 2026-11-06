"""Foreground-session monitoring of this experiment only; no app automation."""
import datetime,hashlib,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
SSH=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/ssh.exe')
SCP=str(Path(os.environ['WINDIR'])/'System32/OpenSSH/scp.exe')
OPTIONS=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i',str(Path.home()/'.ssh/id_ed25519_gdi_abrm02.pub')]
HOST='RESEARCH_USER@PRIVATE_HOST';REMOTE='/abr/RESEARCH_USER/ljmu100-mixed-20261104'
seen=set();last_state=None
for receipt in (HERE/'watch').glob('*-fold*-receipt.json'):
    rec=json.loads(receipt.read_text(encoding='utf-8'))
    archive=Path(rec['archive'])
    if archive.exists() and hashlib.sha256(archive.read_bytes()).hexdigest()==rec['sha256']:
        seen.add(rec['family']+'-fold'+str(rec['fold']))
def ssh(command):return subprocess.check_output([SSH,'-p','10000',*OPTIONS,HOST,command],timeout=60,text=True)
while True:
    try:
        snap=json.loads(ssh('/usr/bin/python3.8 '+REMOTE+'/remote_status.py'))
        (HERE/'watch/latest.json').write_text(json.dumps(snap,indent=2),encoding='utf-8')
        active=[v for v in snap['fits'] if not v['complete']]
        states=[line.split('|')[1] for line in snap['sacct'].splitlines() if line.split('|')[0] in ['35866','35867']]
        current=tuple(states)
        if current!=last_state:
            print(json.dumps({'event':'state','states':states,'utc':snap['utc']}),flush=True);last_state=current
        for v in snap['completed_folds']:
            tag=v['family']+'-fold'+str(v['fold'])
            if tag in seen:continue
            folder='data/runs/mixed/'+v['family']+'/fold'+str(v['fold']);archive=tag+'-results.tar.gz'
            digest=ssh('cd '+REMOTE+' && tar -czf '+archive+' '+folder+' && sha256sum '+archive).split()[0]
            target=BIG/'received'/archive
            subprocess.run([SCP,'-P','10000',*OPTIONS,HOST+':'+REMOTE+'/'+archive,str(target)],check=True,timeout=120)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
            subprocess.run(['tar','-xzf',str(target),'-C',str(BIG/'received')],check=True)
            receipt={'family':v['family'],'fold':v['fold'],'archive':str(target),'sha256':digest,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            (HERE/'watch'/(tag+'-receipt.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            seen.add(tag);print(json.dumps({'event':'fold_retrieved',**receipt}),flush=True)
        if len(states)==2 and all(s=='COMPLETED' for s in states):
            assert len(seen)==15
            print('ALL_15_FOLDS_RECEIVED',flush=True);break
        if any(s not in ['RUNNING','PENDING','COMPLETED','COMPLETING','CONFIGURING'] for s in states):
            print(json.dumps({'event':'ATTENTION','states':states}),flush=True);break
    except (subprocess.SubprocessError,OSError,ValueError) as exc:
        print(json.dumps({'event':'monitor_error','error':str(exc)}),flush=True)
    # Short sleep steps keep this foreground monitor interruptible.
    for _ in range(5):time.sleep(60)
