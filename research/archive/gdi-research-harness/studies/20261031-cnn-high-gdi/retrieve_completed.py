"""Retrieve completed registered training results with remote/local SHA256 checks."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile,datetime
HERE=Path(__file__).resolve().parent
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(cmd):return subprocess.check_output(SSH+[cmd],text=True,encoding='utf-8',timeout=90)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=['high90_x3','rare_sqrt_cap4']);ap.add_argument('--received',type=Path);args=ap.parse_args()
    r=json.loads((HERE/'watch/registry.json').read_text());assert r['thread_id']=='01a1906b-7581-7403-8c8e-797c78676dbe'
    e=next(e for e in r['experiments'] if e['local'].endswith('/'+args.mode));remote=e['remote']
    accounting=command('sacct -nP -j '+','.join(e['jobs'])+' --format=JobIDRaw,State,ExitCode')
    states={line.split('|')[0]:line.split('|')[1:3] for line in accounting.splitlines() if line}
    assert all(states.get(j)==['COMPLETED','0:0'] for j in e['jobs']),states
    marker=json.loads(command('cat '+remote+'/runs/summary_complete.json'))
    remote_sha=command('sha256sum '+remote+'/runs/main-review.tar.gz').split()[0]
    assert remote_sha==marker['sha256']
    dest=args.received or HERE/args.mode/'received';dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'main-review.tar.gz'
    if not archive.exists():subprocess.run(SCP+[HOST+':'+remote+'/runs/main-review.tar.gz',str(archive)],check=True,timeout=300)
    assert sha(archive)==remote_sha
    if not (dest/'results').exists():
        with tarfile.open(archive) as tf:
            for item in tf.getmembers():
                target=(dest/item.name).resolve()
                assert target.is_relative_to(dest.resolve()) and (item.isfile() or item.isdir())
            tf.extractall(dest,filter='data')
    logs=dest/'logs';logs.mkdir(exist_ok=True)
    log_hashes={}
    for job in e['jobs']:
        name='worker-'+job+'.log';digest=command('sha256sum '+remote+'/logs/'+name).split()[0]
        subprocess.run(SCP+[HOST+':'+remote+'/logs/'+name,str(logs/name)],check=True,timeout=90)
        assert sha(logs/name)==digest;log_hashes[name]=digest
    result=dict(retrieved_at=datetime.datetime.now().astimezone().isoformat(),mode=args.mode,jobs=states,archive_sha256=remote_sha,logs=log_hashes,final_cuda_replay=json.loads((dest/'results/final_replay_cuda.json').read_text()))
    assert result['final_cuda_replay']['passed']
    control=dest/'control';control.mkdir(exist_ok=True)
    result['control']={}
    for job in e['jobs']:
        name='worker-'+job+'.json'
        if command('test -f '+remote+'/control/'+name+' && echo yes || true').strip()!='yes':continue
        digest=command('sha256sum '+remote+'/control/'+name).split()[0]
        subprocess.run(SCP+[HOST+':'+remote+'/control/'+name,str(control/name)],check=True,timeout=90)
        assert sha(control/name)==digest;result['control'][name]=digest
    (dest/'retrieval_verification.json').write_text(json.dumps(result,indent=2))
    summary=HERE/args.mode/'analysis/retrieval_verification.json';summary.parent.mkdir(exist_ok=True)
    summary.write_text(json.dumps(dict(result,received_path=str(dest)),indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
