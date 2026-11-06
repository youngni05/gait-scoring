"""Retrieve pilot evidence to D and independently replay both models locally."""
from pathlib import Path
import json,hashlib,subprocess,tarfile,os,sys
from deploy import H,D,REMOTE,AUTH,ssh,sha

def main():
    registry=H.parents[2]/'control/gpu-watch/registry.json'
    reg=json.loads(registry.read_text(encoding='utf-8-sig'))
    entry=next(e for e in reg['experiments'] if e['id']=='cnn-ranksim-692')
    assert entry['remote']==REMOTE and entry['jobs']==json.loads((H/'submission.json').read_text())['jobs']
    code="""from pathlib import Path
import json,hashlib
h=Path(%r);p=h/'runs/pilot-review.tar.gz'
assert (h/'control/approve-main.json').exists() and p.exists(), 'Pilot not ready'
print(json.dumps(dict(archive=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gate=json.loads((h/'control/approve-main.json').read_text()),workers={p.name:json.loads(p.read_text()) for p in (h/'control').glob('worker-*.json')})))
"""%REMOTE
    remote=json.loads(ssh(code));dest=D/'received/pilot';dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'pilot-review.tar.gz'
    if not archive.exists():subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe',*AUTH,'-P','10000','RESEARCH_USER@PRIVATE_HOST:'+remote['archive'],str(archive)],check=True)
    assert sha(archive)==remote['sha256'];extracted=dest/'extracted'
    if not extracted.exists():
        extracted.mkdir()
        with tarfile.open(archive) as tf:tf.extractall(extracted,filter='data')
    gate=remote['gate'];assert gate['manifest_sha256']==sha(H/'manifest.json')==sha(extracted/'manifest.json')
    for n,digest in gate['proofs'].items():assert sha(extracted/n)==digest
    assert len({w['uuid'] for w in gate['workers']})==3
    proofs={}
    for variant in ['mse','ranksim']:
        out=H/'checks'/('local-pilot-'+variant+'.json')
        subprocess.run([sys.executable,'-B',str(H/'review.py'),'--variant',variant,'--device','cpu','--folder',str(extracted/'runs/pilot'/variant/'pilot'),'--output',str(out)],check=True,env=os.environ.copy())
        q=json.loads(out.read_text());assert q['passed'];proofs[variant]=q
    receipt=dict(passed=True,server_local_sha256_matched=True,archive_sha256=remote['sha256'],archive=str(archive),extracted=str(extracted),local_cpu_replays=proofs,server=remote)
    (H/'checks/pilot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(passed=True,archive_sha256=remote['sha256'],replays=proofs,workers=remote['workers'])))
if __name__=='__main__':main()
