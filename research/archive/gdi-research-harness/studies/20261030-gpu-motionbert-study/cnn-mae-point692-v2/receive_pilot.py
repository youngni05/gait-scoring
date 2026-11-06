"""Retrieve pilot evidence to D and independently replay both models locally."""
from pathlib import Path
import json,hashlib,subprocess,tarfile,os,sys
from deploy import H,D,REMOTE,AUTH,ssh,sha

def main():
    registry=H.parents[2]/'control/gpu-watch/registry.json'
    reg=json.loads(registry.read_text(encoding='utf-8-sig'))
    entry=next(e for e in reg['experiments'] if e['id']=='cnn-mae-point-692')
    assert entry['remote']==REMOTE and entry['jobs']==json.loads((H/'submission.json').read_text())['jobs']
    code="""from pathlib import Path
import json,hashlib
h=Path(%r);p=h/'runs/pilot-review.tar.gz'
assert (h/'control/approve-main.json').exists() and p.exists(), 'Pilot not ready'
print(json.dumps(dict(archive=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gate=json.loads((h/'control/approve-main.json').read_text()),workers={p.name:json.loads(p.read_text()) for p in (h/'control').glob('worker-*.json')},server_epoch=__import__('time').time())))
"""%REMOTE
    remote=json.loads(ssh(code))
    assert remote['workers']['worker-0.json']['phase']=='waiting_local_pilot_audit', 'Master is still writing the pilot archive'
    dest=D/'received/pilot';dest.mkdir(parents=True,exist_ok=True)
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
    from runtime_gate import evaluate
    forecast=evaluate(extracted,remote)
    (H/'checks/runtime-estimate.json').write_text(json.dumps(forecast,indent=2)+'\n')
    assert forecast['passed'],'Remaining allocation cannot safely fit projected main tasks; preserve waiting workers and report'
    # Approve only the exact manifest and pilot evidence replayed locally.
    fresh=json.loads(registry.read_text(encoding='utf-8-sig'))
    assert fresh['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
    current=next(e for e in fresh['experiments'] if e['id']=='cnn-mae-point-692')
    assert current['jobs']==entry['jobs'] and current['remote']==REMOTE
    approval=dict(passed=True,manifest_sha256=gate['manifest_sha256'],proofs=gate['proofs'],local_report_sha256=sha(H/'checks/pilot-receipt.json'))
    import base64
    payload=base64.b64encode(json.dumps(approval).encode()).decode()
    ssh("from pathlib import Path\nimport base64,json\np=Path(%r)/'control/approve-main-local.json'\nx=base64.b64decode(%r)\nif p.exists():assert json.loads(p.read_text())==json.loads(x)\nelse:p.write_bytes(x)\n"%(REMOTE,payload))
    print(json.dumps(dict(passed=True,archive_sha256=remote['sha256'],replays=proofs,workers=remote['workers'])))
if __name__=='__main__':main()
