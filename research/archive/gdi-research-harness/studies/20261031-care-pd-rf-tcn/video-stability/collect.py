"""Collect only this video inference result and verify transport hashes."""
import sys,json,hashlib,subprocess,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));from remote import command,AUTH,HOST
REMOTE='/abr/RESEARCH_USER/care-pd-video-stability-20261031'

def main():
    code="""from pathlib import Path
import json,tarfile,hashlib,subprocess
p=Path('/abr/RESEARCH_USER/care-pd-video-stability-20261031')
d=json.loads((p/'main/complete.json').read_text());assert d['passed'] and d['frames']==10500 and len(d['outputs'])==70
a=p/'review.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for folder in ['main','pilot','logs']:
  for f in sorted((p/folder).rglob('*')):
   if f.is_file():t.add(f,arcname=str(f.relative_to(p)))
 for n in ['infer.py','protocol.json','clips.json','submit.slurm']:t.add(p/n,arcname=n)
print(json.dumps(dict(sha256=hashlib.sha256(a.read_bytes()).hexdigest(),bytes=a.stat().st_size)))
"""
    receipt=json.loads(command(code,timeout=80));target=HERE/'review.tar.gz'
    r=subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000',*AUTH,HOST+':'+REMOTE+'/review.tar.gz',str(target)],capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr
    assert target.stat().st_size==receipt['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==receipt['sha256']
    destination=HERE/'received';destination.mkdir(exist_ok=True)
    with tarfile.open(target) as t:
        for m in t.getmembers():
            assert m.isfile() and (destination/m.name).resolve().is_relative_to(destination.resolve())
        t.extractall(destination,filter='data')
    for n in ['infer.py','protocol.json','clips.json','submit.slurm']:
        assert (HERE/n).read_bytes()==(destination/n).read_bytes()
    receipt['passed']=True;(HERE/'collection-verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
