"""Retrieve CPU ablations; verify transport and replay every final RF locally."""
import sys, pathlib, json, subprocess, hashlib, tarfile
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from remote import command, AUTH, HOST
REMOTE='/abr/RESEARCH_USER/care-pd-feature-analysis-20261031'

def main():
    source='ROOT='+repr(REMOTE)+'\n'+r'''
import pathlib,json,subprocess,tarfile,hashlib
p=pathlib.Path(ROOT)
state=subprocess.run(['sacct','-n','-P','-j','32925','--format=JobIDRaw,State,ExitCode'],text=True,capture_output=True,check=True).stdout
assert '32925|COMPLETED|0:0' in state,state
assert json.loads((p/'complete.json').read_text())['complete']
assert len(list((p/'fits').glob('*/*.npz')))==90
assert len(list((p/'fits').glob('*/*.joblib')))==90
a=p/'review.tar.gz'
with tarfile.open(a,'w:gz') as t:
 for n in ['fits','comparison.json','complete.json','study.py','cpu_runner.py','protocol.json','slurm-32925.log']:
  t.add(p/n,arcname=n)
print(json.dumps({'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'bytes':a.stat().st_size,'accounting':state}))
'''
    result=json.loads(command(source,timeout=900))
    archive=HERE/'review.tar.gz'
    subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+[HOST+':'+REMOTE+'/review.tar.gz',str(archive)],check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==result['sha256']
    dest=HERE/'received';dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for m in t.getmembers():assert (dest/m.name).resolve().is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
        t.extractall(dest,filter='data')
    for n in ['study.py','cpu_runner.py','protocol.json']:assert (dest/n).read_bytes()==(HERE/n).read_bytes(),n
    (HERE/'collection-verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
