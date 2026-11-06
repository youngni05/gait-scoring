"""Collect only after both training/replay jobs succeed; verify archive hash."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from remote import command, HERE, REMOTE, AUTH, HOST


def main():
    submission=json.loads((HERE/'submission.json').read_text())
    jobs=submission['main_jobs']
    source='''import pathlib,json,subprocess,hashlib,tarfile
p=pathlib.Path(REMOTE)
s=subprocess.run(['sacct','-n','-P','-j',','.join(JOBS),'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True).stdout
states={r[0]:r[1:3] for line in s.splitlines() if len(r:=line.split('|'))>=3 and r[0] in JOBS}
assert all(states.get(j)==['COMPLETED','0:0'] for j in JOBS),states
assert json.loads((p/'runs/main/summary.json').read_text())['complete']
replays=[json.loads(q.read_text()) for q in (p/'runs/main').glob('replay-*.json')]
assert sum(r['checkpoints'] for r in replays)==30 and all(r['passed'] for r in replays)
archive=p/'review.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name in ['runs','logs','run.py','bootstrap.py','submit.slurm','protocol.json','manifest.csv']:
  t.add(p/name,arcname=name)
print(json.dumps(dict(jobs=states,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))
'''
    result=json.loads(command('REMOTE='+repr(REMOTE)+'\nJOBS='+repr(jobs)+'\n'+source,timeout=900))
    archive=HERE/'review.tar.gz'
    subprocess.run(['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH+
        [HOST+':'+REMOTE+'/review.tar.gz',str(archive)],check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==result['sha256']
    received=HERE/'received'; received.mkdir(exist_ok=True)
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            dest=(received/member.name).resolve()
            assert dest.is_relative_to(received.resolve()) and not member.issym() and not member.islnk()
        t.extractall(received)
    expected=json.loads((HERE/'execution-source-sha256.json').read_text())
    for name in ['run.py','bootstrap.py','submit.slurm']:
        assert hashlib.sha256((received/name).read_bytes()).hexdigest()==expected[name]
    (HERE/'collection-verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
