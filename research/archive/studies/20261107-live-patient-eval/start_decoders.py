"""Start decoder shards in hidden consoles with their own process groups. Resume is safe."""
import json,subprocess
from pathlib import Path
BIG=Path('D:/gait-research-data/D_P/live-patient-eval-20261107');PY='D:/Projects/D_P/tmp/pose-python37/python.exe'
SC=str(Path(__file__).resolve().parent/'consume_patient.py');F=0x08000000|0x00000200
out=[]
for k in range(4):
    log=open(BIG/('consumer-shard%d.log'%k),'ab')
    args=[PY,'-u',SC,'--shard',str(k),'--shards','4']
    try:p=subprocess.Popen(args,creationflags=F|0x01000000,stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=str(BIG),close_fds=True);breakaway=True
    except OSError:p=subprocess.Popen(args,creationflags=F,stdin=subprocess.DEVNULL,stdout=log,stderr=log,cwd=str(BIG),close_fds=True);breakaway=False
    out.append(dict(shard=k,pid=p.pid,job_breakaway=breakaway))
(BIG/'consumer-process.json').write_text(json.dumps(out,indent=1));print(out)
