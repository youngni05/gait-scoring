"""After full round08 verification only, run predeclared round10 and independent review."""
from pathlib import Path
import time,subprocess,sys,json,os
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');R=H/'round10'
(R/'process.json').write_text(json.dumps(dict(pid=os.getpid(),status='waiting_round08_verification')),encoding='utf8')
while not(H/'round08/verification.json').exists():
    p=H/'round08/review-process.json'
    if p.exists() and json.loads(p.read_text()).get('status')=='failed':raise RuntimeError('Round08reviewfailed;do notrunround10')
    time.sleep(10)
assert json.loads((H/'round08/verification.json').read_text())['status']=='passed'
for name in ['round10.py','review10.py']:
    r=subprocess.run([sys.executable,str(H/name)],stdout=(R/(name+'-log.txt')).open('w'),stderr=subprocess.STDOUT)
    if r.returncode:
        (R/'process.json').write_text(json.dumps(dict(pid=os.getpid(),status='failed',step=name,exit_code=r.returncode)),encoding='utf8');raise SystemExit(r.returncode)
(R/'process.json').write_text(json.dumps(dict(pid=os.getpid(),status='verified')),encoding='utf8')
