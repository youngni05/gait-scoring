"""Wait only own round08 receipt; never starts or changes training."""
from pathlib import Path
import time,subprocess,sys,json,os
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');R=H/'round08'
(R/'review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='waiting_for_receipt')),encoding='utf8')
while not(R/'run-receipt.json').exists():time.sleep(10)
r=subprocess.run([sys.executable,str(H/'review8.py')],stdout=(R/'review-log.txt').open('w'),stderr=subprocess.STDOUT)
(R/'review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='verified' if r.returncode==0 else 'failed',exit_code=r.returncode)),encoding='utf8');raise SystemExit(r.returncode)
