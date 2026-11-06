"""Wait for this local study only, then replay; never submit/change any training."""
from pathlib import Path
import sys,time,subprocess,json,os
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');R=H/'native-motion'
(R/'review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='waiting_for_receipt')),encoding='utf8')
while not(R/'receipt.json').exists():
    if(R/'process.json').exists():
        p=json.loads((R/'process.json').read_text())
        try:os.kill(p['pid'],0)
        except OSError:raise RuntimeError('This study training process missing before finalreceipt; inspect original logs, do not restart silently')
    time.sleep(10)
result=subprocess.run([sys.executable,str(H/'native_review.py')],stdout=(R/'review-log.txt').open('w'),stderr=subprocess.STDOUT)
(R/'review-process.json').write_text(json.dumps(dict(pid=os.getpid(),status='verified' if result.returncode==0 else 'failed',exit_code=result.returncode)),encoding='utf8')
raise SystemExit(result.returncode)
