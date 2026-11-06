from pathlib import Path
import time,sys,subprocess
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');start=time.monotonic()
while not (H/'round02/run-receipt.json').exists():
    if time.monotonic()-start>3600:raise TimeoutError('Round02 did not finish within local wait budget')
    time.sleep(5)
with (H/'round2-review-log.txt').open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/'review2.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Round02 independently verified',flush=True)
