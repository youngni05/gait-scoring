from pathlib import Path
import subprocess,sys,time,json
H=Path('D:/Projects/D_P/studies/20261103-frailpol-objective-threshold');start=time.monotonic()
while not (H/'run-receipt.json').exists():
    if time.monotonic()-start>3600:raise TimeoutError('Study did not finish within the local wait budget')
    time.sleep(5)
with (H/'review-log.txt').open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/'review.py')],check=True,stdout=f,stderr=subprocess.STDOUT)
print('Independent review passed',flush=True)
subprocess.run([sys.executable,str(H/'report.py')],check=True)
