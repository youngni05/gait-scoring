from pathlib import Path
import sys,time,subprocess
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn/local-directml');start=time.monotonic()
while not all((H/(w+'-receipt.json')).exists() for w in ['controls','motion']):
    if time.monotonic()-start>21600:raise TimeoutError('Local workers not finished within6hours; inspect actual handles before retry')
    time.sleep(5)
with(H/'review-log.txt').open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/'review_local.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Local temporal models complete and independently reviewed',flush=True)
