from pathlib import Path
import sys,time,subprocess,json
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');start=time.monotonic()
while not all((H/'round06'/f'receipt-{c}.json').exists() for c in ['speed','bilateral']):
    if time.monotonic()-start>14400:raise TimeoutError('Workers not complete within4hours; inspect actual handles before any retry')
    time.sleep(5)
for script,log in [('review6.py','round6-review-log.txt'),('uncertainty.py','uncertainty-log.txt')]:
    with(H/log).open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Round06 completed and independently verified',flush=True)
