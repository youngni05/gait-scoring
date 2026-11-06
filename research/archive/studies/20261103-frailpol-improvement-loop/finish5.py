from pathlib import Path
import sys,subprocess
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
for script,log in [('round5.py','round5-log.txt'),('review5.py','round5-review-log.txt'),('uncertainty.py','uncertainty-log.txt')]:
    with (H/log).open('w',encoding='utf8') as f:subprocess.run([sys.executable,str(H/script)],stdout=f,stderr=subprocess.STDOUT,check=True)
print('Round05 completed and independently verified',flush=True)
