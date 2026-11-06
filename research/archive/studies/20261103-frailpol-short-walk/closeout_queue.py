"""Run the authorized local verification after this study's fit process completes."""
from pathlib import Path
import sys,time,subprocess
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk')
start=time.monotonic()
while not (H/'run-receipt.json').exists():
    if time.monotonic()-start>3600:raise TimeoutError('This study did not finish within the local wait budget')
    time.sleep(5)
for script in ['compare.py','review.py']:
    print('starting',script,flush=True)
    subprocess.run([sys.executable,str(H/script)],check=True,cwd=H)
