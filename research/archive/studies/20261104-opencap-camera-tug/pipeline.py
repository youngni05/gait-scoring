"""Continue this approved experiment when its bounded acquisition completes."""
from pathlib import Path
import json, subprocess, sys, time
HERE=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
deadline=time.monotonic()+45*60
while time.monotonic()<deadline:
    p=HERE/'acquisition.json'
    if p.exists():
        try: ready=json.loads(p.read_text())['complete']
        except json.JSONDecodeError: ready=False
        if ready:break
    time.sleep(10)
else:raise TimeoutError('Acquisition did not finish within bounded 45-minute wait. No fit started.')
assert not (HERE/'selection.json').exists(), 'Do not overwrite an existing training run.'
for stage in ['prepare','train']:
    subprocess.run([sys.executable,str(HERE/'experiment.py'),stage],check=True)
subprocess.run([sys.executable,str(HERE/'verify.py')],check=True)
print('PIPELINE_VERIFIED',flush=True)
