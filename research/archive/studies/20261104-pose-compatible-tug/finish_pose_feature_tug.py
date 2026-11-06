"""Wait only for these two local studies, then run their authorized full review."""
from pathlib import Path
import time,subprocess,sys,json
H=Path('D:/Projects/D_P/studies/20261104-pose-compatible-tug');start=time.monotonic()
while not all((H/p).exists() for p in ['run-receipt.json','complete-feature-controls/run-receipt.json']):
    assert time.monotonic()-start<3600,'Own training receipts not complete after bounded wait'
    time.sleep(20)
with (H/'review-log.txt').open('w',encoding='utf8') as log:r=subprocess.run([sys.executable,str(H/'review_pose_feature_tug.py')],stdout=log,stderr=subprocess.STDOUT)
print(json.dumps(dict(review_exit_code=r.returncode,log=str(H/'review-log.txt'))));sys.exit(r.returncode)
