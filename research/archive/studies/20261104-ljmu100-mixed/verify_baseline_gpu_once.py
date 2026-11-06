"""Verify the completed same-environment baseline rerun without waiting for main fits."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(HERE/'watch/baseline-gpu-receipt.json');assert sha(Path(receipt['archive']))==receipt['sha256']
checks=[]
for family in ['CNN','STT','bilateral']:
    for f in range(5):
        path=BIG/'received/verification-data/runs/baseline'/family/f'fold{f}'/'predictions.npz'
        assert sha(path)==read(path.parent/'complete.json')['prediction_sha256']
        firstpath=BIG/'received/data/runs/baseline'/family/f'fold{f}'/'predictions.npz'
        assert sha(firstpath)==read(firstpath.parent/'complete.json')['prediction_sha256']
        with np.load(path) as rerun,np.load(firstpath) as first:
            assert np.array_equal(rerun['row_index'],first['row_index'])
            errors={k:float(np.max(np.abs(rerun[k]-first[k]))) for k in ['refit','refit_raw','bag']}
            assert max(errors.values())<.001,(family,f,errors)
            checks.append({'family':family,'fold':f,'errors':errors})
out={'passed':True,'job':'35869','folds':checks,'archive_sha256':receipt['sha256'],
     'cross_device_exception_preserved':'CPU replay CNN fold3 bag max error0.0012155359 exceeds0.001. Same-environment replay does not erase this limitation.'}
(HERE/'baseline-gpu-verified.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'passed':True,'folds':len(checks),'max_error':max(max(r['errors'].values()) for r in checks)}))
