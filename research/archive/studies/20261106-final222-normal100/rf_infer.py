"""RF predictions on the 222-person inputs with saved baseline and normal-added forests. No fitting."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, 'D:/Projects/D_P/outputs/gdi-ridge-experiment-20261016/deps')
import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
BIG = Path('D:/gait-research-data/D_P/final222-normal100-20261106')
FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


with np.load(FINAL / 'features.npz') as z:
    x, owner, ids = z['x'], z['window_record'], z['person_id']
n = len(ids)
w = np.concatenate([x.mean(axis=2), x.std(axis=2), np.ptp(x, axis=2), np.abs(np.diff(x, axis=2)).mean(axis=2)], axis=2)
feat = np.zeros((n, 2, 48), dtype=np.float64)
np.add.at(feat, owner, w)
feat /= np.bincount(owner, minlength=n)[:, None, None]
assert np.isfinite(feat).all()
out = {}
for arm in ('baseline', 'mixed'):
    preds = []
    for f in range(5):
        model = joblib.load(SRC / 'runs' / arm / 'RF' / f'fold{f}' / 'model.joblib')
        preds.append(model.predict(feat.reshape(-1, 48)).reshape(-1, 2))
    out[arm] = np.mean(preds, axis=0)
target = BIG / 'rf.npz'
np.savez_compressed(target, **out)
(HERE / 'infer-RF.json').write_text(json.dumps(dict(prediction_sha256=sha(target), protocol_sha256=sha(HERE / 'protocol.json')), indent=1), encoding='utf-8')
print('done RF', flush=True)
