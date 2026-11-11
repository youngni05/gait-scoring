"""Freeze patient model inputs from the reviewed windows. No model output is read here."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'tmp/ljmu-analysis-python-deps'), str(ROOT / 'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')]
import numpy as np
from preprocess import extract_windows

BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
POSE = BIG / 'body25'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


review = json.loads((HERE / 'review.json').read_text(encoding='utf-8'))
cand = {r['record']: r for r in json.loads((HERE / 'candidates.json').read_text(encoding='utf-8'))}
xs, rows = [], []
for rec, d in sorted(review['accepted'].items()):
    c = cand[rec]['candidates'][d['candidate']]
    raw = np.full((124, 75), np.nan)
    for k, sel in enumerate(c['person_indices']):
        if sel >= 0:
            path = POSE / rec / ('%s_%012d_keypoints.json' % (rec, c['start'] + k))
            raw[k] = json.loads(path.read_text())['people'][sel]['pose_keypoints_2d']
    e = extract_windows(raw)
    assert e['window_start'].tolist() == [0]
    xs.append(e['x'])
    rows.append(dict(record=rec, group=d['group'], person=d['person'], start=c['start'],
                     missing=float(e['window_missing_fraction'][0]), moving=d['moving'], assistive_device=d['assistive_device']))
x = np.concatenate(xs).astype(np.float32)
assert x.shape == (len(rows), 2, 124, 12) and np.isfinite(x).all()
np.savez_compressed(BIG / 'inputs.npz', x=x, owner=np.arange(len(rows)))
(HERE / 'preparation.json').write_text(json.dumps(dict(
    protocol_sha256=sha(HERE / 'protocol.json'), review_sha256=sha(HERE / 'review.json'),
    inputs_sha256=sha(BIG / 'inputs.npz'), records=rows), indent=1), encoding='utf-8')
print(x.shape, len({r['person'] for r in rows}))
