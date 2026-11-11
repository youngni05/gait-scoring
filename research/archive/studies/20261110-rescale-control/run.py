"""Rescale-only control: linear changes of the baseline ensemble score versus the retrained normal-added model."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ST = ROOT / 'studies'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


# development rows
Z = np.load('D:/gait-research-data/D_P/ljmu100-mixed-20261104/comparison-predictions.npz')
p_dev, y_dev, src = Z['baseline__fusion'], Z['y'], Z['source']
stan, lj = src == 'Stanford', src == 'LJMU'
assert np.all(y_dev[lj] == 100) and 79 < p_dev[lj].mean() < 82, p_dev[lj].mean()  # record-level mean; 80.47 in the paper is the person-level mean

# evaluation set
old = np.load('D:/gait-research-data/D_P/final223-frozen-20261102/predictions.npz')
rows = list(csv.DictReader((ROOT / 'gdi-research-harness/studies/20261102-final223-frozen/scores.csv').open(encoding='utf-8-sig')))
assert [r['video_id'] for r in rows] == old['video_id'].astype(str).tolist()
ids = np.array([r['person_id'] for r in rows])
y = np.array([[float(r['true_left']), float(r['true_right'])] for r in rows])
p_eval = old['fusion']


def person_mae(p):
    e = np.abs(p - y)
    return np.array([e[ids == q].mean() for q in np.unique(ids)])


base_people = person_mae(p_eval)
assert abs(base_people.mean() - 6.248559) < 1e-5
final = read(ST / '20261106-final222-normal100/results.json')['models']['fusion']['all']
added_mae = final['mixed_stanford_cal']['mae']

# external group means (same sources as the paper figures)
live = read(ST / '20261107-live-patient-eval/results.json')
older = read(ST / '20261106-older-normal100/results.json')['arms']
sb = read(ST / '20261105-unseen-normal-test/results.json')['self']['baseline']
sa = read(ST / '20261105-unseen-normal-test/stanford-calibration/results.json')['external']['mixed_stanford_cal']
groups = {
    'LIVE_TD': (live['TD']['before']['mean'], live['TD']['after_measured']['mean']),
    'LIVE_CP': (live['groups']['CP']['before']['mean'], live['groups']['CP']['after_measured']['mean']),
    'self_normal': (sb['normal_mean'], sa['self_normal_mean']),
    'self_abnormal': (sb['abnormal_mean'], sa['self_abnormal_mean']),
    'older_healthy': (older['baseline']['healthy65']['mean'], older['stan']['healthy65']['mean']),
    'older_stroke': (older['baseline']['stroke65']['mean'], older['stan']['stroke65']['mean']),
}
pairs = [('LIVE', 'LIVE_TD', 'LIVE_CP'), ('self', 'self_normal', 'self_abnormal'), ('older', 'older_healthy', 'older_stroke')]

# rescales
ps, ys = p_dev[stan].ravel(), y_dev[stan].ravel()
A = np.linalg.lstsq(np.column_stack([p_dev.ravel(), np.ones(p_dev.size)]), y_dev.ravel(), rcond=None)[0]


def constrained(anchor_pred, anchor_value):
    """min sum (a*ps+b-ys)^2 subject to a*anchor_pred+b = anchor_value."""
    u = ps - anchor_pred
    v = ys - anchor_value
    a = float((u * v).sum() / (u * u).sum())
    return np.array([a, anchor_value - a * anchor_pred])


B = constrained(float(p_dev[lj].mean()), 100.)
C = constrained(groups['LIVE_TD'][0], groups['LIVE_TD'][1])
rng = np.random.default_rng(20261110)
draw = rng.integers(0, len(base_people), (10000, len(base_people)))
out = {'baseline': {}, 'normal_added': {}}


def summary(name, scale=None, added=False):
    if added:
        means = {k: v[1] for k, v in groups.items()}
        mae = added_mae
        ci = final['mixed_stanford_cal']['delta_mae_ci95']
    else:
        a, b = (1., 0.) if scale is None else scale
        means = {k: a * v[0] + b for k, v in groups.items()}
        pm = person_mae(p_eval * a + b)
        mae = float(pm.mean())
        dm = pm - base_people
        ci = [float(v) for v in np.quantile(dm[draw].mean(1), [.025, .975])]
    row = {'scale': None if scale is None else [round(float(scale[0]), 4), round(float(scale[1]), 3)],
           'evaluation_mae': round(mae, 3), 'delta_mae_ci95': [round(v, 3) for v in ci],
           'means': {k: round(v, 2) for k, v in means.items()},
           'gaps': {s: round(means[n] - means[ab], 2) for s, n, ab in pairs}}
    out[name] = row


summary('baseline')
summary('normal_added', added=True)
summary('A_mixed_fit', A)
summary('B_ljmu_to_100', B)
summary('C_live_td_match', C)
out['protocol_sha256'] = hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest()
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
for k, v in out.items():
    if isinstance(v, dict):
        print(k, v['scale'], v['evaluation_mae'], v['delta_mae_ci95'], v['means'], v['gaps'])
