"""Post-hoc: is underprediction of high legs tied to a lower other leg? Saved development predictions only."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
Z = np.load(SRC / 'comparison-predictions.npz')
print([k for k in Z.files if 'fusion' in k or k in ('y', 'source', 'person_id')])
S = np.load(OLD / 'stanford-cal-fusion.npz')
print(S.files[:10], {k: S[k].shape for k in S.files[:6]})
s = Z['source'] == 'Stanford'
y = Z['y'][s]
base = Z['baseline__fusion'][s]
with np.load(SRC / 'data.npz') as z:
    pid = z['person_id'][z['source'] == 'Stanford']
assert len(pid) == len(y)
out = {}
hi = y.max(1)
lo = y.min(1)
out['records_best_ge100'] = int((hi >= 100).sum())
k = hi >= 100
out['other_leg_distribution'] = dict(lt70=int((lo[k] < 70).sum()), r70_80=int(((lo[k] >= 70) & (lo[k] < 80)).sum()), r80_90=int(((lo[k] >= 80) & (lo[k] < 90)).sum()), r90_100=int(((lo[k] >= 90) & (lo[k] < 100)).sum()), ge100=int((lo[k] >= 100).sum()))
leg_y = y.ravel()
other = y[:, ::-1].ravel()
pred = base.ravel()
ids = np.repeat(pid, 2)


def bias(mask):
    q = np.unique(ids[mask])
    return np.array([(pred - leg_y)[mask & (ids == p)].mean() for p in q])


rng = np.random.default_rng(20261107)
out['high_leg_bias'] = {}
for name, l, h in (('ge100', 100, 1e9), ('ge90', 90, 1e9)):
    for label, m in (('other_lt80', other < 80), ('other_80_90', (other >= 80) & (other < 90)), ('other_ge90', other >= 90)):
        mask = (leg_y >= l) & (leg_y < h) & m
        if mask.sum() == 0:
            continue
        b = bias(mask)
        ci = np.quantile(b[rng.integers(0, len(b), (10000, len(b)))].mean(1), [.025, .975]) if len(b) > 1 else [np.nan, np.nan]
        out['high_leg_bias'][f'{name}_{label}'] = dict(people=len(b), legs=int(mask.sum()), mean_measured=float(leg_y[mask].mean()), mean_pred=float(pred[mask].mean()), bias=float(b.mean()), ci95=[float(x) for x in ci])
# Error on leg GDI and other-leg GDI, person-level bootstrap
X = np.column_stack([np.ones(len(leg_y)), leg_y - 80, other - 80])
coef = np.linalg.lstsq(X, pred, rcond=None)[0]
people = np.unique(ids)
rows = {p: np.flatnonzero(ids == p) for p in people}
draws = []
for _ in range(2000):
    i = np.concatenate([rows[p] for p in rng.choice(people, len(people))])
    draws.append(np.linalg.lstsq(X[i], pred[i], rcond=None)[0])
draws = np.array(draws)
out['prediction_on_own_and_other_leg'] = dict(own_leg=float(coef[1]), own_ci95=[float(x) for x in np.quantile(draws[:, 1], [.025, .975])], other_leg=float(coef[2]), other_ci95=[float(x) for x in np.quantile(draws[:, 2], [.025, .975])])
out['asymmetry_tracking'] = dict(measured_lr_gap_sd=float((y[:, 0] - y[:, 1]).std()), predicted_lr_gap_sd=float((base[:, 0] - base[:, 1]).std()), corr=float(np.corrcoef(y[:, 0] - y[:, 1], base[:, 0] - base[:, 1])[0, 1]))
(HERE / 'posthoc-asymmetry.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out, indent=0))
