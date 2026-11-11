"""Post-hoc, not preregistered: do the model's own ankle and knee inputs differ between groups? No model runs here."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
ROOT = HERE.parents[1]
meta = json.loads((HERE / 'preparation.json').read_text(encoding='utf-8'))
old = json.loads((ROOT / 'studies/20261105-unseen-normal-test/preparation.json').read_text(encoding='utf-8-sig'))
NAMES = {8: 'knee_angle', 9: 'ankle_angle', 10: 'toe_ankle_distance'}


def summarize(x):
    """Per record, left and right averaged: mean, within-window range."""
    return {n: (x[:, :, :, k].mean(axis=(1, 2)), np.ptp(x[:, :, :, k], axis=2).mean(1)) for k, n in NAMES.items()}


with np.load(BIG / 'inputs.npz') as z:
    p = z['x'].astype(float)
with np.load(OLD / 'inputs.npz') as z:
    t = z['live__x'].astype(float)
    towner = z['live__owner']
tn = np.array([t[towner == i].mean(0) for i in range(12)])
ids = {'TD': np.array(old['live_people'])}
grp = np.array([r['group'] for r in meta['records']])
pid = np.array([r['person'] for r in meta['records']])
sets = {'TD': (summarize(tn), ids['TD'], np.ones(12, bool))}
for g in ('CP', 'ITW'):
    sets[g] = (summarize(p), pid, grp == g)
out = {}
for g, (s, who, keep) in sets.items():
    out[g] = {}
    for n, (m, r) in s.items():
        q = sorted(set(who[keep].tolist()))
        pm = np.array([m[keep][who[keep] == i].mean() for i in q])
        pr = np.array([r[keep][who[keep] == i].mean() for i in q])
        out[g][n] = dict(mean=round(float(pm.mean()), 3), sd=round(float(pm.std(ddof=1)), 3), range=round(float(pr.mean()), 3), range_sd=round(float(pr.std(ddof=1)), 3))
(HERE / 'posthoc-features.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out))
