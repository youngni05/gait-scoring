"""Post-hoc: what distinguishes high measured-GDI people in the development data? No model runs here."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
LIVE = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in ('x', 'window_record', 'y', 'person_id', 'fold', 'source')}
gate = json.loads((ROOT / 'studies/20261105-normal-transfer-followup/source-gate-results.json').read_text(encoding='utf-8-sig'))
old = json.loads((ROOT / 'studies/20261105-unseen-normal-test/preparation.json').read_text(encoding='utf-8-sig'))
meta = json.loads((HERE / 'preparation.json').read_text(encoding='utf-8'))


def per_record(x, owner, n):
    """Knee and ankle range, step separation range and step rate, averaged over legs then windows."""
    knee = np.ptp(x[:, :, :, 8], axis=2).mean(1) * 180 / np.pi
    ankle = np.ptp(x[:, :, :, 9], axis=2).mean(1) * 180 / np.pi
    sep = x[:, 0, :, 11]
    step = np.ptp(sep, axis=1)
    s = sep - sep.mean(1, keepdims=True)
    power = np.abs(np.fft.rfft(s, axis=1)) ** 2
    freq = np.fft.rfftfreq(124, 1 / 30)
    band = (freq >= .3) & (freq <= 2.5)
    rate = freq[band][power[:, band].argmax(1)] * 2 * 60
    w = np.stack([knee, ankle, step, rate], 1)
    out = np.zeros((n, 4))
    np.add.at(out, owner, w)
    f = np.concatenate([x.mean(2), x.std(2), np.ptp(x, axis=2), np.abs(np.diff(x, axis=2)).mean(2)], axis=2).mean(1)
    F = np.zeros((n, 48))
    np.add.at(F, owner, f)
    c = np.bincount(owner, minlength=n)[:, None]
    return out / c, F / c


def prob(F, m):
    return 1 / (1 + np.exp(-(((F - np.array(m['mean'])) / np.array(m['scale'])) @ np.array(m['coef']) + m['intercept'])))


M, F = per_record(d['x'], d['window_record'], len(d['y']))
models = {m['fold']: m for m in gate['models']}
P = np.zeros(len(d['y']))
for f in range(5):
    k = d['fold'] == f
    P[k] = prob(F[k], models[f])
stan = d['source'] == 'Stanford'
y = d['y']
hi, lo = y.max(1), y.min(1)
names = ['knee_range_deg', 'ankle_range_deg', 'step_separation', 'steps_per_min']
out = {'note': 'Development Stanford records and LJMU. Probability is the held-out summary-statistic source gate.'}


def show(mask, ids=None):
    r = {n: round(float(M[mask, i].mean()), 2) for i, n in enumerate(names)}
    r.update(records=int(mask.sum()), P_LJMU_mean=round(float(P[mask].mean()), 3), share_P_ge_half=round(float((P[mask] >= .5).mean()), 3))
    return r


out['groups'] = {
    'Stanford_both_legs_lt70': show(stan & (hi < 70)),
    'Stanford_best_leg_70_90': show(stan & (hi >= 70) & (hi < 90)),
    'Stanford_best_leg_90_100': show(stan & (hi >= 90) & (hi < 100)),
    'Stanford_best_leg_ge100': show(stan & (hi >= 100)),
    'Stanford_both_legs_ge90': show(stan & (lo >= 90)),
    'Stanford_both_legs_ge100': show(stan & (lo >= 100)),
    'LJMU': show(~stan),
}
k = stan & (hi >= 100)
out['ge100_legs'] = dict(records=int(k.sum()), other_leg_mean=round(float(lo[k].mean()), 2), other_leg_lt90=round(float((lo[k] < 90).mean()), 3),
                         gap_mean=round(float((hi - lo)[k].mean()), 2), both_ge100=int((stan & (lo >= 100)).sum()))
ext = {}
with np.load(OLD / 'inputs.npz') as z:
    for k_, nm in (('live', 'LIVE_TD'), ('self', 'self')):
        ext[nm] = (z[k_ + '__x'].astype(float), z[k_ + '__owner'], int(z[k_ + '__n']))
with np.load(LIVE / 'inputs.npz') as z:
    ext['patient'] = (z['x'].astype(float), z['owner'], len(z['owner']))
nums = np.array(old['self_numbers'])
grp = np.array([r['group'] for r in meta['records']])
for nm, (x, o, n) in ext.items():
    m, f = per_record(x, o, n)
    p = np.mean([prob(f, g) for g in gate['models']], axis=0)
    sets = {nm: np.ones(n, bool)}
    if nm == 'self':
        sets = {'self_normal': np.isin(nums, old['self_normal']), 'self_abnormal': np.isin(nums, old['self_abnormal'])}
    if nm == 'patient':
        sets = {'LIVE_CP': grp == 'CP', 'LIVE_ITW': grp == 'ITW'}
    for key, w in sets.items():
        r = {q: round(float(m[w, i].mean()), 2) for i, q in enumerate(names)}
        r.update(records=int(w.sum()), P_LJMU_mean=round(float(p[w].mean()), 3), share_P_ge_half=round(float((p[w] >= .5).mean()), 3))
        out['groups'][key] = r
(HERE / 'posthoc-high-scorers.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
for key, v in out['groups'].items():
    print(key, v)
print(out['ge100_legs'])
