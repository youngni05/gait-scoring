"""Post-hoc response to review M1 and M2. Uses saved predictions only; no model runs here.

M1: does group explain the rise beyond the before-addition score?
M2: how do predictions change on the 222 evaluation people, by measured GDI band?
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
spec = {'__file__': str(HERE / 'analyze.py')}
exec(open(HERE / 'analyze.py', encoding='utf-8').read().split("rng = np.random.default_rng")[0], spec)
tdp = spec['tdp']
b = {'TD': tdp['before'], 'CP': None, 'ITW': None}
a = {'TD': tdp['after_measured'], 'CP': None, 'ITW': None}
for g in ('CP', 'ITW'):
    v = R['groups'][g]['per_person']
    b[g] = np.array([x['before'] for x in v.values()])
    a[g] = np.array([x['after_measured'] for x in v.values()])
before = np.concatenate([b[g] for g in ('TD', 'CP', 'ITW')])
rise = np.concatenate([a[g] - b[g] for g in ('TD', 'CP', 'ITW')])
group = np.array(['TD'] * len(b['TD']) + ['CP'] * len(b['CP']) + ['ITW'] * len(b['ITW']))


def fit(i, quad=False):
    c = before[i] - 75
    cols = [np.ones(len(i)), c, group[i] == 'CP', group[i] == 'ITW'] + ([c ** 2] if quad else [])
    return np.linalg.lstsq(np.column_stack(cols).astype(float), rise[i], rcond=None)[0]


rng = np.random.default_rng(20261107)
idx = {g: np.flatnonzero(group == g) for g in ('TD', 'CP', 'ITW')}
out = {'people': {g: len(v) for g, v in idx.items()}}
for name, quad in (('linear', False), ('quadratic', True)):
    point = fit(np.arange(len(rise)), quad)
    draws = np.array([fit(np.concatenate([rng.choice(v, len(v)) for v in idx.values()]), quad) for _ in range(10000)])
    out['M1_' + name] = dict(
        slope_per_point=float(point[1]), slope_ci95=[float(x) for x in np.quantile(draws[:, 1], [.025, .975])],
        CP_vs_TD_at_same_before=float(point[2]), CP_ci95=[float(x) for x in np.quantile(draws[:, 2], [.025, .975])],
        ITW_vs_TD_at_same_before=float(point[3]), ITW_ci95=[float(x) for x in np.quantile(draws[:, 3], [.025, .975])])
lo, hi = b['TD'].min(), b['TD'].max()
inside = (b['CP'] >= lo) & (b['CP'] <= hi)
out['M1_overlap'] = dict(td_before_range=[float(lo), float(hi)], cp_inside=int(inside.sum()), cp_max=float(b['CP'].max()),
                         cp_mean_before=float(b['CP'][inside].mean()) if inside.any() else None, cp_mean_rise=float((a['CP'] - b['CP'])[inside].mean()) if inside.any() else None,
                         td_mean_before=float(b['TD'].mean()), td_mean_rise=float((a['TD'] - b['TD']).mean()))
Z = np.load('D:/gait-research-data/D_P/final222-normal100-20261106/fusion-predictions.npz')
ids, y = Z['person_id'].astype(str), Z['y']
base, cal = Z['baseline'], Z['mixed_stanford_cal']
assert len(set(ids.tolist())) == 222
BANDS = {'lt50': (-1e9, 50), '50_70': (50, 70), '70_90': (70, 90), '90_100': (90, 100), 'ge100': (100, 1e9)}
out['M2_by_measured_band'] = {}
for k, (l, h) in BANDS.items():
    m = (y >= l) & (y < h)
    q = np.unique(ids[m.any(1)])
    v = np.array([(cal - base)[m & (ids == p)[:, None]].mean() for p in q])
    d = v[np.random.default_rng(20261106).integers(0, len(v), (10000, len(v)))].mean(1)
    out['M2_by_measured_band'][k] = dict(people=len(q), before=float(np.mean([base[m & (ids == p)[:, None]].mean() for p in q])), rise=float(v.mean()), ci95=[float(x) for x in np.quantile(d, [.025, .975])])
person_b = np.array([base[ids == p].mean() for p in np.unique(ids)])
person_r = np.array([(cal - base)[ids == p].mean() for p in np.unique(ids)])
out['M2_all'] = dict(rise=float(person_r.mean()), sd=float(person_r.std(ddof=1)), corr_with_before=float(np.corrcoef(person_b, person_r)[0, 1]), slope=float(np.polyfit(person_b, person_r, 1)[0]))
out['M2_by_before_band'] = {}
for k, (l, h) in {'lt70': (0, 70), '70_75': (70, 75), '75_80': (75, 80), '80_85': (80, 85), 'ge85': (85, 200)}.items():
    m = (person_b >= l) & (person_b < h)
    out['M2_by_before_band'][k] = dict(people=int(m.sum()), rise=float(person_r[m].mean()) if m.any() else None)
    for g in ('TD', 'CP', 'ITW'):
        w = (b[g] >= l) & (b[g] < h)
        out['M2_by_before_band'][k]['LIVE_' + g] = dict(people=int(w.sum()), rise=float((a[g] - b[g])[w].mean()) if w.any() else None)
(HERE / 'posthoc-baseline-domain.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(out, indent=0))
