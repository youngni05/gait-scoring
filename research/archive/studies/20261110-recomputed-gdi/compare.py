"""Model scores (baseline, normal-added) against GDI recomputed with healthy adults as the reference group."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent
z = np.load('D:/gait-research-data/D_P/recomputed-gdi-20261110/gait-vectors.npz')
person, cohort, age = z['person'], z['cohort'], z['age']
model = json.loads((ST / '20261106-older-normal100/people-scores.json').read_text(encoding='utf-8-sig'))
rng = np.random.default_rng(20261110)
out = {}


def corr(a, b):
    rk = lambda v: np.argsort(np.argsort(v))
    return round(float(np.corrcoef(a, b)[0, 1]), 3), round(float(np.corrcoef(rk(a), rk(b))[0, 1]), 3)


for ref in ('R_all', 'R_young'):
    g = z[f'gdi_{ref}']
    gp = {p: float(g[person == p].mean()) for p in set(person.tolist())}
    rows = [m for m in model if m['id'] in gp]
    ids = [m['id'] for m in rows]
    G = np.array([gp[i] for i in ids])
    Bm = np.array([m['baseline'] for m in rows])
    Am = np.array([m['added'] for m in rows])
    coh = np.array([m['cohort'] for m in rows])
    ag = np.array([m['age'] for m in rows])
    eb, ea = np.abs(Bm - G), np.abs(Am - G)
    draw = rng.integers(0, len(G), (10000, len(G)))
    res = {'matched_people': len(G), 'by_cohort': {c: int((coh == c).sum()) for c in ('healthy', 'stroke')},
           'ge65': int((ag >= 65).sum()),
           'mae_baseline': round(float(eb.mean()), 2), 'mae_added': round(float(ea.mean()), 2),
           'mae_difference': round(float((ea - eb).mean()), 2),
           'mae_difference_ci95': [round(float(v), 2) for v in np.quantile((ea - eb)[draw].mean(1), [.025, .975])],
           'corr_baseline_pearson_spearman': corr(Bm, G), 'corr_added_pearson_spearman': corr(Am, G)}
    cd = []
    for d in draw[:4000]:
        cd.append(np.corrcoef(Am[d], G[d])[0, 1] - np.corrcoef(Bm[d], G[d])[0, 1])
    res['pearson_difference_ci95'] = [round(float(v), 3) for v in np.nanquantile(cd, [.025, .975])]
    for c in ('healthy', 'stroke'):
        k = coh == c
        res[c] = {'n': int(k.sum()), 'recomputed_gdi': round(float(G[k].mean()), 2),
                  'baseline': round(float(Bm[k].mean()), 2), 'added': round(float(Am[k].mean()), 2),
                  'bias_baseline': round(float((Bm - G)[k].mean()), 2), 'bias_added': round(float((Am - G)[k].mean()), 2),
                  'mae_baseline': round(float(eb[k].mean()), 2), 'mae_added': round(float(ea[k].mean()), 2)}
        if k.sum() > 5:
            res[c]['corr_baseline'] = corr(Bm[k], G[k])
            res[c]['corr_added'] = corr(Am[k], G[k])
    res['healthy_minus_stroke'] = {'recomputed': round(float(G[coh == 'healthy'].mean() - G[coh == 'stroke'].mean()), 2),
                                   'baseline': round(float(Bm[coh == 'healthy'].mean() - Bm[coh == 'stroke'].mean()), 2),
                                   'added': round(float(Am[coh == 'healthy'].mean() - Am[coh == 'stroke'].mean()), 2)}
    out[ref] = res
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False))
