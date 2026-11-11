"""Post-hoc, not preregistered: is the larger gap new separation or a stretched scale? Uses saved results only."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
spec = {'__file__': str(HERE / 'analyze.py')}
exec(open(HERE / 'analyze.py', encoding='utf-8').read().split("rng = np.random.default_rng")[0], spec)
tdp = spec['tdp']
cp = R['groups']['CP']['per_person']
itw = R['groups']['ITW']['per_person']
b = {'CP': np.array([v['before'] for v in cp.values()]), 'ITW': np.array([v['before'] for v in itw.values()]), 'TD': tdp['before']}
a = {'CP': np.array([v['after_measured'] for v in cp.values()]), 'ITW': np.array([v['after_measured'] for v in itw.values()]), 'TD': tdp['after_measured']}


def auc(pos, neg):
    return float(np.mean([(p > n) + .5 * (p == n) for p in pos for n in neg]))


def cohen(x, y):
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / s)


out = {}
for g in ('CP', 'ITW'):
    out[g] = dict(before=dict(gap=float(b['TD'].mean() - b[g].mean()), auc_td_over_group=auc(b['TD'], b[g]), d=cohen(b['TD'], b[g])),
                  after=dict(gap=float(a['TD'].mean() - a[g].mean()), auc_td_over_group=auc(a['TD'], a[g]), d=cohen(a['TD'], a[g])))
allb = np.concatenate([b[k] for k in b])
alla = np.concatenate([a[k] for k in a])
fit = np.polyfit(allb, alla, 2)
resid = alla - np.polyval(fit, allb)
out['after_vs_before_all_people'] = dict(people=len(allb), pearson=float(np.corrcoef(allb, alla)[0, 1]), spearman=float(np.corrcoef(np.argsort(np.argsort(allb)), np.argsort(np.argsort(alla)))[0, 1]),
                                         rise_vs_before_pearson=float(np.corrcoef(allb, alla - allb)[0, 1]), quadratic_residual_sd=float(resid.std(ddof=3)))
k = 0
for g in ('CP', 'ITW', 'TD'):
    n = len(b[g])
    out['after_vs_before_all_people'][g + '_mean_residual_from_common_curve'] = float(resid[k:k + n].mean())
    k += n
(HERE / 'posthoc-interpret.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1))
