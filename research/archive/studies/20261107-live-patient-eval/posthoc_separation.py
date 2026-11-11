"""Post hoc: how well do person scores separate LIVE typically developing children from children with cerebral palsy?
AUC of the CNN-STT ensemble score, before and after normal addition (measured-value calibration). Reuses the data
loading of analyze.py without running its reporting part. Writes posthoc-separation.json only."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / 'analyze.py').read_text(encoding='utf-8')
head = src.split('rng = np.random.default_rng')[0]
ns = {'__file__': str(HERE / 'analyze.py'), '__name__': 'live_head'}
exec(compile(head, 'analyze_head', 'exec'), ns)
assert Path(ns['HERE']) == HERE
PR, TD, pid, grp, tdid, people = ns['PR'], ns['TD'], ns['pid'], ns['grp'], ns['tdid'], ns['people']


def auc(pos, neg):
    """Probability that a random TD child scores above a random CP child (ties count half)."""
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + .5 * (d == 0).mean())


rng = np.random.default_rng(20261110)
B = 10000
out = {}
k = grp == 'CP'
for arm in ('before', 'after_measured'):
    td = people(TD[arm], tdid)[0]
    cp = people(PR[arm][k], pid[k])[0]
    dt = rng.integers(0, len(td), (B, len(td)))
    dc = rng.integers(0, len(cp), (B, len(cp)))
    boot = [auc(td[a], cp[b]) for a, b in zip(dt, dc)]
    out[arm] = dict(auc=round(auc(td, cp), 3), auc_ci95=[round(float(v), 3) for v in np.quantile(boot, [.025, .975])],
                    td=[round(float(v), 2) for v in td], cp=[round(float(v), 2) for v in cp],
                    gap=round(float(td.mean() - cp.mean()), 2))
print(json.dumps({a: {x: v[x] for x in ('auc', 'auc_ci95', 'gap')} for a, v in out.items()}))
(HERE / 'posthoc-separation.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
