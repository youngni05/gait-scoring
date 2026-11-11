"""Tune a threshold on one external set, apply it to the others. Stored scores only."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


sep = read(ST / '20261107-live-patient-eval/posthoc-separation.json')
prep = read(ST / '20261105-unseen-normal-test/preparation.json')
normal_v = {str(v) for v in prep['self_normal']}
vb = read(ST / '20261105-unseen-normal-test/results.json')['self']['baseline']['videos']
va = read(ST / '20261105-unseen-normal-test/stanford-calibration/results.json')['external']['mixed_stanford_cal']['videos']
ppl = read(ST / '20261106-older-normal100/people-scores.json')
h = [p for p in ppl if p['cohort'] == 'healthy' and p['age'] >= 65]
s = [p for p in ppl if p['cohort'] == 'stroke' and p['age'] >= 65]
nk = sorted([k for k in vb if k in normal_v], key=int)
ak = sorted([k for k in vb if k not in normal_v], key=int)
RAW = {
    'LIVE': ((sep['before']['td'], sep['after_measured']['td']), (sep['before']['cp'], sep['after_measured']['cp'])),
    'self': (([vb[k] for k in nk], [va[k] for k in nk]), ([vb[k] for k in ak], [va[k] for k in ak])),
    'older': (([p['baseline'] for p in h], [p['added'] for p in h]), ([p['baseline'] for p in s], [p['added'] for p in s])),
}
SETS = list(RAW)
MODELS = ('baseline', 'normal_added')


def data(m):
    return {k: (np.asarray(v[0][m], float), np.asarray(v[1][m], float)) for k, v in RAW.items()}


def bal(n, a, t):
    return ((n >= t).mean() + (a < t).mean()) / 2


def tune(n, a):
    u = np.unique(np.concatenate([n, a]))
    cand = (u[:-1] + u[1:]) / 2
    sc = np.array([bal(n, a, t) for t in cand])
    best = cand[np.isclose(sc, sc.max())]
    return float(np.median(best)), float(sc.max())


def analyse(D):
    th = {k: tune(*D[k]) for k in SETS}
    pairs = {f'{a}->{b}': float(bal(*D[b], th[a][0])) for a in SETS for b in SETS if a != b}
    return th, pairs


rng = np.random.default_rng(20261110)
out = {}
point = {}
for mi, m in enumerate(MODELS):
    D = data(mi)
    th, pairs = analyse(D)
    point[m] = float(np.mean(list(pairs.values())))
    out[m] = {'tuned_threshold': {k: round(v[0], 2) for k, v in th.items()},
              'in_set_balanced_accuracy': {k: round(v[1], 3) for k, v in th.items()},
              'threshold_spread': round(max(v[0] for v in th.values()) - min(v[0] for v in th.values()), 2),
              'transferred_balanced_accuracy': {k: round(v, 3) for k, v in pairs.items()},
              'mean_transferred': round(point[m], 3),
              'fixed_90': {k: round(float(bal(*D[k], 90.)), 3) for k in SETS}}
diffs = []
for _ in range(2000):
    idx = {k: (rng.integers(0, len(RAW[k][0][0]), len(RAW[k][0][0])), rng.integers(0, len(RAW[k][1][0]), len(RAW[k][1][0]))) for k in SETS}
    means = []
    for mi in range(2):
        D = data(mi)
        Db = {k: (D[k][0][idx[k][0]], D[k][1][idx[k][1]]) for k in SETS}
        means.append(np.mean(list(analyse(Db)[1].values())))
    diffs.append(means[1] - means[0])
out['mean_transferred_difference'] = round(point['normal_added'] - point['baseline'], 3)
out['mean_transferred_difference_ci95'] = [round(float(v), 3) for v in np.quantile(diffs, [.025, .975])]
(HERE / 'results-transfer.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False))
