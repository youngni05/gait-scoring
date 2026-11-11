"""One fixed threshold on the external sets: baseline vs normal-added. Stored scores only."""
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
sets = {
    'LIVE': {'normal': (sep['before']['td'], sep['after_measured']['td']), 'abnormal': (sep['before']['cp'], sep['after_measured']['cp'])},
    'self': {'normal': ([vb[k] for k in nk], [va[k] for k in nk]), 'abnormal': ([vb[k] for k in ak], [va[k] for k in ak])},
    'older': {'normal': ([p['baseline'] for p in h], [p['added'] for p in h]), 'abnormal': ([p['baseline'] for p in s], [p['added'] for p in s])},
}
sizes = {k: (len(v['normal'][0]), len(v['abnormal'][0])) for k, v in sets.items()}
assert sizes == {'LIVE': (9, 14), 'self': (6, 7), 'older': (11, 25)}, sizes
rng = np.random.default_rng(20261110)
B = 10000


def metrics(n, a, t):
    spec = float((n >= t).mean())
    sens = float((a < t).mean())
    return spec, sens, (spec + sens) / 2


out = {}
for t in (90, 85, 95):
    res = {}
    pooled_n = {m: [] for m in (0, 1)}
    pooled_a = {m: [] for m in (0, 1)}
    groups = list(sets) + ['pooled']
    for name in groups:
        if name == 'pooled':
            N = [np.concatenate(pooled_n[m]) for m in (0, 1)]
            A = [np.concatenate(pooled_a[m]) for m in (0, 1)]
        else:
            N = [np.asarray(sets[name]['normal'][m]) for m in (0, 1)]
            A = [np.asarray(sets[name]['abnormal'][m]) for m in (0, 1)]
            for m in (0, 1):
                pooled_n[m].append(N[m])
                pooled_a[m].append(A[m])
        i = rng.integers(0, len(N[0]), (B, len(N[0])))
        j = rng.integers(0, len(A[0]), (B, len(A[0])))
        ba = [np.array([(N[m][x] >= t).mean() / 2 + (A[m][y] < t).mean() / 2 for x, y in zip(i, j)]) for m in (0, 1)]
        q = lambda v: [round(float(u), 3) for u in np.quantile(v, [.025, .975])]
        row = {}
        for m, label in ((0, 'baseline'), (1, 'normal_added')):
            spec, sens, bal = metrics(N[m], A[m], t)
            row[label] = {'normal_called_normal': round(spec, 3), 'abnormal_called_abnormal': round(sens, 3),
                          'balanced_accuracy': round(bal, 3), 'balanced_accuracy_ci95': q(ba[m])}
        row['difference_ci95'] = q(ba[1] - ba[0])
        row['n'] = [len(N[0]), len(A[0])]
        res[name] = row
    out[f'threshold_{t}'] = res
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
for t, res in out.items():
    print(t)
    for k, v in res.items():
        print(' ', k, v['n'], 'base', v['baseline']['normal_called_normal'], v['baseline']['abnormal_called_abnormal'], v['baseline']['balanced_accuracy'],
              '| added', v['normal_added']['normal_called_normal'], v['normal_added']['abnormal_called_abnormal'], v['normal_added']['balanced_accuracy'], v['normal_added']['balanced_accuracy_ci95'], '| diff', v['difference_ci95'])
