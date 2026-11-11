"""Bootstrap intervals for the self-captured and older external comparisons. No new inference."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parents[0]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


rng = np.random.default_rng(20261110)
B = 10000


def auc(a, b):
    d = a[:, None] - b[None, :]
    return float((d > 0).mean() + .5 * (d == 0).mean())


def compare(nb, na, ab, aa):
    nb, na, ab, aa = map(np.asarray, (nb, na, ab, aa))
    i = rng.integers(0, len(nb), (B, len(nb)))
    j = rng.integers(0, len(ab), (B, len(ab)))
    rise_diff = (na - nb)[i].mean(1) - (aa - ab)[j].mean(1)
    gap_before = nb[i].mean(1) - ab[j].mean(1)
    gap_after = na[i].mean(1) - aa[j].mean(1)
    aucs = np.array([auc(na[x], aa[y]) for x, y in zip(i[:2000], j[:2000])])
    aucb = np.array([auc(nb[x], ab[y]) for x, y in zip(i[:2000], j[:2000])])
    q = lambda v: [round(float(t), 3) for t in np.quantile(v, [.025, .975])]
    return dict(n_normal=len(nb), n_abnormal=len(ab),
                rise_normal=round(float((na - nb).mean()), 2), rise_abnormal=round(float((aa - ab).mean()), 2),
                rise_difference=round(float((na - nb).mean() - (aa - ab).mean()), 2), rise_difference_ci95=q(rise_diff),
                gap_before=round(float(nb.mean() - ab.mean()), 2), gap_before_ci95=q(gap_before),
                gap_after=round(float(na.mean() - aa.mean()), 2), gap_after_ci95=q(gap_after),
                gap_change_ci95=q(gap_after - gap_before),
                auc_before=round(auc(nb, ab), 3), auc_before_ci95=q(aucb), auc_after=round(auc(na, aa), 3), auc_after_ci95=q(aucs))


prep = read(ST / '20261105-unseen-normal-test/preparation.json')
normal = {str(v) for v in prep['self_normal']}
vb = read(ST / '20261105-unseen-normal-test/results.json')['self']['baseline']['videos']
va = read(ST / '20261105-unseen-normal-test/stanford-calibration/results.json')['external']['mixed_stanford_cal']['videos']
keys = sorted(vb, key=int)
nk = [k for k in keys if k in normal]
ak = [k for k in keys if k not in normal]
assert len(nk) == 6 and len(ak) == 7, (nk, ak)
out = {'self_video_level': compare([vb[k] for k in nk], [va[k] for k in nk], [vb[k] for k in ak], [va[k] for k in ak])}
ppl = read(ST / '20261106-older-normal100/people-scores.json')
h = [p for p in ppl if p['cohort'] == 'healthy' and p['age'] >= 65]
s = [p for p in ppl if p['cohort'] == 'stroke' and p['age'] >= 65]
out['older_person_level'] = compare([p['baseline'] for p in h], [p['added'] for p in h], [p['baseline'] for p in s], [p['added'] for p in s])
(HERE / 'results.json').write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(out, indent=1, ensure_ascii=False))
