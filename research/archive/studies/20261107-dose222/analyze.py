"""Baseline versus small addition on the 222 people, same metrics as final222."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/dose222-20261107'); FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102')
OLD = ROOT / 'gdi-research-harness/studies/20261102-final223-frozen'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pid = sha(HERE / 'protocol.json')
for f in ('CNN', 'STT', 'bilateral'): assert read(HERE / f'infer-{f}.json')['protocol_sha256'] == pid
ens = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
old = np.load(FINAL / 'predictions.npz'); rows = list(csv.DictReader((OLD / 'scores.csv').open(encoding='utf-8-sig')))
assert [r['video_id'] for r in rows] == old['video_id'].astype(str).tolist()
ids = np.array([r['person_id'] for r in rows]); y = np.array([[float(r['true_left']), float(r['true_right'])] for r in rows])
P = {f: np.load(BIG / f'pred-{f}.npz') for f in ('CNN', 'STT', 'bilateral')}
def tags(fam, f): return sorted(k for k in P[fam].files if k.startswith(f'{f}__') and not k.endswith('__excluded'))
def refit(fam, f): return P[fam][f'{f}__refit']
def bag(fam, f): return np.mean([P[fam][k] for k in tags(fam, f) if not bool(P[fam][k + '__excluded'])], axis=0)
def fusion(f):
    w, t = ens[f]['cnn_weight'], ens[f]['fraction']
    b = w * bag('CNN', f) + (1 - w) * bag('STT', f); s = refit('bilateral', f)
    m, d = b.mean(1), (1 - t) * (b[:, 0] - b[:, 1]) + t * (s[:, 0] - s[:, 1])
    return np.stack([m + d / 2, m - d / 2], axis=1)
def mean5(fn): return np.mean([fn(f) for f in range(5)], axis=0)
base = {'CNN': old['CNN_refit'], 'STT': mean5(lambda f: old[f'STT_{f}_refit']), 'bilateral_STT': mean5(lambda f: old[f'bilateral_{f}']),
        'CNN_ensemble': old['CNN_bag'], 'STT_ensemble': mean5(lambda f: old[f'STT_{f}_bag']), 'fusion': old['fusion']}
small = {'CNN': mean5(lambda f: refit('CNN', f)), 'STT': mean5(lambda f: refit('STT', f)), 'bilateral_STT': mean5(lambda f: refit('bilateral', f)),
         'CNN_ensemble': mean5(lambda f: bag('CNN', f)), 'STT_ensemble': mean5(lambda f: bag('STT', f)), 'fusion': mean5(fusion)}
BANDS = {'all': np.ones_like(y, bool), 'lt70': y < 70, 'ge90': y >= 90, 'ge100': y >= 100}
def stats(p, mask):
    q = np.unique(ids[mask.any(1)]); e = p - y
    return np.array([np.abs(e[mask & (ids == v)[:, None]]).mean() for v in q]), np.array([e[mask & (ids == v)[:, None]].mean() for v in q])
large = read(ROOT / 'studies/20261106-final222-normal100/results.json')['models']
R = dict(protocol_sha256=pid, people=222, records=377, models={})
for m in base:
    R['models'][m] = {}
    for b, mask in BANDS.items():
        bm, bb = stats(base[m], mask); sm, sb = stats(small[m], mask); d = sm - bm; n = len(d)
        draws = np.random.default_rng(20261106).integers(0, n, (10000, n))
        R['models'][m][b] = dict(people=n, baseline_mae=float(bm.mean()), small_mae=float(sm.mean()), baseline_bias=float(bb.mean()), small_bias=float(sb.mean()),
            delta_mae=float(d.mean()), delta_ci95=[float(v) for v in np.quantile(d[draws].mean(1), [.025, .975])],
            large_measured_mae=large[m][b]['mixed_stanford_cal']['mae'], large_mixed_mae=large[m][b]['mixed']['mae'])
Y = y.ravel(); R['correlation'] = dict(baseline=float(np.corrcoef(Y, base['fusion'].ravel())[0, 1]), small=float(np.corrcoef(Y, small['fusion'].ravel())[0, 1]))
assert abs(R['models']['fusion']['all']['baseline_mae'] - 6.248559) < 1e-5
(HERE / 'results.json').write_text(json.dumps(R, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
for m, v in R['models'].items():
    print(m, {b: (round(a['baseline_mae'], 3), round(a['small_mae'], 3), round(a['delta_mae'], 3), [round(c, 3) for c in a['delta_ci95']], round(a['large_measured_mae'], 3)) for b, a in v.items() if b in ('all', 'ge100')})
print(R['models']['fusion']); print(R['correlation'])
