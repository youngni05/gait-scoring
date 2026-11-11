"""Frozen comparison on the 222 people. Reads outcomes only here, after all inference is saved."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/final222-normal100-20261106')
FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = ROOT / 'gdi-research-harness/studies/20261102-final223-frozen'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


proto = read(HERE / 'protocol.json')
pid = sha(HERE / 'protocol.json')
for fam in ('CNN', 'STT', 'bilateral', 'RF'):
    info = read(HERE / f'infer-{fam}.json')
    assert info['protocol_sha256'] == pid
ens = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in ('y', 'person_id', 'source')}
stan = d['source'] == 'Stanford'
RAW = {f: np.load(BIG / f'member-raw-{f}.npz') for f in ('CNN', 'STT', 'bilateral')}
rf = np.load(BIG / 'rf.npz')
old = np.load(FINAL / 'predictions.npz')
rows = list(csv.DictReader((OLD / 'scores.csv').open(encoding='utf-8-sig')))
videos = old['video_id'].astype(str)
assert [r['video_id'] for r in rows] == videos.tolist()
ids = np.array([r['person_id'] for r in rows])
assert ids.tolist() == old['person_id'].astype(str).tolist()
y = np.array([[float(r['true_left']), float(r['true_right'])] for r in rows])
stored = np.array([[float(r['fusion_left']), float(r['fusion_right'])] for r in rows])
table_error = float(np.abs(stored - old['fusion']).max())
assert table_error < 1e-9, table_error


def coef(p, t):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1), np.ones(p.size)]), t.reshape(-1), rcond=None)[0]


members = {}
for fam in RAW:
    for f in range(5):
        root = SRC / 'received/data/runs/mixed' / fam / f'fold{f}'
        items = []
        for m in read(root / 'complete.json')['members']:
            with np.load(root / m['tag'] / 'replay.npz') as z:
                tr, tp = z['train_row'], z['train_raw']
            keep = stan[tr]
            items.append(dict(tag=m['tag'], excluded=m['excluded_from_bag'], mixed=np.array(m['calibration']),
                              stan=coef(tp[keep], d['y'][tr][keep]), raw=RAW[fam][f"{f}__{m['tag']}"]))
        members[fam, f] = items


def scored(m, cal):
    return m['raw'] * m[cal][0] + m[cal][1]


def bag(fam, f, cal):
    return np.mean([scored(m, cal) for m in members[fam, f] if not m['excluded']], axis=0)


def refit(fam, f, cal):
    return scored(next(m for m in members[fam, f] if m['tag'] == 'refit'), cal)


def fusion(f, cal):
    w, t = ens[f]['cnn_weight'], ens[f]['fraction']
    b = w * bag('CNN', f, cal) + (1 - w) * bag('STT', f, cal)
    s = refit('bilateral', f, cal)
    m, delta = b.mean(1), (1 - t) * (b[:, 0] - b[:, 1]) + t * (s[:, 0] - s[:, 1])
    return np.stack([m + delta / 2, m - delta / 2], axis=1)


def over_folds(fn, *args):
    return np.mean([fn(*args[:1], f, *args[1:]) for f in range(5)], axis=0)


def mean5(fn):
    return np.mean([fn(f) for f in range(5)], axis=0)


base = {
    'CNN': old['CNN_refit'], 'STT': mean5(lambda f: old[f'STT_{f}_refit']),
    'bilateral_STT': mean5(lambda f: old[f'bilateral_{f}']), 'CNN_ensemble': old['CNN_bag'],
    'STT_ensemble': mean5(lambda f: old[f'STT_{f}_bag']), 'fusion': old['fusion'], 'RF': rf['baseline'],
}


def arm(cal):
    return {
        'CNN': mean5(lambda f: refit('CNN', f, cal)), 'STT': mean5(lambda f: refit('STT', f, cal)),
        'bilateral_STT': mean5(lambda f: refit('bilateral', f, cal)), 'CNN_ensemble': mean5(lambda f: bag('CNN', f, cal)),
        'STT_ensemble': mean5(lambda f: bag('STT', f, cal)), 'fusion': mean5(lambda f: fusion(f, cal)),
    }


mixed, stancal = arm('mixed'), arm('stan')
mixed['RF'] = rf['mixed']
ARMS = {'baseline': base, 'mixed': mixed, 'mixed_stanford_cal': stancal}
BANDS = {'all': np.ones_like(y, bool), 'lt50': y < 50, 'lt70': y < 70, 'ge90': y >= 90, 'ge100': y >= 100}


def person_stats(p, mask):
    people = np.unique(ids[mask.any(axis=1)])
    e = p - y
    return (people, np.array([np.abs(e[mask & (ids == q)[:, None]]).mean() for q in people]),
            np.array([e[mask & (ids == q)[:, None]].mean() for q in people]))


def compare(model):
    out = {}
    for band, mask in BANDS.items():
        people, b_mae, b_bias = person_stats(base[model], mask)
        n = len(people)
        draws = np.random.default_rng(20261106).integers(0, n, (10000, n))
        row = dict(people=n, records=int(mask.any(axis=1).sum()), side_observations=int(mask.sum()))
        for name, preds in ARMS.items():
            if model not in preds:
                continue
            _, mae, bias = person_stats(preds[model], mask)
            dm = mae - b_mae
            row[name] = dict(mae=float(mae.mean()), bias=float(bias.mean()), delta_mae=float(dm.mean()),
                             delta_mae_ci95=[float(v) for v in np.quantile(dm[draws].mean(1), [.025, .975])],
                             improved_people=int((dm < 0).sum()), worsened_people=int((dm > 0).sum()),
                             all_under=bool((preds[model][mask] < y[mask]).all()))
        out[band] = row
    return out


baseline_mae = float(person_stats(base['fusion'], BANDS['all'])[1].mean())
assert abs(baseline_mae - 6.248559) < 1e-5, baseline_mae
R = dict(protocol_sha256=pid, people=222, records=377, checks=dict(
    baseline_fusion_mae=baseline_mae, stored_table_max_error=table_error,
    replay_max_error={f: read(HERE / f'infer-{f}.json')['replay_max_error'] for f in RAW},
    checkpoints={f: read(HERE / f'infer-{f}.json')['checkpoints_hash_checked'] for f in RAW}))
R['models'] = {m: compare(m) for m in base}
Y = y.ravel()
R['fusion_association'] = {a: dict(correlation=float(np.corrcoef(Y, p['fusion'].ravel())[0, 1]),
                                   slope=float(np.polyfit(Y, p['fusion'].ravel(), 1)[0]),
                                   prediction_sd=float(p['fusion'].std())) for a, p in ARMS.items()}
(HERE / 'results.json').write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')
np.savez_compressed(BIG / 'fusion-predictions.npz', person_id=ids, video_id=videos, y=y,
                    baseline=base['fusion'], mixed=mixed['fusion'], mixed_stanford_cal=stancal['fusion'])

# Independent recomputation of the primary numbers with observation weights.
check = {}
for name in ARMS:
    e = np.abs(ARMS[name]['fusion'] - y)
    counts = {q: (ids == q).sum() * 2 for q in set(ids.tolist())}
    w = np.array([1 / (222 * counts[q]) for q in ids])
    check[name] = float((e.sum(1) * w).sum())
    assert abs(check[name] - R['models']['fusion']['all'][name]['mae']) < 1e-10
R['independent_primary_mae'] = check
(HERE / 'results.json').write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')
for m in R['models']:
    a = R['models'][m]['all']
    h = R['models'][m]['ge100']
    print(m, {k: (round(a[k]['mae'], 3), round(h[k]['mae'], 3)) for k in ARMS if k in a})
print(json.dumps(R['models']['fusion'], indent=0, ensure_ascii=False)[:3200])
print(json.dumps(R['fusion_association']), json.dumps(R['checks']))
