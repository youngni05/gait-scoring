"""Summaries of retrained arms (dose-seed and finetune) with the paper's measured-value calibration.

usage: python analyze_arms.py <received runs root> <arm or '.'> [<arm> ...]
A run root holds <arm>/<family>/fold<f>/{complete.json, inner*/replay.npz, refit/replay.npz}; for the finetune
study pass the root of the families and arm '.'. Prints and returns one summary per arm."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path('D:/Projects/D_P')
ST = ROOT / 'studies'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


ENS = read(ST / '20261104-ljmu100-mixed/bundle/ensemble.json')

# evaluation labels and paper baseline, same order as external.npz eval records
old = np.load('D:/gait-research-data/D_P/final223-frozen-20261102/predictions.npz')
feat = np.load('D:/gait-research-data/D_P/final223-frozen-20261102/features.npz')
rows = list(csv.DictReader((ROOT / 'gdi-research-harness/studies/20261102-final223-frozen/scores.csv').open(encoding='utf-8-sig')))
assert [r['video_id'] for r in rows] == old['video_id'].astype(str).tolist() == feat['video_id'].astype(str).tolist()
EVAL_IDS = np.array([r['person_id'] for r in rows])
EVAL_Y = np.array([[float(r['true_left']), float(r['true_right'])] for r in rows])
EVAL_BASE = old['fusion']

# external unit definitions
UN = read(ST / '20261105-unseen-normal-test/preparation.json')
LIVETD_PEOPLE = np.array(UN['live_people'])
SELF_NORMAL = np.isin(np.array(UN['self_numbers']), UN['self_normal'])
LP = read(ST / '20261107-live-patient-eval/preparation.json')['records']
LP_GROUP = np.array([r['group'] for r in LP])
LP_PERSON = np.array([r['group'] + ':' + r['person'] for r in LP])
FZ = ST / '20261103-frozen-transfer-and-deviation'
OLD_ROWS = read(FZ / 'rows.json')
OLD_OWNER = np.load('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103/features.npz')['owner']
OLD_PEOPLE = {p['id']: p for p in read(ST / '20261106-older-normal100/people-scores.json')}


def coef(p, t):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1), np.ones(p.size)]), t.reshape(-1), rcond=None)[0]


def members(root, fam, f):
    done = read(root / fam / f'fold{f}' / 'complete.json')
    out = []
    for m in done['members']:
        with np.load(root / fam / f'fold{f}' / m['tag'] / 'replay.npz') as z:
            k = z['train_source'] == 'Stanford'
            a, b = coef(z['train_raw'][k], z['train_y'][k])
            out.append({'tag': m['tag'], 'excluded': m['excluded_from_bag'],
                        'ext': {n[5:]: z[n] * a + b for n in z.files if n.startswith('ext__')}})
    return out


def fused(root, f):
    M = {fam: members(root, fam, f) for fam in ('CNN', 'STT', 'bilateral')}
    w, t = ENS[f]['cnn_weight'], ENS[f]['fraction']
    out = {}
    for name in M['CNN'][0]['ext']:
        bag = lambda fam: np.mean([m['ext'][name] for m in M[fam] if not m['excluded']], axis=0)
        b = w * bag('CNN') + (1 - w) * bag('STT')
        s = next(m for m in M['bilateral'] if m['tag'] == 'refit')['ext'][name]
        mid, d = b.mean(1), (1 - t) * (b[:, 0] - b[:, 1]) + t * (s[:, 0] - s[:, 1])
        out[name] = np.stack([mid + d / 2, mid - d / 2], axis=1)
    return out


def person_mean(values, ids):
    q = sorted(set(ids.tolist()))
    return np.array([values[ids == p].mean() for p in q]), q


def older_people(window_scores):
    d = {}
    for r, row in enumerate(OLD_ROWS):
        ix = np.flatnonzero(OLD_OWNER == r)
        if len(ix) and row['primary']:
            d.setdefault(row['person'], []).append(float(window_scores[ix].mean()))
    return {k: float(np.mean(v)) for k, v in d.items()}


def summarise(root):
    F = [fused(root, f) for f in range(5)]
    ext = {k: np.mean([F[f][k] for f in range(5)], axis=0) for k in F[0]}
    rng = np.random.default_rng(20261110)
    # evaluation
    q = sorted(set(EVAL_IDS.tolist()))
    pm = np.array([np.abs(ext['eval'] - EVAL_Y)[EVAL_IDS == p].mean() for p in q])
    bm = np.array([np.abs(EVAL_BASE - EVAL_Y)[EVAL_IDS == p].mean() for p in q])
    draw = rng.integers(0, len(q), (10000, len(q)))
    res = {'eval_mae': round(float(pm.mean()), 3), 'eval_delta_mae': round(float((pm - bm).mean()), 3),
           'eval_delta_mae_ci95': [round(float(v), 3) for v in np.quantile((pm - bm)[draw].mean(1), [.025, .975])],
           'eval_corr': round(float(np.corrcoef(ext['eval'].ravel(), EVAL_Y.ravel())[0, 1]), 3)}
    # externals
    td, _ = person_mean(ext['livetd'].mean(1), LIVETD_PEOPLE)
    cp, _ = person_mean(ext['livepat'].mean(1)[LP_GROUP == 'CP'], LP_PERSON[LP_GROUP == 'CP'])
    sv = ext['self'].mean(1)
    op = older_people(ext['older'])
    h = np.array([v for k, v in op.items() if OLD_PEOPLE[k]['cohort'] == 'healthy' and OLD_PEOPLE[k]['age'] >= 65])
    s = np.array([v for k, v in op.items() if OLD_PEOPLE[k]['cohort'] == 'stroke' and OLD_PEOPLE[k]['age'] >= 65])
    assert len(td) == 9 and len(cp) == 14 and len(h) == 11 and len(s) == 25
    res['means'] = {'LIVE_TD': round(float(td.mean()), 2), 'LIVE_CP': round(float(cp.mean()), 2),
                    'self_normal': round(float(sv[SELF_NORMAL].mean()), 2), 'self_abnormal': round(float(sv[~SELF_NORMAL].mean()), 2),
                    'older_healthy': round(float(h.mean()), 2), 'older_stroke': round(float(s.mean()), 2)}
    m = res['means']
    res['gaps'] = {'LIVE': round(m['LIVE_TD'] - m['LIVE_CP'], 2), 'self': round(m['self_normal'] - m['self_abnormal'], 2),
                   'older': round(m['older_healthy'] - m['older_stroke'], 2)}
    return res


if __name__ == '__main__':
    base = Path(sys.argv[1])
    out = {}
    for arm in sys.argv[2:]:
        out[arm] = summarise(base / arm if arm != '.' else base)
        print(arm, json.dumps(out[arm], ensure_ascii=False), flush=True)
