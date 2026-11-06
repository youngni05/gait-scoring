"""Replay gates first, then fixed summaries. No fitting on older-cohort data."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tmp/ljmu-cnn-python-deps'))
import numpy as np

BIG = Path('D:/gait-research-data/D_P/older-normal100-20261106')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
INP = Path('D:/gait-research-data/D_P/frozen-transfer-and-deviation-20261103')
UN = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
FZ = ROOT / 'studies/20261103-frozen-transfer-and-deviation'
UT = ROOT / 'studies/20261105-unseen-normal-test'
FAM = ['CNN', 'STT', 'bilateral']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


proto = read(HERE / 'protocol.json')
G = proto['replay_gates']
ens = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
base = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/baseline-models.json')
RAW = {f: np.load(BIG / f'raw-{f}.npz') for f in FAM}
EXT = {f: np.load(UN / f'member-raw-{f}.npz') for f in FAM}
frozen = np.load(INP / 'frozen-predictions.npz')
with np.load(SRC / 'data.npz') as z:
    y = z['y']
    stan = z['source'] == 'Stanford'


def coef(p, t):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1), np.ones(p.size)]), t.reshape(-1), rcond=None)[0]


M = {}
for fam in FAM:
    for f in range(5):
        root = SRC / 'received/data/runs/mixed' / fam / f'fold{f}'
        rows = []
        for m in read(root / 'complete.json')['members']:
            with np.load(root / m['tag'] / 'replay.npz') as z:
                tr = z['train_row']
                tp = z['train_raw']
            keep = stan[tr]
            rows.append(dict(tag=m['tag'], excluded=m['excluded_from_bag'], mixed=np.array(m['calibration']), stan=coef(tp[keep], y[tr][keep])))
        M[fam, f] = rows
B = {}
for m in base:
    B.setdefault((m['family'], m['fold']), []).append(dict(tag=m['tag'], excluded=False, cal=np.array(m['calibration'])))


def member_pred(fam, f, cal, arm, raw_of):
    if arm == 'baseline':
        return [(m, raw_of(fam, f, m['tag'], 'baseline') * m['cal'][0] + m['cal'][1]) for m in B[fam, f]]
    return [(m, raw_of(fam, f, m['tag'], 'mixed') * m[cal][0] + m[cal][1]) for m in M[fam, f]]


def fusion(f, arm, cal, raw_of):
    def bag(fam):
        return np.mean([p for m, p in member_pred(fam, f, cal, arm, raw_of) if not m['excluded']], axis=0)
    bil = next(p for m, p in member_pred('bilateral', f, cal, arm, raw_of) if m['tag'] == 'refit')
    w = ens[f]['cnn_weight']
    t = ens[f]['fraction']
    b = w * bag('CNN') + (1 - w) * bag('STT')
    common = b.mean(1)
    diff = (1 - t) * (b[:, 0] - b[:, 1]) + t * (bil[:, 0] - bil[:, 1])
    return np.stack([common + diff / 2, common - diff / 2], axis=1)


def older(fam, f, tag, arm):
    return RAW[fam][f'{arm}__{f}__{tag}']


def live(fam, f, tag, arm):
    return EXT[fam][f'live__{f}__{tag}']


# Replay gates: stored Stanford-only predictions and the published LIVE values.
member_err = 0.0
for fam in FAM:
    for f in range(5):
        for m, p in member_pred(fam, f, None, 'baseline', older):
            key = f'bilateral_{f}' if fam == 'bilateral' else f"{fam}_{f}_{m['tag']}"
            member_err = max(member_err, float(np.abs(p - frozen[key]).max()))
pred = {'baseline': np.mean([fusion(f, 'baseline', None, older) for f in range(5)], axis=0)}
for c in ['stan', 'mixed']:
    pred[c] = np.mean([fusion(f, 'mixed', c, older) for f in range(5)], axis=0)
fusion_err = float(np.abs(pred['baseline'] - frozen['fusion']).max())
livep = np.array(read(UT / 'preparation.json')['live_people'])


def live_mean(c):
    v = np.mean([fusion(f, 'mixed', c, live) for f in range(5)], axis=0).mean(1)
    return float(np.mean([v[livep == q].mean() for q in sorted(set(livep.tolist()))]))


live_mixed = live_mean('mixed')
live_stan = live_mean('stan')
live_err = abs(live_mixed - read(UT / 'results.json')['live']['mixed']['person_mean'])
checks = dict(member_max=member_err, fusion_max=fusion_err, live_mixed_value=live_mixed, live_mixed_error=live_err, live_stan_value=live_stan)
print(json.dumps(checks, indent=1))
limit = G['baseline_members_vs_saved_frozen_predictions_points']
assert member_err <= limit and fusion_err <= limit and live_err <= G['external_live_person_mean_vs_saved_points'], 'REPLAY_FAILED'

# Same aggregation as the 2026-11-03 frozen test.
rows = read(FZ / 'rows.json')
people = {p['id']: p for p in read(FZ / 'people.json')}
with np.load(INP / 'features.npz') as z:
    owner = z['owner']


def norm(i):
    m = re.fullmatch(r'TVC0*(\d+)', i)
    return f'TVC{m[1]}' if m else i


clin = {norm(p['id']): p for p in read(ROOT / 'studies/20261103-reference188-pilot/speed-control/participants.json')}


def person(p):
    d = {}
    for r, row in enumerate(rows):
        ix = np.flatnonzero(owner == r)
        if len(ix) and row['primary']:
            d.setdefault(row['person'], []).append(float(p[ix].mean(0).mean()))
    return {k: float(np.mean(v)) for k, v in d.items()}


def rank(a):
    a = np.asarray(a, float)
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    for v in np.unique(a):
        r[a == v] = r[a == v].mean()
    return r


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, s):
    D = np.column_stack([np.ones(len(s)), rank(s)])
    ra = rank(a)
    rb = rank(b)
    ra = ra - D @ np.linalg.lstsq(D, ra, rcond=None)[0]
    rb = rb - D @ np.linalg.lstsq(D, rb, rcond=None)[0]
    return corr(ra, rb)


def auc(h, s):
    return float((h[:, None] > s[None, :]).mean() + .5 * (h[:, None] == s[None, :]).mean())


def describe(v):
    return dict(n=len(v), mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)))


R = dict(replay=checks, arms={})
P = {a: person(v) for a, v in pred.items()}
for arm, v in P.items():
    H = np.array([x for k, x in v.items() if people[k]['cohort'] == 'healthy' and people[k]['age'] >= 65])
    S = np.array([x for k, x in v.items() if people[k]['cohort'] == 'stroke' and people[k]['age'] >= 65])
    Y = [x for k, x in v.items() if people[k]['cohort'] == 'healthy' and people[k]['age'] < 65]
    g = np.random.default_rng(20261106)
    diffs = []
    aucs = []
    for _ in range(10000):
        a = g.choice(H, len(H))
        b = g.choice(S, len(S))
        diffs.append(a.mean() - b.mean())
        aucs.append(auc(a, b))
    keys = sorted(k for k in v if k in clin and people[k]['age'] >= 65)
    x = np.array([v[k] for k in keys])
    poma = np.array([clin[k]['POMA'] for k in keys])
    speed = np.array([clin[k]['speed']['mean_mps'] for k in keys])
    R['arms'][arm] = dict(
        healthy65=describe(H), stroke65=describe(S), healthy_under65=describe(Y) if Y else None,
        difference=float(H.mean() - S.mean()), difference_ci95=np.quantile(diffs, [.025, .975]).tolist(),
        auc=auc(H, S), auc_ci95=np.quantile(aucs, [.025, .975]).tolist(),
        stroke_above_healthy_min=int((S > H.min()).sum()), healthy_below_stroke_max=int((H < S.max()).sum()),
        clinical_n=len(keys), POMA_spearman=corr(rank(x), rank(poma)), speed_spearman=corr(rank(x), rank(speed)),
        POMA_partial_speed=partial(x, poma, speed))
old_keys = [k for k in P['baseline'] if people[k]['age'] >= 65]
R['paired_rise_stan'] = {c: float(np.mean([P['stan'][k] - P['baseline'][k] for k in old_keys if people[k]['cohort'] == c])) for c in ['healthy', 'stroke']}
(HERE / 'results.json').write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding='utf-8')
for a, r in R['arms'].items():
    print(a, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
print(R['paired_rise_stan'])
