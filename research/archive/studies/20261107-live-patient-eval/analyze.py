"""Frozen comparison of patient and typically developing children. Runs only after inference is saved."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


proto = read(HERE / 'protocol.json')
meta = read(HERE / 'preparation.json')
assert sha(HERE / 'protocol.json') == meta['protocol_sha256']
ens = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/ensemble.json')
oldmeta = read(ROOT / 'studies/20261105-unseen-normal-test/preparation.json')
gate = read(ROOT / 'studies/20261105-normal-transfer-followup/source-gate-results.json')
FAMS = ('CNN', 'STT', 'bilateral')
RAW = {f: np.load(BIG / f'raw-{f}.npz') for f in FAMS}
TDRAW = {f: np.load(OLD / f'member-raw-{f}.npz') for f in FAMS}
TDP = {f: np.load(OLD / f'pred-{f}.npz') for f in FAMS}
with np.load(SRC / 'data.npz') as z:
    y, source = z['y'], z['source']
stan = source == 'Stanford'


def coef(p, t):
    return np.linalg.lstsq(np.column_stack([p.reshape(-1), np.ones(p.size)]), t.reshape(-1), rcond=None)[0]


def tags(f, arm, fold):
    pre = f'{arm}__{fold}__'
    return sorted(k[len(pre):-5] for k in RAW[f].files if k.startswith(pre) and k.endswith('__raw'))


def stan_coef(f, fold, tag):
    root = SRC / 'received/data/runs/mixed' / f / f'fold{fold}'
    with np.load(root / tag / 'replay.npz') as z:
        tr, tp = z['train_row'], z['train_raw']
    k = stan[tr]
    return coef(tp[k], y[tr][k])


CACHE = {}


def scored(f, arm, fold, tag, cal, patient=True):
    """cal is 'stored' (the member's own calibration) or 'measured'."""
    key = (f, fold, tag)
    if cal == 'measured':
        assert arm == 'mixed'
        if key not in CACHE:
            CACHE[key] = stan_coef(f, fold, tag)
        a, b = CACHE[key]
    else:
        a, b = RAW[f][f'{arm}__{fold}__{tag}__cal']
    raw = RAW[f][f'{arm}__{fold}__{tag}__raw'] if patient else None
    return raw * a + b if patient else TDRAW[f][f'live__{fold}__{tag}'] * a + b


def bag(f, arm, fold, cal, patient=True):
    keep = [t for t in tags(f, arm, fold) if not bool(RAW[f][f'{arm}__{fold}__{t}__excluded'])]
    return np.mean([scored(f, arm, fold, t, cal, patient) for t in keep], axis=0)


def td_stored(f, arm, fold, kind):
    return TDP[f][f'live__{arm}__{fold}__{kind}']


def components(arm, fold, cal, patient):
    if not patient and cal == 'stored':
        return td_stored('CNN', arm, fold, 'bag'), td_stored('STT', arm, fold, 'bag'), td_stored('bilateral', arm, fold, 'refit')
    return bag('CNN', arm, fold, cal, patient), bag('STT', arm, fold, cal, patient), scored('bilateral', arm, fold, 'refit', cal, patient)


def fusion(arm, cal, patient):
    out = []
    for fold in range(5):
        w, t = ens[fold]['cnn_weight'], ens[fold]['fraction']
        cnn, stt, s = components(arm, fold, cal, patient)
        b = w * cnn + (1 - w) * stt
        m, d = b.mean(1), (1 - t) * (b[:, 0] - b[:, 1]) + t * (s[:, 0] - s[:, 1])
        out.append(np.stack([m + d / 2, m - d / 2], axis=1))
    return np.mean(out, axis=0).mean(1)


recs = meta['records']
pid = np.array([r['group'] + ':' + r['person'] for r in recs])
grp = np.array([r['group'] for r in recs])
tdid = np.array(oldmeta['live_people'])
ARMS = {'before': ('baseline', 'stored'), 'after_mixed': ('mixed', 'stored'), 'after_measured': ('mixed', 'measured')}
PR = {a: fusion(*v, True) for a, v in ARMS.items()}
TD = {a: fusion(*v, False) for a, v in ARMS.items()}


def people(v, ids):
    q = sorted(set(ids.tolist()))
    return np.array([v[ids == i].mean() for i in q]), q


def ci(v, draws):
    return [float(x) for x in np.quantile(v[draws].mean(1), [.025, .975])]


tdp = {a: people(v, tdid)[0] for a, v in TD.items()}
assert abs(tdp['before'].mean() - 81.76) < .01 and abs(tdp['after_measured'].mean() - 98.83) < .01, (tdp['before'].mean(), tdp['after_measured'].mean())
assert abs(tdp['after_mixed'].mean() - 95.49) < .01, tdp['after_mixed'].mean()
rng = np.random.default_rng(proto.get('seed', 20261107))
B = 10000
tddraw = rng.integers(0, len(tdp['before']), (B, len(tdp['before'])))
R = dict(protocol_sha256=meta['protocol_sha256'], review_sha256=meta['review_sha256'],
         checks=dict(td_before=float(tdp['before'].mean()), td_after_mixed=float(tdp['after_mixed'].mean()), td_after_measured=float(tdp['after_measured'].mean()),
                     td_replay={f: read(HERE / f'infer-{f}.json')['td_replay_max_error'] for f in FAMS}))
R['TD'] = dict(people=len(tdp['before']), records=12, **{a: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), ci95=ci(v, tddraw)) for a, v in tdp.items()},
               rise=dict(mean=float((tdp['after_measured'] - tdp['before']).mean()), ci95=ci(tdp['after_measured'] - tdp['before'], tddraw)))
R['groups'] = {}
for g in ('CP', 'ITW'):
    k = grp == g
    p = {a: people(v[k], pid[k])[0] for a, v in PR.items()}
    q = people(PR['before'][k], pid[k])[1]
    n = len(q)
    draw = rng.integers(0, n, (B, n))
    rise = p['after_measured'] - p['before']
    tdrise = tdp['after_measured'] - tdp['before']
    diff = rise[draw].mean(1) - tdrise[tddraw].mean(1)
    level = p['after_measured'][draw].mean(1) - tdp['after_measured'][tddraw].mean(1)
    dci = [float(x) for x in np.quantile(diff, [.025, .975])]
    lci = [float(x) for x in np.quantile(level, [.025, .975])]
    m = float(p['after_measured'].mean())
    td_m = float(tdp['after_measured'].mean())
    if dci[1] < 0 and m < td_m:
        verdict = 'supports_gait_response'
    elif dci[0] <= 0 <= dci[1] and m >= 90:
        verdict = 'supports_dataset_response'
    else:
        verdict = 'inconclusive'
    R['groups'][g] = dict(people=n, records=int(k.sum()), **{a: dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()), ci95=ci(v, draw)) for a, v in p.items()},
                          rise=dict(mean=float(rise.mean()), sd=float(rise.std(ddof=1)), ci95=ci(rise, draw), min=float(rise.min()), max=float(rise.max())),
                          rise_minus_TD=dict(mean=float(rise.mean() - tdrise.mean()), ci95=dci),
                          after_measured_minus_TD=dict(mean=float(m - td_m), ci95=lci), verdict=verdict,
                          per_person={i: dict(before=round(float(a), 2), after_measured=round(float(b), 2)) for i, a, b in zip(q, p['before'], p['after_measured'])})


def feats(x, owner, n):
    w = np.concatenate([x.mean(2), x.std(2), np.ptp(x, axis=2), np.abs(np.diff(x, axis=2)).mean(2)], axis=2).mean(1)
    f = np.zeros((n, 48))
    np.add.at(f, owner, w)
    return f / np.bincount(owner, minlength=n)[:, None]


with np.load(BIG / 'inputs.npz') as z:
    F = feats(z['x'].astype(np.float64), z['owner'], len(recs))
prob = []
for m in gate['models']:
    logit = ((F - np.array(m['mean'])) / np.array(m['scale'])) @ np.array(m['coef']) + m['intercept']
    prob.append(1 / (1 + np.exp(-logit)))
prob = np.array(prob)
G = dict(share_folds_ge_half=float((prob >= .5).mean()), records_any_fold_ge_half=int((prob >= .5).any(0).sum()), mean_probability=float(prob.mean()),
         max_record_mean_probability=float(prob.mean(0).max()), by_group={})
for f in ('CNN', 'STT'):
    base = np.array([scored(f, 'baseline', fold, 'refit', 'stored') for fold in range(5)])
    after = np.array([scored(f, 'mixed', fold, 'refit', 'measured') for fold in range(5)])
    gated = np.where(prob[:, :, None] >= .5, 100., base).mean(0).mean(1)
    for g in ('CP', 'ITW'):
        k = grp == g
        G['by_group'].setdefault(g, {})[f] = dict(
            before=float(people(base.mean(0).mean(1)[k], pid[k])[0].mean()),
            gate=float(people(gated[k], pid[k])[0].mean()),
            after_measured=float(people(after.mean(0).mean(1)[k], pid[k])[0].mean()))
R['source_control'] = G
device = np.array([r['assistive_device'] is not None for r in recs])
R['records_with_visible_device'] = [r['record'] for r in recs if r['assistive_device']]
(HERE / 'results.json').write_text(json.dumps(R, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in R.items() if k not in ('groups',)}, indent=0)[:1800])
for g, v in R['groups'].items():
    print(g, {k: v[k] for k in ('people', 'records', 'verdict', 'rise', 'rise_minus_TD', 'after_measured_minus_TD')})
    print({a: (round(v[a]['mean'], 2), [round(c, 2) for c in v[a]['ci95']], round(v[a]['min'], 1), round(v[a]['max'], 1)) for a in ARMS})
    print(v['per_person'])
