"""Build cycle-normalised inputs and the LIVE ranking set. No model output is read here."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/model-improvement-20261109')
(BIG / 'data').mkdir(parents=True, exist_ok=True)
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
LIVE = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def cycle(x):
    """x: n,2,124,12. Returns cycle-normalised copy and a flag of windows that kept original frames."""
    out = x.copy()
    kept = np.zeros(len(x), bool)
    periods = np.zeros(len(x))
    for i, w in enumerate(x):
        s = w[0, :, 11] - w[0, :, 11].mean()
        den = (s * s).sum()
        if den <= 1e-8:
            kept[i] = True
            continue
        ac = np.array([(s[:-L] * s[L:]).sum() / den for L in range(15, 63)])
        T = 15 + int(ac.argmax())
        if ac.max() < .2 or 2 * T > 124:
            kept[i] = True
            continue
        periods[i] = T
        src = np.linspace(0, 2 * T - 1, 124)
        a = np.floor(src).astype(int)
        b = np.minimum(a + 1, 123)
        f = (src - a)[None, :, None]
        out[i] = w[:, a, :] * (1 - f) + w[:, b, :] * f
    return out.astype(np.float32), kept, periods


with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in z.files}
xc, kept, per = cycle(d['x'])
d2 = dict(d)
d2['x'] = xc
np.savez_compressed(BIG / 'data/data-cycle.npz', **d2)
summary = dict(windows=len(xc), kept_original=int(kept.sum()), median_period=float(np.median(per[~kept])),
               period_by_source={s: float(np.median(per[(~kept) & (d['source'][d['window_record']] == s)])) for s in ('Stanford', 'LJMU')})
old = read(ROOT / 'studies/20261105-unseen-normal-test/preparation.json')
pre = read(ROOT / 'studies/20261107-live-patient-eval/preparation.json')
with np.load(OLD / 'inputs.npz') as z:
    td_x, td_o = z['live__x'], z['live__owner']
    self_x, self_o = z['self__x'], z['self__owner']
with np.load(LIVE / 'inputs.npz') as z:
    p_x, p_o = z['x'], z['owner']
td_people = old['live_people']
p_people = ['LIVE-' + r['group'] + ':' + r['person'] for r in pre['records']]
p_group = [r['group'] for r in pre['records']]


def fold_of(person):
    return int(hashlib.sha256(('20261109:' + person).encode()).hexdigest(), 16) % 5


live_rows = []
for i, person in enumerate(td_people):
    live_rows.append(dict(record='TD:%d' % i, person=person, group='TD', fold=fold_of(person)))
for i, person in enumerate(p_people):
    live_rows.append(dict(record=pre['records'][i]['record'], person=person, group=p_group[i], fold=fold_of(person)))
live_x = np.concatenate([td_x, p_x]).astype(np.float32)
live_o = np.concatenate([td_o, p_o + len(td_people)])
out = {}
for name, x, o in (('live', live_x, live_o), ('self', self_x.astype(np.float32), self_o)):
    out[name + '__x'] = x
    out[name + '__owner'] = o
    cx, kk, _ = cycle(x)
    out[name + '__xcycle'] = cx
    summary[name + '_kept_original'] = int(kk.sum())
np.savez_compressed(BIG / 'data/external.npz', **out)
(BIG / 'data/live-rows.json').write_text(json.dumps(live_rows, indent=1), encoding='utf-8')
summary['live_fold_counts'] = {g: [sum(1 for r in live_rows if r['group'] == g and r['fold'] == f and r['person'] not in [] ) for f in range(5)] for g in ('TD', 'CP', 'ITW')}
summary['live_people_by_fold'] = {g: [len({r['person'] for r in live_rows if r['group'] == g and r['fold'] == f}) for f in range(5)] for g in ('TD', 'CP', 'ITW')}
summary['sha256'] = {n: sha(BIG / 'data' / n) for n in ('data-cycle.npz', 'external.npz', 'live-rows.json')}
summary['source_data_sha256'] = sha(SRC / 'data.npz')
(HERE / 'preparation.json').write_text(json.dumps(summary, indent=1) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=1))
