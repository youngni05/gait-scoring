"""Audit and evaluate the four improvement arms against the frozen CNN references. CPU only."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REC = Path('D:/gait-research-data/D_P/model-improvement-20261109/received/runs')
DATA = Path('D:/gait-research-data/D_P/model-improvement-20261109/data')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
B = ROOT / 'studies/20261104-ljmu100-mixed/bundle/CNN'
sys.path[:0] = [str(B), str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps')]
spec = importlib.util.spec_from_file_location('rr', B / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from model import GdiCNN, channels
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as Fn
torch.set_num_threads(7)
ARMS = ('R_rank', 'D_adversarial', 'C0_cycle', 'C1_cycle')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in z.files}
with np.load(DATA / 'data-cycle.npz') as z:
    xcyc = z['x']
y, pid, fold, src = d['y'], d['person_id'], d['fold'], d['source']
stan = src == 'Stanford'
ext = dict(np.load(DATA / 'external.npz'))
rows = read(DATA / 'live-rows.json')
lg = np.array([x['group'] for x in rows])
lf = np.array([x['fold'] for x in rows])
lp = np.array([x['person'] for x in rows])
old = read(ROOT / 'studies/20261105-unseen-normal-test/preparation.json')
nums = np.array(old['self_numbers'])
snorm = np.isin(nums, old['self_normal'])
audit = {}


def members(arm, f):
    done = read(REC / arm / f'fold{f}' / 'complete.json')
    return done, [m for m in done['members']]


def load_model(path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = GdiCNN(.5)
    m.load_state_dict(ck['model'], strict=True)
    m.eval()
    return m, ck


def arm_outputs(arm):
    """Development OOF bag, LIVE and self bags per fold, with Stanford-only calibration."""
    oof = np.full_like(y, np.nan, dtype=float)
    live, selfv = [], []
    worst = 0.
    for f in range(5):
        done, mem = members(arm, f)
        dev, lv, sv = [], [], []
        for m in mem:
            folder = REC / arm / f'fold{f}' / m['tag']
            assert sha(folder / 'latest.pt') == m['checkpoint_sha256']
            z = np.load(folder / 'replay.npz')
            a, b = z['stanford_calibration']
            if m['tag'] == 'refit':
                net, ck = load_model(folder / 'latest.pt')
                xs = xcyc if arm.startswith('C') else d['x']
                rows_ = z['evaluation_row']
                w = np.isin(d['window_record'], rows_)
                mp = {int(v): i for i, v in enumerate(rows_)}
                ws = r.WindowSet(xs[w].astype(np.float32), np.array([mp[int(v)] for v in d['window_record'][w]]), len(rows_))
                again = r.predict(net, ws, ck['lo'], ck['span'], 'cpu')
                worst = max(worst, float(np.abs(again - z['evaluation_raw']).max()))
            if m['excluded_from_bag']:
                continue
            ev = z['evaluation_row']
            dev.append((ev, z['evaluation_raw'] * a + b))
            lv.append(z['live_raw'] * a + b)
            sv.append(z['self_raw'] * a + b)
        ev = dev[0][0]
        oof[ev] = np.mean([v for _, v in dev], axis=0)
        live.append(np.mean(lv, axis=0))
        selfv.append(np.mean(sv, axis=0))
    audit[arm] = dict(refit_replay_max_error=worst)
    return oof, np.array(live), np.array(selfv)


def ref_bag(arm_dir, cal_stan):
    """Existing CNN bag OOF from replay files, Stanford-only calibration."""
    oof = np.full_like(y, np.nan, dtype=float)
    for f in range(5):
        root = SRC / 'received/data/runs/mixed/CNN' / f'fold{f}'
        preds = []
        for m in read(root / 'complete.json')['members']:
            if m['excluded_from_bag']:
                continue
            with np.load(root / m['tag'] / 'replay.npz') as z:
                tr, tp, ev, er = z['train_row'], z['train_raw'], z['evaluation_row'], z['evaluation_raw']
            k = stan[tr]
            co = np.linalg.lstsq(np.column_stack([tp[k].ravel(), np.ones(tp[k].size)]), y[tr][k].ravel(), rcond=None)[0]
            preds.append(er * co[0] + co[1])
        oof[ev] = np.mean(preds, axis=0)
    return oof


Z = np.load(SRC / 'comparison-predictions.npz')
base = np.full_like(y, np.nan, dtype=float)
base[stan] = Z['baseline__CNN_bag'][Z['source'] == 'Stanford']
mixed = ref_bag('mixed', True)
basemods = read(ROOT / 'studies/20261104-ljmu100-mixed/bundle/baseline-models.json')


def ext_baseline():
    live, selfv = [], []
    LW = r.WindowSet(ext['live__x'], ext['live__owner'], len(rows))
    SW = r.WindowSet(ext['self__x'], ext['self__owner'], int(ext['self__owner'].max()) + 1)
    for f in range(5):
        lv, sv = [], []
        for m in [v for v in basemods if v['family'] == 'CNN' and v['fold'] == f]:
            net, ck = load_model(SRC / m['checkpoint'])
            a, b = m['calibration']
            lv.append(r.predict(net, LW, ck['lo'], ck['span'], 'cpu') * a + b)
            sv.append(r.predict(net, SW, ck['lo'], ck['span'], 'cpu') * a + b)
        live.append(np.mean(lv, axis=0))
        selfv.append(np.mean(sv, axis=0))
    return np.array(live), np.array(selfv)


BL, BS = ext_baseline()


def person_err(p, mask):
    q = np.unique(pid[mask.any(1)])
    e = p - y
    return q, np.array([np.abs(e[mask & (pid == v)[:, None]]).mean() for v in q]), np.array([e[mask & (pid == v)[:, None]].mean() for v in q])


BANDS = {'all': np.repeat(stan[:, None], 2, 1), 'ge90': stan[:, None] & (y >= 90), 'ge100': stan[:, None] & (y >= 100), 'lt70': stan[:, None] & (y < 70)}


def dev_table(p):
    out = {}
    for k, m in BANDS.items():
        q, mae, bias = person_err(p, m)
        _, mb, bb = person_err(base, m)
        dm, db = mae - mb, np.abs(bias) - np.abs(bb)
        draw = np.random.default_rng(20261109).integers(0, len(q), (10000, len(q)))
        out[k] = dict(people=len(q), mae=round(float(mae.mean()), 3), bias=round(float(bias.mean()), 3), delta_mae=round(float(dm.mean()), 3),
                      delta_mae_ci95=[round(float(v), 3) for v in np.quantile(dm[draw].mean(1), [.025, .975])],
                      delta_abs_bias=round(float(db.mean()), 3), delta_abs_bias_ci95=[round(float(v), 3) for v in np.quantile(db[draw].mean(1), [.025, .975])])
    return out


def ext_table(arm, live, selfv, held_only):
    o = {}
    for g in ('TD', 'CP', 'ITW'):
        idx = np.flatnonzero(lg == g)
        sc, bs = [], []
        people = {}
        for i in idx:
            folds = [lf[i]] if held_only else range(5)
            v = np.mean([live[f][i].mean() for f in folds])
            b = np.mean([BL[f][i].mean() for f in folds])
            people.setdefault(lp[i], []).append((v, b))
        arr = np.array([np.mean(v, axis=0) for v in people.values()])
        o[g] = dict(people=len(arr), score=round(float(arr[:, 0].mean()), 2), baseline=round(float(arr[:, 1].mean()), 2), rise=round(float((arr[:, 0] - arr[:, 1]).mean()), 2))
    for name, k in (('self_normal', snorm), ('self_abnormal', ~snorm)):
        v = selfv.mean(0).mean(1)[k]
        b = BS.mean(0).mean(1)[k]
        o[name] = dict(videos=int(k.sum()), score=round(float(v.mean()), 2), baseline=round(float(b.mean()), 2), rise=round(float((v - b).mean()), 2))
    o['TD_minus_CP'] = round(o['TD']['score'] - o['CP']['score'], 2)
    o['TD_minus_CP_baseline'] = round(o['TD']['baseline'] - o['CP']['baseline'], 2)
    return o


res = {'reference': {'before_cnn_bag': dev_table(base), 'normal_added_measured_cal': dev_table(mixed)}}
for arm in ARMS:
    oof, live, selfv = arm_outputs(arm)
    res[arm] = dict(development=dev_table(oof), external=ext_table(arm, live, selfv, arm == 'R_rank'))
    s = res[arm]
    s['success'] = dict(accuracy=s['development']['all']['delta_mae_ci95'][1] <= .10,
                        high_scores=s['development']['ge90']['delta_abs_bias_ci95'][1] < 0,
                        normal_response=s['external']['TD_minus_CP'] > s['external']['TD_minus_CP_baseline'])
    print(arm, json.dumps(s['success']), json.dumps(s['development']['all']), json.dumps(s['development']['ge90']), json.dumps(s['external']), flush=True)
res['audit'] = audit
(HERE / 'results.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8')
print(json.dumps(res['reference'], indent=0)[:1500])
