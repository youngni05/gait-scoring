"""Post-hoc: which inputs separate high measured-GDI development records from external normal walking?"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
NAMES = ['ankle_x', 'ankle_y', 'knee_x', 'knee_y', 'hip_x', 'hip_y', 'toe_x', 'toe_y', 'knee_angle', 'ankle_angle', 'toe_ankle', 'other_ankle_x']
STATS = ['mean', 'sd', 'range', 'step']
with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in ('x', 'window_record', 'y', 'source')}
old = json.loads((ROOT / 'studies/20261105-unseen-normal-test/preparation.json').read_text(encoding='utf-8-sig'))


def feats(x, o, n):
    f = np.concatenate([x.mean(2), x.std(2), np.ptp(x, axis=2), np.abs(np.diff(x, axis=2)).mean(2)], axis=2).mean(1)
    F = np.zeros((n, 48))
    np.add.at(F, o, f)
    return F / np.bincount(o, minlength=n)[:, None]


def signed(x, o, n):
    """Mean signs that reveal walking direction and leg order, kept per leg."""
    a = x[:, :, :, 11].mean(2)
    v = np.zeros((n, 2))
    np.add.at(v, o, a)
    return v / np.bincount(o, minlength=n)[:, None]


F = feats(d['x'], d['window_record'], len(d['y']))
stan = d['source'] == 'Stanford'
hi = d['y'].min(1) >= 90
G = {'Stanford_both_legs_ge90': F[stan & hi], 'Stanford_rest': F[stan & ~hi], 'LJMU': F[~stan]}
with np.load(OLD / 'inputs.npz') as z:
    live = feats(z['live__x'].astype(float), z['live__owner'], 12)
    self_ = feats(z['self__x'].astype(float), z['self__owner'], 13)
    tx = z['self__x']
nums = np.array(old['self_numbers'])
G['LIVE_TD'] = live
G['self_normal'] = self_[np.isin(nums, old['self_normal'])]
ref = G['Stanford_both_legs_ge90']
pool = F[stan].std(0)
labels = [f'{n}.{s}' for s in STATS for n in NAMES]
order = {}
for k in ('LJMU', 'LIVE_TD', 'self_normal'):
    z = (G[k].mean(0) - ref.mean(0)) / pool
    idx = np.argsort(-np.abs(z))[:8]
    order[k] = [(labels[i], round(float(z[i]), 2), round(float(ref[:, i].mean()), 3), round(float(G[k][:, i].mean()), 3)) for i in idx]
z_rest = (ref.mean(0) - G['Stanford_rest'].mean(0)) / pool
idx = np.argsort(-np.abs(z_rest))[:8]
order['within_Stanford_high_vs_rest'] = [(labels[i], round(float(z_rest[i]), 2)) for i in idx]
for k in ('LJMU', 'LIVE_TD', 'self_normal'):
    a = (G[k].mean(0) - ref.mean(0)) / pool
    order[k + '_distance_all_stats'] = round(float(np.sqrt((a ** 2).mean())), 2)
    a2 = (G[k].mean(0) - G['Stanford_rest'].mean(0)) / pool
    order[k + '_alignment_with_high_direction'] = round(float((a2 @ z_rest) / np.sqrt((a2 ** 2).sum() * (z_rest ** 2).sum())), 2)
# Hip and thigh geometry from the raw coordinate channels: pelvis tilt proxy and thigh length constancy
(HERE / 'posthoc-input-shift.json').write_text(json.dumps(order, indent=1) + '\n', encoding='utf-8')
for k, v in order.items():
    print(k, v)
print(len(G['Stanford_both_legs_ge90']))
