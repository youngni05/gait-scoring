"""Source-shortcut diagnostics for D_adversarial and C1_cycle, as in the neighbour test. CPU only."""
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
from model import GdiCNN, channels
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as Fn
torch.set_num_threads(7)
K = 20
rng = np.random.default_rng(20261109)
with np.load(SRC / 'data.npz') as z:
    d = {k: z[k] for k in z.files}
with np.load(DATA / 'data-cycle.npz') as z:
    xc = z['x']
ext = dict(np.load(DATA / 'external.npz'))
old = json.loads((ROOT / 'studies/20261105-unseen-normal-test/preparation.json').read_text(encoding='utf-8-sig'))
nums = np.array(old['self_numbers'])
wr = d['window_record']
fw = d['fold'][wr]
st = (d['source'] == 'Stanford')[wr]
y = d['y']


def feat(net, x):
    for i, (conv, norm) in enumerate(zip(net.convs, net.norms)):
        x = norm(Fn.relu(conv(x)))
        if i % 2:
            x = net.dropout(Fn.max_pool1d(x, 3 if i == 5 else 2))
    return x.transpose(1, 2).reshape(x.shape[0], -1)


def emb(net, x):
    with torch.no_grad():
        return np.concatenate([feat(net, channels(torch.from_numpy(np.ascontiguousarray(x[s:s + 512]))).reshape(-1, 12, 124)).numpy() for s in range(0, len(x), 512)])


out = {}
for arm in ('D_adversarial', 'C1_cycle'):
    X = (xc if arm.startswith('C') else d['x']).astype(np.float32)
    selfx = ext['self__xcycle' if arm.startswith('C') else 'self__x'].astype(np.float32)
    sn = selfx[np.isin(ext['self__owner'], np.flatnonzero(np.isin(nums, old['self_normal'])))]
    ljmu_share_high, side_legs, gdi_legs, side_self, acc = [], [], [], [], []
    for f in range(5):
        ck = torch.load(REC / arm / f'fold{f}' / 'refit' / 'latest.pt', map_location='cpu', weights_only=True)
        net = GdiCNN(.5)
        net.load_state_dict(ck['model'], strict=True)
        net.eval()
        tr = np.flatnonzero(fw != f)
        pick = rng.choice(tr, min(9000, len(tr)), replace=False)
        E = emb(net, X[pick])
        lab = np.repeat(~st[pick], 2).astype(int)
        m, s = E.mean(0), E.std(0) + 1e-6
        Z = (E - m) / s
        clf = LogisticRegression(C=1., max_iter=3000, class_weight='balanced').fit(Z, lab)
        acc.append(float(clf.score(Z, lab)))
        held = np.flatnonzero((fw == f) & st)
        q = (emb(net, X[held]) - m) / s
        sc = (q @ clf.coef_[0] + clf.intercept_[0]).reshape(-1, 2)
        side_legs.append(sc.ravel())
        gdi_legs.append(y[wr[held]].ravel())
        high = held[y[wr[held]].min(1) >= 90]
        qh = (emb(net, X[high]) - m) / s
        for a in range(0, len(qh), 256):
            v = qh[a:a + 256]
            d2 = (v ** 2).sum(1)[:, None] - 2 * v @ Z.T + (Z ** 2).sum(1)[None]
            ljmu_share_high.append(lab[np.argpartition(d2, K, axis=1)[:, :K]].mean(1))
        side_self.append(float((((emb(net, sn) - m) / s) @ clf.coef_[0] + clf.intercept_[0]).mean()))
        print(arm, f, flush=True)
    sl, g = np.concatenate(side_legs), np.concatenate(gdi_legs)
    slope = float(np.polyfit(g, sl, 1)[0])
    rk = lambda v: np.argsort(np.argsort(v))
    out[arm] = dict(source_accuracy=round(float(np.mean(acc)), 3), high_ljmu_share=round(float(np.concatenate(ljmu_share_high).mean()), 3),
                    side_slope_per_gdi_point=round(slope, 4), side_spearman_gdi=round(float(np.corrcoef(rk(g), rk(sl))[0, 1]), 3),
                    side_mean_dev=round(float(sl.mean()), 3), side_self_normal=round(float(np.mean(side_self)), 3),
                    shortcut_reduced=bool(slope > 0 or np.concatenate(ljmu_share_high).mean() >= .2))
    print(arm, out[arm], flush=True)
(HERE / 'diagnostics.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
