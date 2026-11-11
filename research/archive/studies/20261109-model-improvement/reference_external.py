"""External scores of the existing normal-added CNN bag (measured-value calibration on Stanford training rows).
Completes the protocol reference that analyze.py only scored on development data. CPU only."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = Path('D:/gait-research-data/D_P/model-improvement-20261109/data')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
B = ROOT / 'studies/20261104-ljmu100-mixed/bundle/CNN'
sys.path[:0] = [str(B), str(ROOT / 'outputs/gdi-ridge-experiment-20261016/deps')]
spec = importlib.util.spec_from_file_location('rr', B / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
from model import GdiCNN
import torch
torch.set_num_threads(7)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


with np.load(SRC / 'data.npz') as z:
    y, stan = z['y'], z['source'] == 'Stanford'
ext = dict(np.load(DATA / 'external.npz'))
rows = read(DATA / 'live-rows.json')
lg = np.array([x['group'] for x in rows])
lp = np.array([x['person'] for x in rows])
old = read(ROOT / 'studies/20261105-unseen-normal-test/preparation.json')
snorm = np.isin(np.array(old['self_numbers']), old['self_normal'])
LW = r.WindowSet(ext['live__x'], ext['live__owner'], len(rows))
SW = r.WindowSet(ext['self__x'], ext['self__owner'], int(ext['self__owner'].max()) + 1)
live, selfv = [], []
for f in range(5):
    root = SRC / 'received/data/runs/mixed/CNN' / f'fold{f}'
    lv, sv = [], []
    for m in read(root / 'complete.json')['members']:
        if m['excluded_from_bag']:
            continue
        with np.load(root / m['tag'] / 'replay.npz') as z:
            tr, tp = z['train_row'], z['train_raw']
        k = stan[tr]
        a, b = np.linalg.lstsq(np.column_stack([tp[k].ravel(), np.ones(tp[k].size)]), y[tr][k].ravel(), rcond=None)[0]
        ck = torch.load(root / m['tag'] / 'latest.pt', map_location='cpu', weights_only=True)
        net = GdiCNN(.5)
        net.load_state_dict(ck['model'], strict=True)
        net.eval()
        lv.append(r.predict(net, LW, ck['lo'], ck['span'], 'cpu') * a + b)
        sv.append(r.predict(net, SW, ck['lo'], ck['span'], 'cpu') * a + b)
    live.append(np.mean(lv, axis=0))
    selfv.append(np.mean(sv, axis=0))
live, selfv = np.array(live), np.array(selfv)
o = {}
for g in ('TD', 'CP', 'ITW'):
    people = {}
    for i in np.flatnonzero(lg == g):
        people.setdefault(lp[i], []).append(live[:, i].mean())
    o[g] = dict(people=len(people), score=round(float(np.mean([np.mean(v) for v in people.values()])), 2))
for name, k in (('self_normal', snorm), ('self_abnormal', ~snorm)):
    o[name] = dict(videos=int(k.sum()), score=round(float(selfv.mean(0).mean(1)[k].mean()), 2))
o['TD_minus_CP'] = round(o['TD']['score'] - o['CP']['score'], 2)
print(json.dumps(o))
(HERE / 'reference-external.json').write_text(json.dumps(o, indent=1) + '\n', encoding='utf-8')
