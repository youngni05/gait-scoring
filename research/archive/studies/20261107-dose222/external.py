"""Small-addition adult members on TD, self-recorded and LIVE patient inputs. No fitting."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MATCH = Path('D:/gait-research-data/D_P/matched-normal100-20261105')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
LIVE = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
BIG = Path('D:/gait-research-data/D_P/dose222-20261107')
fam = sys.argv[1]
B = ROOT / 'studies/20261104-ljmu100-mixed/bundle' / fam
sys.path[:0] = [str(B), str(B / 'vendor')]
spec = importlib.util.spec_from_file_location('rr', B / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def shape(x):
    return x if fam == 'CNN' else np.ascontiguousarray(x[..., :8].reshape(-1, 2, 124, 4, 2))


r.setup(20261028, 'cpu')
r.torch.set_num_threads(7)
c = read(B / 'protocol.json')
S = {}
with np.load(OLD / 'inputs.npz') as z:
    for k in ('live', 'self'):
        S[k] = r.WindowSet(shape(z[k + '__x']), z[k + '__owner'], int(z[k + '__n']))
with np.load(LIVE / 'inputs.npz') as z:
    S['patient'] = r.WindowSet(shape(z['x']), z['owner'], len(z['owner']))
out, checked = {}, 0
for f in range(5):
    root = MATCH / 'received/data/adult/runs/mixed' / fam / f'fold{f}'
    for m in read(root / 'complete.json')['members']:
        path = root / m['tag'] / 'latest.pt'
        assert sha(path) == m['checkpoint_sha256']
        checked += 1
        ck = r.torch.load(path, map_location='cpu', weights_only=True)
        model = r.GdiCNN(c['dropout']) if fam == 'CNN' else r.new_model()
        model.load_state_dict(ck['model'], strict=True)
        a, b = m['calibration']
        for k, v in S.items():
            out[f"{k}__{f}__{m['tag']}"] = r.predict(model, v, ck['lo'], ck['span'], 'cpu') * a + b
        out[f"excluded__{f}__{m['tag']}"] = np.array(bool(m['excluded_from_bag']))
np.savez_compressed(BIG / f'external-{fam}.npz', **out)
print('done', fam, checked, flush=True)
