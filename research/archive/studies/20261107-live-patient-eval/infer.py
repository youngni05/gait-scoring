"""Raw outputs of the frozen before and after members on reviewed patient windows. No fitting."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
OB = ROOT / 'studies/20261104-ljmu100-mixed/bundle'
fam = sys.argv[1]
B = OB / fam
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
r.torch.set_num_threads(6)
c = read(B / 'protocol.json')
meta = read(HERE / 'preparation.json')
assert sha(BIG / 'inputs.npz') == meta['inputs_sha256']
assert sha(HERE / 'protocol.json') == meta['protocol_sha256']
with np.load(BIG / 'inputs.npz') as z:
    data = r.WindowSet(shape(z['x']), z['owner'], len(meta['records']))
with np.load(OLD / 'inputs.npz') as z:
    td = r.WindowSet(shape(z['live__x']), z['live__owner'], int(z['live__n']))
saved = np.load(OLD / f'pred-{fam}.npz')
base = read(OB / 'baseline-models.json')


def members(arm, f):
    if arm == 'baseline':
        return [(SRC / m['checkpoint'], m['checkpoint_sha256'], m['tag'], m['calibration'], False)
                for m in base if m['family'] == fam and m['fold'] == f]
    root = SRC / 'received/data/runs/mixed' / fam / f'fold{f}'
    return [(root / m['tag'] / 'latest.pt', m['checkpoint_sha256'], m['tag'], m['calibration'], m['excluded_from_bag'])
            for m in read(root / 'complete.json')['members']]


out, checked, worst = {}, 0, 0.
for arm in ('baseline', 'mixed'):
    for f in range(5):
        tdbag, refit = [], None
        for path, digest, tag, cal, excluded in members(arm, f):
            assert sha(path) == digest
            checked += 1
            ck = r.torch.load(path, map_location='cpu', weights_only=True)
            model = r.GdiCNN(c['dropout']) if fam == 'CNN' else r.new_model()
            model.load_state_dict(ck['model'], strict=True)
            raw = r.predict(model, data, ck['lo'], ck['span'], 'cpu')
            t = r.predict(model, td, ck['lo'], ck['span'], 'cpu') * cal[0] + cal[1]
            out[f'{arm}__{f}__{tag}__raw'] = raw
            out[f'{arm}__{f}__{tag}__cal'] = np.array(cal)
            out[f'{arm}__{f}__{tag}__excluded'] = np.array(bool(excluded))
            if not excluded:
                tdbag.append(t)
            if tag == 'refit':
                refit = t
        worst = max(worst, float(np.abs(np.mean(tdbag, axis=0) - saved[f'live__{arm}__{f}__bag']).max()),
                    float(np.abs(refit - saved[f'live__{arm}__{f}__refit']).max()))
        assert worst <= .005, worst
        print(fam, arm, f, flush=True)
target = BIG / f'raw-{fam}.npz'
np.savez_compressed(target, **out)
(HERE / f'infer-{fam}.json').write_text(json.dumps(dict(
    family=fam, checkpoints_hash_checked=checked, td_replay_max_error=worst, inputs_sha256=meta['inputs_sha256'],
    prediction_sha256=sha(target), torch=str(r.torch.__version__)), indent=1), encoding='utf-8')
print('done', fam, checked, worst, flush=True)
