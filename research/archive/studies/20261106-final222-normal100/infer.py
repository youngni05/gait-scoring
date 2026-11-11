"""Raw outputs of every normal-added neural member on the 222-person inputs. No fitting, no outcomes."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIG = Path('D:/gait-research-data/D_P/final222-normal100-20261106')
FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102')
SRC = Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
OLD = Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
BIG.mkdir(parents=True, exist_ok=True)
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
r.torch.set_num_threads(8)
c = read(B / 'protocol.json')
with np.load(FINAL / 'features.npz') as z:
    x, owner, ids = z['x'], z['window_record'], z['person_id']
data = r.WindowSet(shape(x), owner, len(ids))
assert len(ids) == 377 and len(set(ids.tolist())) == 222
people = {'Stanford:' + str(v) for v in ids.tolist()}
with np.load(OLD / 'inputs.npz') as z:
    live = r.WindowSet(shape(z['live__x']), z['live__owner'], int(z['live__n']))
saved = np.load(OLD / f'member-raw-{fam}.npz')
out, checked, worst = {}, 0, 0.
for f in range(5):
    root = SRC / 'received/data/runs/mixed' / fam / f'fold{f}'
    done = read(root / 'complete.json')
    for m in done['members']:
        path = root / m['tag'] / 'latest.pt'
        assert sha(path) == m['checkpoint_sha256']
        checked += 1
        ck = r.torch.load(path, map_location='cpu', weights_only=True)
        identity = ck['identity']
        seen = set(identity['train_people']) | set(identity.get('validation_people', []))
        assert not seen & people, 'person overlap'
        model = r.GdiCNN(c['dropout']) if fam == 'CNN' else r.new_model()
        model.load_state_dict(ck['model'], strict=True)
        replay = r.predict(model, live, ck['lo'], ck['span'], 'cpu')
        worst = max(worst, float(np.abs(replay - saved[f"live__{f}__{m['tag']}"]).max()))
        assert worst <= 0.005, worst
        out[f"{f}__{m['tag']}"] = r.predict(model, data, ck['lo'], ck['span'], 'cpu')
        print(fam, f, m['tag'], flush=True)
target = BIG / f'member-raw-{fam}.npz'
np.savez_compressed(target, **out)
(HERE / f'infer-{fam}.json').write_text(json.dumps(dict(
    family=fam, checkpoints_hash_checked=checked, replay_max_error=worst, person_overlap=0,
    protocol_sha256=sha(HERE / 'protocol.json'), features_sha256=sha(FINAL / 'features.npz'),
    prediction_sha256=sha(target), torch=str(r.torch.__version__)), indent=1), encoding='utf-8')
print('done', fam, checked, worst, flush=True)
