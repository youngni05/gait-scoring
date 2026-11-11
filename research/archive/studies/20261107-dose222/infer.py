"""Small-addition adult members on the 222 people. No fitting, no outcomes read."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
FINAL = Path('D:/gait-research-data/D_P/final223-frozen-20261102')
MATCH = Path('D:/gait-research-data/D_P/matched-normal100-20261105')
BIG = Path('D:/gait-research-data/D_P/dose222-20261107'); BIG.mkdir(parents=True, exist_ok=True)
fam = sys.argv[1]; B = ROOT / 'studies/20261104-ljmu100-mixed/bundle' / fam
sys.path[:0] = [str(B), str(B / 'vendor')]
spec = importlib.util.spec_from_file_location('rr', B / 'run.py'); r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def shape(x): return x if fam == 'CNN' else np.ascontiguousarray(x[..., :8].reshape(-1, 2, 124, 4, 2))
r.setup(20261028, 'cpu'); r.torch.set_num_threads(3)
c = read(B / 'protocol.json')
with np.load(FINAL / 'features.npz') as z: x, owner, ids = z['x'], z['window_record'], z['person_id']
data = r.WindowSet(shape(x), owner, len(ids)); assert len(ids) == 377 and len(set(ids.tolist())) == 222
people = {'Stanford:' + str(v) for v in ids.tolist()}
with np.load(MATCH / 'adult/data.npz') as z:
    add = np.flatnonzero(z['source'] != 'Stanford'); keep = np.isin(z['window_record'], add); m = {int(k): i for i, k in enumerate(add)}
    added = r.WindowSet(shape(z['x'][keep]), np.array([m[int(k)] for k in z['window_record'][keep]]), len(add))
out, checked, worst = {}, 0, 0.
for f in range(5):
    root = MATCH / 'received/data/adult/runs/mixed' / fam / f'fold{f}'
    for mem in read(root / 'complete.json')['members']:
        path = root / mem['tag'] / 'latest.pt'; assert sha(path) == mem['checkpoint_sha256']; checked += 1
        ck = r.torch.load(path, map_location='cpu', weights_only=True); identity = ck['identity']
        seen = set(identity['train_people']) | set(identity.get('validation_people', [])); assert not seen & people, 'person overlap'
        model = r.GdiCNN(c['dropout']) if fam == 'CNN' else r.new_model(); model.load_state_dict(ck['model'], strict=True)
        with np.load(root / mem['tag'] / 'replay.npz') as z:
            pos = {int(v): i for i, v in enumerate(z['train_row'])}; saved = z['train_raw'][[pos[int(v)] for v in add]]
        worst = max(worst, float(np.abs(r.predict(model, added, ck['lo'], ck['span'], 'cpu') - saved).max())); assert worst <= 0.005, worst
        a, b = mem['calibration']
        out[f"{f}__{mem['tag']}"] = r.predict(model, data, ck['lo'], ck['span'], 'cpu') * a + b
        out[f"{f}__{mem['tag']}__excluded"] = np.array(bool(mem['excluded_from_bag']))
        print(fam, f, mem['tag'], flush=True)
target = BIG / f'pred-{fam}.npz'; np.savez_compressed(target, **out)
(HERE / f'infer-{fam}.json').write_text(json.dumps(dict(family=fam, checkpoints_hash_checked=checked, replay_max_error=worst, person_overlap=0, protocol_sha256=sha(HERE / 'protocol.json'), prediction_sha256=sha(target)), indent=1), encoding='utf-8')
print('done', fam, checked, worst, flush=True)
