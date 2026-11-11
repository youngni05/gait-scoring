"""Audit received runs: every checkpoint hash against complete.json, and a CPU replay of one refit member per
family and arm on the evaluation-set inputs against the stored ext__eval output."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REC = Path('D:/gait-research-data/D_P/dose-seed-20261110/received/runs')
DATA = Path('D:/gait-research-data/D_P/dose-seed-20261110/data')
ARMS = ['n4_s0', 'n8_s0', 'n16_s0', 'n32_s0', 'n0_s1', 'n0_s2', 'n64_s1', 'n64_s2']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


out = {'checkpoints_checked': 0, 'hash_mismatch': [], 'replay': {}}
for arm in ARMS:
    for fam in ('CNN', 'STT', 'bilateral'):
        for f in range(5):
            done = read(REC / arm / fam / f'fold{f}' / 'complete.json')
            assert done['arm'] == arm and done['family'] == fam and done['fold'] == f
            for m in done['members']:
                ck = REC / arm / fam / f'fold{f}' / m['tag'] / 'latest.pt'
                out['checkpoints_checked'] += 1
                if sha(ck) != m['checkpoint_sha256']:
                    out['hash_mismatch'].append(str(ck))

ext = np.load(DATA / 'external.npz')
for fam in ('CNN', 'STT', 'bilateral'):
    # each family ships its own windows/model modules; drop the previous family's copies first
    for k in [k for k in list(sys.modules) if k in ('windows', 'bilateral', 'transformer_model') or k.split('.')[0] in ('model', 'sources', 'einops')]:
        del sys.modules[k]
    B = HERE / 'bundle' / fam
    sys.path[:0] = [str(B), str(B / 'vendor')]
    spec = importlib.util.spec_from_file_location(f'rr_{fam}', B / 'run.py')
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    sys.path[:2] = []
    r.setup(20261028, 'cpu')
    r.torch.set_num_threads(6)
    c = read(B / 'protocol.json')
    x = ext['eval__x']
    if fam != 'CNN':
        x = np.ascontiguousarray(x[..., :8].reshape(-1, 2, 124, 4, 2))
    ws = r.WindowSet(x, ext['eval__owner'], int(ext['eval__n']))
    worst = 0.
    for arm in ARMS:
        folder = REC / arm / fam / 'fold0' / 'refit'
        ck = r.torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
        net = r.GdiCNN(c['dropout']) if fam == 'CNN' else r.new_model()
        net.load_state_dict(ck['model'], strict=True)
        raw = r.predict(net, ws, ck['lo'], ck['span'], 'cpu')
        with np.load(folder / 'replay.npz') as z:
            worst = max(worst, float(np.abs(raw - z['ext__eval']).max()))
        print(fam, arm, worst, flush=True)
    out['replay'][fam] = worst
(HERE / 'audit.json').write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps({k: (v if k != 'hash_mismatch' else len(v)) for k, v in out.items()}))
