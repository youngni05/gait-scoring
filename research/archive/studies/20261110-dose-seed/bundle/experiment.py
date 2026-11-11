"""Dose and seed arms of the normal-added CNN-STT ensemble.

Same frozen training functions, folds, epoch rule and calibration storage as studies/20261104-ljmu100-mixed.
Each arm keeps all Stanford records and the LJMU people chosen for that dose (all of their records), and may
shift the seed schedule. After every fit, raw outputs on the external inputs are stored for local analysis."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('action', choices=['check', 'pilot', 'train'])
ap.add_argument('--family', choices=['CNN', 'STT', 'bilateral'], required=True)
ap.add_argument('--arm', required=True)
ap.add_argument('--data-root', type=Path, required=True)
ap.add_argument('--folds', default='0,1,2,3,4')
ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
a = ap.parse_args()
sys.path[:0] = [str(HERE / a.family), str(HERE / a.family / 'vendor')]
spec = importlib.util.spec_from_file_location('original_training', HERE / a.family / 'run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
torch = r.torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


PROTO = read(HERE / 'study-protocol.json')
ARM = PROTO['arms'][a.arm]


def shape(x):
    return x if a.family == 'CNN' else np.ascontiguousarray(x[..., :8].reshape(-1, 2, 124, 4, 2))


def chosen_ljmu(d, k):
    """Nested, fold-balanced choice: rank people inside each fold by a fixed hash, then take rank 0 of every fold,
    rank 1 of every fold, and so on, until k people are chosen."""
    lj = d['source'] == 'LJMU'
    by_fold = {}
    for p in sorted(set(d['person_id'][lj].tolist())):
        f = int(d['fold'][d['person_id'] == p][0])
        by_fold.setdefault(f, []).append(p)
    for f in by_fold:
        by_fold[f].sort(key=lambda p: hashlib.sha256(('20261110:' + p).encode()).hexdigest())
    order = [by_fold[f][i] for i in range(max(map(len, by_fold.values()))) for f in sorted(by_fold) if i < len(by_fold[f])]
    assert len(order) == 64 and len(set(order)) == 64
    return set(order[:k])


def load_data():
    seal = read(HERE / 'manifest.json')
    for name, digest in seal.items():
        assert sha(HERE / name) == digest, name
    assert sha(a.data_root / 'data.npz') == PROTO['data_sha256']
    with np.load(a.data_root / 'data.npz') as z:
        d = {k: z[k].copy() for k in z.files}
    keep_people = chosen_ljmu(d, ARM['ljmu_people'])
    keep = (d['source'] == 'Stanford') | np.isin(d['person_id'], list(keep_people))
    new_index = np.full(len(keep), -1)
    new_index[keep] = np.arange(keep.sum())
    w = keep[d['window_record']]
    x = d.pop('x')[w]
    rec = new_index[d.pop('window_record')[w]]
    start = d.pop('window_start')[w]
    fb = d.pop('canonical_fallback')[w]
    for k in list(d):
        assert len(d[k]) == len(keep), k
        d[k] = d[k][keep]
    d['x'] = r.WindowSet(shape(x), rec, int(keep.sum()), start, fb)
    d['x'].validate()
    lj = d['source'] == 'LJMU'
    assert int((d['source'] == 'Stanford').sum()) == 1061
    assert len(set(d['person_id'][lj].tolist())) == ARM['ljmu_people']
    assert np.all(d['y'][lj] == 100)
    for f in range(5):
        for j in range(5):
            if j != f:
                r.split(d, f, j)
    d['chosen_ljmu'] = sorted(keep_people)
    return d


def load_external():
    assert sha(a.data_root / 'external.npz') == PROTO['external_sha256']
    out = {}
    with np.load(a.data_root / 'external.npz') as z:
        for name in PROTO['external_sets']:
            out[name] = r.WindowSet(shape(z[f'{name}__x']), z[f'{name}__owner'], int(z[f'{name}__n']))
    return out


def identity(d, c, f, j, epochs, purpose):
    tr, va, te = r.split(d, f, j)
    seed = c['seed'] + 100 * f + (50 if j is None else j)
    return {'bundle_sha256': sha(HERE / 'manifest.json'), 'family': a.family, 'arm': a.arm,
            'ljmu_people': d['chosen_ljmu'], 'seed_offset': ARM['seed_offset'],
            'purpose': purpose, 'outer_fold': f, 'inner_fold': j, 'epochs': epochs,
            ('fit_seed' if a.family == 'CNN' else 'seed'): seed,
            'train_people': np.unique(d['person_id'][tr]).tolist(),
            'validation_people': np.unique(d['person_id'][va]).tolist(),
            'outer_evaluation_people': np.unique(d['person_id'][te]).tolist()}


def ext_raw(model, lo, span, ext):
    return {f'ext__{k}': r.predict(model, v, lo, span, a.device) for k, v in ext.items()}


def train_fold(d, c, f, ext):
    dest = a.data_root / 'runs' / a.arm / a.family / f'fold{f}'
    if (dest / 'complete.json').exists():
        done = read(dest / 'complete.json')
        assert done['bundle_sha256'] == sha(HERE / 'manifest.json')
        return
    histories, members = [], []
    te = d['fold'] == f
    for j in range(5):
        if j == f:
            continue
        tr, va, _ = r.split(d, f, j)
        folder = dest / f'inner{j}'
        ident = identity(d, c, f, j, c['max_epochs'], 'inner_epoch_selection')
        model, lo, span, h = r.fit(d['x'][tr], d['y'][tr], d['person_id'][tr], d['x'][va], d['y'][va], d['person_id'][va],
                                   c, c['max_epochs'], folder, ident, a.device)
        histories.append(h)
        train_raw = r.predict(model, d['x'][tr], lo, span, a.device)
        np.savez_compressed(folder / 'replay.npz', train_row=np.flatnonzero(tr), train_raw=train_raw,
                            train_source=d['source'][tr], train_y=d['y'][tr],
                            evaluation_row=np.flatnonzero(te), evaluation_raw=r.predict(model, d['x'][te], lo, span, a.device),
                            **ext_raw(model, lo, span, ext))
        members.append({'tag': f'inner{j}', 'excluded_from_bag': bool((np.std(train_raw, axis=0) < .01).any()),
                        'checkpoint_sha256': sha(folder / 'latest.pt'), 'calibration': list(map(float, h[-1]['calibration']))})
        del model
    epoch, means = r.select_epoch(histories)
    r.save(dest / 'selection.json', {'selected_epoch': epoch, 'inner_mean_person_mae_by_epoch': means})
    tr, _, te = r.split(d, f, None)
    folder = dest / 'refit'
    model, lo, span, h = r.fit(d['x'][tr], d['y'][tr], d['person_id'][tr], None, None, None, c, epoch, folder,
                               identity(d, c, f, None, epoch, 'outer_refit'), a.device)
    train_raw = r.predict(model, d['x'][tr], lo, span, a.device)
    np.savez_compressed(folder / 'replay.npz', train_row=np.flatnonzero(tr), train_raw=train_raw,
                        train_source=d['source'][tr], train_y=d['y'][tr],
                        evaluation_row=np.flatnonzero(te), evaluation_raw=r.predict(model, d['x'][te], lo, span, a.device),
                        **ext_raw(model, lo, span, ext))
    members.append({'tag': 'refit', 'excluded_from_bag': bool((np.std(train_raw, axis=0) < .01).any()),
                    'checkpoint_sha256': sha(folder / 'latest.pt'), 'calibration': list(map(float, h[-1]['calibration']))})
    r.save(dest / 'complete.json', {'bundle_sha256': sha(HERE / 'manifest.json'), 'family': a.family, 'arm': a.arm,
                                    'fold': f, 'selected_epoch': epoch, 'members': members,
                                    'evaluation_people': np.unique(d['person_id'][te]).tolist()})
    print('FOLD_COMPLETE', a.arm, a.family, f, flush=True)


def pilot(d, c, ext):
    tr, va, _ = r.split(d, 0, 1)
    out = a.data_root / 'runs' / 'pilot' / a.arm / a.family
    ident = {'family': a.family, 'arm': a.arm, 'purpose': 'real_pilot', ('fit_seed' if a.family == 'CNN' else 'seed'): c['seed'],
             'epochs': 3, 'train_people': np.unique(d['person_id'][tr]).tolist(), 'validation_people': np.unique(d['person_id'][va]).tolist()}
    model, lo, span, h = r.fit(d['x'][tr], d['y'][tr], d['person_id'][tr], d['x'][va], d['y'][va], d['person_id'][va], c, 3, out, ident, a.device)
    raw = r.predict(model, d['x'][va], lo, span, a.device)
    ck = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
    m2 = r.GdiCNN(c['dropout']) if a.family == 'CNN' else r.new_model()
    m2.load_state_dict(ck['model'], strict=True)
    m2.to(a.device)
    err = float(np.max(np.abs(raw - r.predict(m2, d['x'][va], ck['lo'], ck['span'], a.device))))
    assert err < .001, err
    e = ext_raw(model, lo, span, ext)
    assert all(np.isfinite(v).all() for v in e.values())
    r.save(out / 'verification.json', {'passed': True, 'family': a.family, 'arm': a.arm, 'epochs': 3,
                                       'mean_epoch_seconds': float(np.mean([v['seconds'] for v in h])),
                                       'records': int(len(d['y'])), 'windows': int(len(d['x'].values)),
                                       'ljmu_people': d['chosen_ljmu'], 'checkpoint_replay_max_error': err,
                                       'external_windows': {k: int(len(v)) for k, v in e.items()},
                                       'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0) if a.device == 'cuda' else None})


d = load_data()
c = read(HERE / a.family / 'protocol.json')
c['seed'] = c['seed'] + ARM['seed_offset']
r.setup(c['seed'], a.device)
if a.action == 'check':
    print(json.dumps({'arm': a.arm, 'family': a.family, 'records': int(len(d['y'])), 'windows': int(len(d['x'].values)),
                      'ljmu_people': len(d['chosen_ljmu']), 'ljmu_by_fold': {f: int(len(set(d['person_id'][(d['fold'] == f) & (d['source'] == 'LJMU')].tolist()))) for f in range(5)},
                      'seed': c['seed']}))
    load_external()
elif a.action == 'pilot':
    pilot(d, c, load_external())
else:
    assert (HERE / 'approved.json').exists()
    ext = load_external()
    for f in [int(x) for x in a.folds.split(',')]:
        train_fold(d, c, f, ext)
