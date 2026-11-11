"""Short continued training of the published baseline members on Stanford + LJMU100.

Every member starts from the baseline member with the same outer and inner fold (same Stanford training
people), so validation and outer people are never seen. Lower learning rate, short schedule, epoch chosen by
the available inner curves (bilateral STT has no baseline inner members and uses the STT epoch of its fold).
Raw outputs on the external inputs are stored after every fit."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('action', choices=['check', 'pilot', 'train'])
ap.add_argument('--family', choices=['CNN', 'STT', 'bilateral'], required=True)
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
FT = PROTO['finetune'][a.family]
BASE = {(m['family'], m['fold'], m['tag']): m for m in read(HERE / 'baseline-models.json')}


def shape(x):
    return x if a.family == 'CNN' else np.ascontiguousarray(x[..., :8].reshape(-1, 2, 124, 4, 2))


def load_data():
    for name, digest in read(HERE / 'manifest.json').items():
        assert sha(HERE / name) == digest, name
    assert sha(a.data_root / 'data.npz') == PROTO['data_sha256']
    with np.load(a.data_root / 'data.npz') as z:
        d = {k: z[k].copy() for k in z.files}
    x = d.pop('x')
    d['x'] = r.WindowSet(shape(x), d.pop('window_record'), len(d['y']), d.pop('window_start'), d.pop('canonical_fallback'))
    d['x'].validate()
    assert len(d['y']) == 1942 and len(set(d['person_id'])) == 756
    return d


def load_external():
    assert sha(a.data_root / 'external.npz') == PROTO['external_sha256']
    with np.load(a.data_root / 'external.npz') as z:
        return {n: r.WindowSet(shape(z[f'{n}__x']), z[f'{n}__owner'], int(z[f'{n}__n'])) for n in PROTO['external_sets']}


def baseline_state(f, tag):
    m = BASE.get((a.family, f, tag))
    if m is None:
        return None, None
    path = a.data_root / m['checkpoint']
    assert sha(path) == m['checkpoint_sha256'], path
    ck = torch.load(path, map_location='cpu', weights_only=True)
    return ck, m['checkpoint_sha256']


def patch_init(state):
    """The original fit builds a fresh model; make that model start from the baseline weights."""
    if a.family == 'CNN':
        orig = r.__dict__.setdefault('_orig_GdiCNN', r.GdiCNN)

        def make(dropout):
            net = orig(dropout)
            net.load_state_dict(state, strict=True)
            return net
        r.GdiCNN = make
    else:
        orig = r.__dict__.setdefault('_orig_new_model', r.new_model)

        def make():
            net = orig()
            net.load_state_dict(state, strict=True)
            return net
        r.new_model = make


def same_scale(ck, y):
    if a.family == 'CNN':
        lo, span = r.scale_from_train(y)
        return torch.equal(ck['lo'].cpu(), lo) and torch.equal(ck['span'].cpu(), span)
    lo = float(np.float32(y.min()))
    span = float(np.float32(max(float(y.max() - y.min()), 1e-6)))
    return ck['lo'] == lo and ck['span'] == span


def identity(d, c, f, j, epochs, purpose, init_sha):
    tr, va, te = r.split(d, f, j)
    seed = c['seed'] + 100 * f + (50 if j is None else j)
    return {'bundle_sha256': sha(HERE / 'manifest.json'), 'family': a.family, 'arm': 'finetune_from_baseline',
            'init_checkpoint_sha256': init_sha, 'purpose': purpose, 'outer_fold': f, 'inner_fold': j, 'epochs': epochs,
            ('fit_seed' if a.family == 'CNN' else 'seed'): seed,
            'train_people': np.unique(d['person_id'][tr]).tolist(),
            'validation_people': np.unique(d['person_id'][va]).tolist(),
            'outer_evaluation_people': np.unique(d['person_id'][te]).tolist()}


def one_fit(d, c, f, j, epochs, folder, ext, purpose):
    ck, init_sha = baseline_state(f, 'refit' if j is None else f'inner{j}')
    if ck is None:
        return None
    tr, va, te = r.split(d, f, j)
    assert same_scale(ck, d['y'][tr]), 'Target scale differs from the baseline member'
    patch_init(ck['model'])
    vx, vy, vi = (None, None, None) if j is None else (d['x'][va], d['y'][va], d['person_id'][va])
    model, lo, span, h = r.fit(d['x'][tr], d['y'][tr], d['person_id'][tr], vx, vy, vi, c, epochs, folder,
                               identity(d, c, f, j, epochs, purpose, init_sha), a.device)
    train_raw = r.predict(model, d['x'][tr], lo, span, a.device)
    np.savez_compressed(folder / 'replay.npz', train_row=np.flatnonzero(tr), train_raw=train_raw,
                        train_source=d['source'][tr], train_y=d['y'][tr],
                        evaluation_row=np.flatnonzero(te), evaluation_raw=r.predict(model, d['x'][te], lo, span, a.device),
                        **{f'ext__{k}': r.predict(model, v, lo, span, a.device) for k, v in ext.items()})
    member = {'tag': 'refit' if j is None else f'inner{j}', 'init_checkpoint_sha256': init_sha,
              'excluded_from_bag': bool((np.std(train_raw, axis=0) < .01).any()),
              'checkpoint_sha256': sha(folder / 'latest.pt')}
    return member, h


def select(histories):
    curves = np.array([[row['validation_person_mae'] for row in h] for h in histories])
    w = np.array([h[0]['validation_people'] for h in histories], dtype=float)
    means = np.average(curves, axis=0, weights=w)
    return int(np.argmin(means)) + 1, means.tolist()


def train_fold(d, c, f, ext):
    dest = a.data_root / 'runs' / a.family / f'fold{f}'
    if (dest / 'complete.json').exists():
        return
    members, histories = [], []
    if a.family == 'bilateral':
        sel = read(a.data_root / 'runs' / 'STT' / f'fold{f}' / 'selection.json')
        epoch, means = sel['selected_epoch'], None
    else:
        for j in range(5):
            if j == f:
                continue
            res = one_fit(d, c, f, j, c['max_epochs'], dest / f'inner{j}', ext, 'inner_epoch_selection')
            if res is not None:
                members.append(res[0])
                histories.append(res[1])
        assert len(histories) >= 3, 'Too few inner members'
        epoch, means = select(histories)
    r.save(dest / 'selection.json', {'selected_epoch': epoch, 'inner_mean_person_mae_by_epoch': means,
                                     'inner_members_used': len(histories), 'source': 'STT selection' if a.family == 'bilateral' else 'own inner curves'})
    res = one_fit(d, c, f, None, epoch, dest / 'refit', ext, 'outer_refit')
    members.append(res[0])
    r.save(dest / 'complete.json', {'bundle_sha256': sha(HERE / 'manifest.json'), 'family': a.family, 'fold': f,
                                    'selected_epoch': epoch, 'members': members})
    print('FOLD_COMPLETE', a.family, f, flush=True)


def pilot(d, c, ext):
    f, j = 0, (None if a.family == 'bilateral' else 1)
    ck, init_sha = baseline_state(f, 'refit' if j is None else f'inner{j}')
    tr, va, te = r.split(d, f, j)
    assert same_scale(ck, d['y'][tr])
    patch_init(ck['model'])
    out = a.data_root / 'runs' / 'pilot' / a.family
    c = dict(c, max_epochs=2)
    vx, vy, vi = (None, None, None) if j is None else (d['x'][va], d['y'][va], d['person_id'][va])
    model, lo, span, h = r.fit(d['x'][tr], d['y'][tr], d['person_id'][tr], vx, vy, vi, c, 2, out,
                               identity(d, c, f, j, 2, 'pilot', init_sha), a.device)
    va = te if j is None else va
    # starting point check: the patched model before any step reproduces the baseline member
    probe = (r.GdiCNN(c['dropout']) if a.family == 'CNN' else r.new_model()).to(a.device)
    base_raw = r.predict(probe, d['x'][va], ck['lo'], ck['span'], a.device)
    ref = (r._orig_GdiCNN(c['dropout']) if a.family == 'CNN' else r._orig_new_model())
    ref.load_state_dict(ck['model'])
    ref.to(a.device)
    start_err = float(np.max(np.abs(base_raw - r.predict(ref, d['x'][va], ck['lo'], ck['span'], a.device))))
    assert start_err < 1e-4, start_err
    e = {k: r.predict(model, v, lo, span, a.device) for k, v in ext.items()}
    r.save(out / 'verification.json', {'passed': True, 'family': a.family, 'start_matches_baseline_max_error': start_err,
                                       'scale_matches_baseline': True, 'epoch_seconds': [row['seconds'] for row in h],
                                       'validation_mae_by_epoch': [row.get('validation_person_mae') for row in h],
                                       'external_windows': {k: int(len(v)) for k, v in e.items()}})


d = load_data()
c = read(HERE / a.family / 'protocol.json')
c.update(FT)
r.setup(c['seed'], a.device)
if a.action == 'check':
    tr, _, _ = r.split(d, 0, 1)
    ck, _ = baseline_state(0, 'inner1') if a.family != 'bilateral' else baseline_state(0, 'refit')
    if a.family == 'bilateral':
        tr, _, _ = r.split(d, 0, None)
    print(json.dumps({'family': a.family, 'records': len(d['y']), 'scale_matches': bool(same_scale(ck, d['y'][tr])),
                      'members': sum(1 for k in BASE if k[0] == a.family), 'settings': FT}))
    load_external()
elif a.action == 'pilot':
    pilot(d, c, load_external())
else:
    assert (HERE / 'approved.json').exists()
    ext = load_external()
    for f in [int(x) for x in a.folds.split(',')]:
        train_fold(d, c, f, ext)
