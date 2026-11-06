"""Read-only real-input audit plus synthetic CPU engineering checks, never efficacy."""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import sys
import time
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'outputs/motionbert-smoke-20261015/deps'))
sys.path.insert(0, str(HERE / 'bundle'))
import torch
import run as r
from model import GdiCNN, SameConv, channels


def reject(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid case was accepted')


def main():
    out = HERE / 'checks' / ('local-%d' % time.time_ns())
    out.mkdir(parents=True)
    started = time.perf_counter()
    r.setup(321, 'cpu')
    d = r.data()
    c = r.config()
    assert c['max_epochs'] == 100
    with np.load(HERE.parent / 'expansion695/train_only.npz', allow_pickle=False) as z:
        assert np.array_equal(d['x'], z['x'][:, 0])
        for k in ('y', 'person_id', 'video_id', 'fold', 'cohort'):
            assert np.array_equal(d[k], z[k])
    print('Real-input identity, labels and all 25 splits: passed', flush=True)
    bad = dict(d, y=d['y'].copy())
    bad['y'][0, 0] = np.nan
    reject(lambda: r.validate_data(bad))
    bad = dict(d, person_id=d['person_id'].copy())
    a = np.flatnonzero(d['fold'] == 0)[0]
    b = np.flatnonzero(d['fold'] == 1)[0]
    bad['person_id'][a] = bad['person_id'][b]
    reject(lambda: r.split(bad, 0, 1))
    reject(lambda: r.split(d, 0, 0))
    reject(lambda: channels(torch.zeros(2, 17, 124, 3)))
    reject(lambda: GdiCNN()(torch.zeros(2, 124, 51)))
    x = torch.arange(2 * 124 * 17 * 3).reshape(2, 124, 17, 3).float()
    mapped = channels(x)
    for t, j, k in ((0, 0, 0), (100, 7, 1), (123, 16, 2)):
        assert torch.equal(mapped[:, 3*j+k, t], x[:, t, j, k])
    conv = SameConv(1, 1, 8, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1)
    original = np.arange(12, dtype=np.float32)
    expected = np.convolve(np.pad(original, (3, 4)), np.ones(8), mode='valid')
    np.testing.assert_allclose(conv(torch.from_numpy(original)[None, None]).detach().numpy()[0, 0], expected)
    p = np.array(['a', 'a', 'b'])
    y = np.zeros((3, 2))
    assert r.person_mae(y, np.array([[1, 1], [3, 3], [10, 10]]), p) == 6
    curves = [[{'validation_person_mae': e} for e in [3, 2, 2]] for _ in range(4)]
    assert r.select_epoch(curves)[0] == 2
    reject(lambda: r.select_epoch(curves[:3]))
    lo, span = r.scale_from_train(np.array([[40., 70.], [80., 90.]]))
    np.testing.assert_array_equal(lo.numpy(), [40, 70])
    np.testing.assert_array_equal(span.numpy(), [40, 20])
    tr, _, te = r.split(d, 0)
    changed_labels = d['y'].copy()
    changed_labels[te] = 9999
    for a, b in zip(r.scale_from_train(d['y'][tr]), r.scale_from_train(changed_labels[tr])):
        assert torch.equal(a, b)
    r.smoke(c, out / 'smoke', 'cpu')
    print('Axes, SAME convolution, metrics, rejected inputs and synthetic step resume: passed', flush=True)
    rng = np.random.default_rng(20261030)
    sx = rng.uniform(-1, 1, (64, 124, 17, 3)).astype(np.float32)
    sy = rng.uniform(40, 100, (64, 2))
    sp = np.array(['s%03d' % i for i in range(64)])
    ident = {'fit_seed': 77, 'epochs': 2, 'purpose': 'synthetic_resume_test'}
    with (out / 'synthetic_fit.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        full, _, _, _ = r.fit(sx, sy, sp, sx[:16], sy[:16], sp[:16], c, 2, out / 'full', ident, 'cpu')
        r.fit(sx, sy, sp, sx[:16], sy[:16], sp[:16], c, 1, out / 'resume', ident, 'cpu')
        resumed, _, _, _ = r.fit(sx, sy, sp, sx[:16], sy[:16], sp[:16], c, 2, out / 'resume', ident, 'cpu')
        for k, v in full.state_dict().items():
            assert torch.equal(v, resumed.state_dict()[k]), k
        reject(lambda: r.fit(sx, sy, sp, None, None, None, c, 2, out / 'resume', dict(ident, fit_seed=78), 'cpu'))
    print('Epoch-boundary interruption/resume: exact match', flush=True)
    # Complete nested workflow, using only generated coordinates and generated labels.
    # Real metadata is retained solely to exercise the established fold dimensions.
    sd = {k: v for k, v in d.items() if k not in ('x', 'y')}
    sd['x'] = rng.uniform(-1, 1, d['x'].shape).astype(np.float32)
    sd['y'] = rng.uniform(40, 100, d['y'].shape)
    sc = dict(c, max_epochs=1)
    workflow = out / 'synthetic_workflow'
    reject(lambda: r.summarize(sd, sc, workflow))
    for f in range(5):
        with (out / 'workflow.log').open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            r.train_fold(sd, sc, f, workflow, 'cpu', None)
        print('Synthetic nested workflow fold %d/5 complete' % (f + 1), flush=True)
    with (out / 'workflow.log').open('a', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        r.summarize(sd, sc, workflow)
        r.export_review(workflow)
    with np.load(workflow / 'oof_predictions.npz', allow_pickle=False) as z:
        pred = z['predictions'][:, 0]
        assert pred.shape == (1064, 2) and np.isfinite(pred).all()
        independently = sum(float(np.abs(pred[sd['person_id'] == person] - sd['y'][sd['person_id'] == person]).mean())
                            for person in set(sd['person_id'])) / 695
    summary = r.load(workflow / 'summary.json')
    assert summary['fits'] == 25 and abs(summary['metrics'][0]['mae'] - independently) < 1e-10
    for f in range(5):
        ck = torch.load(workflow / ('fold%d/refit/latest.pt' % f), map_location='cpu', weights_only=True)
        model = GdiCNN(c['dropout'])
        model.load_state_dict(ck['model'])
        mask = sd['fold'] == f
        replay = r.predict(model, channels(torch.from_numpy(sd['x'][mask])), ck['lo'], ck['span'], 'cpu')
        assert np.array_equal(replay, pred[mask])
    # Corrupted completed predictions must be refused, then restore this test fixture.
    target = workflow / 'fold0/predictions.npz'
    content = target.read_bytes()
    target.write_bytes(content + b'corruption-test')
    reject(lambda: r.summarize(sd, sc, workflow))
    target.write_bytes(content)
    for path in (HERE / 'bundle').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 8))
    with zipfile.ZipFile(HERE / 'gdi-cnn695.zip') as z:
        assert z.testzip() is None
        for name, digest in r.load(HERE / 'bundle/manifest.json').items():
            assert hashlib.sha256(z.read('gdi-cnn695-20261030/' + name)).hexdigest() == digest
    proof = {'passed': True, 'seconds': time.perf_counter() - started, 'torch': str(torch.__version__),
             'device': 'cpu', 'bundle_sha256': r.sha(HERE / 'bundle/manifest.json'),
             'real_data_optimizer_steps': 0, 'actual_gpu_execution': False,
             'real_input_exact_match': True, 'all_splits_person_disjoint': True,
             'synthetic_nested_fits': 25, 'synthetic_nested_optimizer_steps': summary['optimizer_steps'],
             'synthetic_complete_oof_records': 1064, 'exact_checkpoint_prediction_replay': True,
             'exact_epoch_resume': True, 'person_macro_independent_check': True,
             'reject_nonfinite_leakage_wrong_axes_wrong_identity_incomplete_and_corrupt_outputs': True,
             'python38_syntax_checked': True, 'zip_contents_verified': True,
             'details': out.relative_to(HERE).as_posix()}
    r.save(HERE / 'local_verification.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
