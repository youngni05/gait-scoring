"""Read-only audit of multiwindow STT receipts; writes derived evidence in analysis only."""
from pathlib import Path
import csv, json, math, shutil, sys, tarfile
import numpy as np
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H / 'bundle'))
import run as r
import torch

received = H / 'received-full-20261031'
assert r.sha(received/'receipt-20261031.json')=='70536c761093213c44a9e3dca2ac1e04fbd834415053c8a1b6c4589e6bac8318'
receipt=r.load(received/'receipt-20261031.json')
EXPECTED={Path(k).name:v for k,v in receipt['files'].items() if k.endswith('.tar.gz')}
out = H / 'analysis'
out.mkdir(exist_ok=True)
def close(a, b, tol=1e-10):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol)
def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}
def macro_mae(y, p, ids):
    _, inv, count = np.unique(ids, return_inverse=True, return_counts=True)
    return float(np.sum(np.abs(y - p).mean(axis=1) / count[inv]) / len(count))

for name, digest in EXPECTED.items():
    assert r.sha(received / name) == digest, name
r.save(out / 'transfer_verification.json', dict(passed=True, server_and_local_sha256=EXPECTED))
# Extraction is into a derived directory, preserving the received archive.
unpacked = out / 'unpacked'
unpacked.mkdir(exist_ok=True)
if not (unpacked / 'bundle/sources.json').exists():
    with tarfile.open(received / 'main-review.tar.gz') as tf:
        tf.extractall(unpacked, filter='data')
results = unpacked / 'results'
checkpoint_root=out/'checkpoints'
checkpoint_root.mkdir(exist_ok=True)
with tarfile.open(received/'checkpoints-review-20261031.tar.gz') as tf:tf.extractall(checkpoint_root,filter='data')
for name,digest in receipt['files'].items():
    if name.endswith('.tar.gz'):continue
    assert r.sha(checkpoint_root/name)==digest,name
for p in (checkpoint_root/'runs/main').glob('fold*/*/latest.pt'):
    dest=results/p.relative_to(checkpoint_root/'runs/main');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)

for p in (unpacked / 'bundle').iterdir():
    assert r.sha(p) == r.sha(H / 'bundle' / p.name), p.name
assert 'STT_TRAINING_AND_SUMMARY_FINISHED' in (checkpoint_root / 'logs/worker-32844.log').read_text()
d = r.data()
c = r.load(H / 'bundle/protocol.json')
r.setup(c['seed'], 'cpu')
summary = r.load(results / 'summary.json')
assert r.sha(results / 'oof_predictions.npz') == summary['oof_sha256']
pred = np.full_like(d['y'], np.nan)
raw = np.full_like(pred, np.nan)
seen = np.zeros(len(pred), int)
fits, all_curves, replay, original_audit_checks = [], [], [], []
calibration_errors = []
steps = 0

def audit(f, j, epochs, folder, purpose):
    """Independently audit every epoch; all25 checkpoint hashes are also checked.

    The original audit_fit is executed unchanged with each actual checkpoint.
    Keep cross-environment exact coefficient failures separate from this audit.
    """
    tr, va, _ = r.split(d, f, j)
    ident = r.identity_for(d, c, f, j, epochs, purpose)
    for prefix, mask in [('train', tr), ('validation', va)]:
        for field, key in [('x', 'x'), ('y', 'y'), ('ids', 'person_id')]:
            ident[prefix + '_' + field + '_sha'] = None if prefix == 'validation' and j is None else r.digest_array(d[key][mask])
    done = r.load(folder / 'fit_complete.json')
    h = r.load(folder / 'curve.json')
    assert ident == done['identity'] == r.load(folder / 'identity.json')
    assert len(h) == done['epochs'] == epochs
    assert done['curve_sha256'] == r.sha(folder / 'curve.json')
    assert done['optimizer_steps'] == epochs * math.ceil(2 * len(d['x'][tr].values) / c['batch_size'])
    for epoch, row in enumerate(h, 1):
        assert row['epoch'] == epoch
        path = folder / 'raw_predictions' / ('epoch%04d.npz' % epoch)
        assert r.sha(path) == row['raw_predictions_sha256']
        z = arrays(path)
        a, b = z['train_raw'], z['validation_raw']
        assert a.shape == (int(tr.sum()), 2) and np.isfinite(a).all()
        coef = r.calibration(a, d['y'][tr])
        calibration_errors.append(float(np.max(np.abs(coef-row['calibration']))))
        close(coef, row['calibration'])
        close(coef, [row['calibration_slope'], row['calibration_intercept']])
        close(macro_mae(d['y'][tr], a, d['person_id'][tr]), row['train_raw_person_mae'])
        close(macro_mae(d['y'][tr], r.corrected(a, coef), d['person_id'][tr]), row['train_person_mae'])
        expected_lr = c['eta_min'] + (c['learning_rate'] - c['eta_min']) * (1 + math.cos(math.pi * ((epoch - 1) % 40) / 40)) / 2
        close(expected_lr, row['lr'])
        if j is None:
            assert b.shape == (0, 2) and 'validation_person_mae' not in row
        else:
            assert b.shape == (int(va.sum()), 2) and np.isfinite(b).all()
            assert row['validation_people'] == len(np.unique(d['person_id'][va]))
            close(macro_mae(d['y'][va], b, d['person_id'][va]), row['validation_raw_person_mae'])
            close(macro_mae(d['y'][va], r.corrected(b, coef), d['person_id'][va]), row['validation_person_mae'])
        all_curves.append(dict(outer_fold=f, fit=folder.name, **row))
    fits.append(dict(outer_fold=f, fit=folder.name, epochs=epochs, steps=done['optimizer_steps'],
        first_slope=h[0]['calibration_slope'], last_slope=h[-1]['calibration_slope'],
        min_slope=min(v['calibration_slope'] for v in h), max_slope=max(v['calibration_slope'] for v in h),
        first_positive_epoch=next((v['epoch'] for v in h if v['calibration_slope'] > 0), None),
        last_intercept=h[-1]['calibration_intercept'], last_train_mae=h[-1]['train_person_mae'],
        last_validation_mae=h[-1].get('validation_person_mae')))
    if j is not None:
        try:
            r.audit_fit(d,c,f,j,epochs,folder,purpose)
            original_audit_checks.append(dict(fold=f,inner=j,passed=True))
        except ValueError as error:
            if str(error)!='Coefficient mismatch':raise
            original_audit_checks.append(dict(fold=f,inner=j,passed=False,reason=str(error),independent_full_epoch_audit_passed=True))
    return h, done['optimizer_steps']

for f in range(5):
    dest = results / ('fold%d' % f)
    histories = []
    for j in range(5):
        if j != f:
            h, n = audit(f, j, c['max_epochs'], dest / ('inner%d' % j), 'inner_epoch_selection')
            histories.append(h)
            steps += n
    epoch, means = r.select_epoch(histories)
    selection = r.load(dest / 'selection.json')
    done = r.load(dest / 'complete.json')
    assert epoch == selection['selected_epoch'] == done['selected_epoch']
    assert selection['outer_evaluation_used_for_selection'] is False
    close(means, selection['inner_mean_person_mae_by_epoch'])
    h, n = audit(f, None, epoch, dest / 'refit', 'outer_refit')
    steps += n
    assert r.sha(dest / 'refit/latest.pt') == done['checkpoint_sha256'] == r.load(dest / 'refit/fit_complete.json')['checkpoint_sha256']
    # Checkpoints copied into derived results above.
    # Original unmodified auditor, including checkpoint integrity, for all refits.
    try:
        r.audit_fit(d, c, f, None, epoch, dest / 'refit', 'outer_refit')
        original_audit_checks.append(dict(fold=f, passed=True))
    except ValueError as error:
        # Exact float equality is not portable between server NumPy 1.24.4 and
        # local 2.3.5. Do not modify the original check or claim that it passed.
        # The independent audit above recomputes ALL epochs using the original
        # auditor's existing 1e-10 coefficient/metric tolerance.
        if str(error) != 'Coefficient mismatch':
            raise
        original_audit_checks.append(dict(fold=f, passed=False, reason=str(error),
            independent_full_epoch_audit_passed=True, coefficient_tolerance=1e-10))
    assert r.sha(dest / 'selection.json') == done['selection_sha256']
    assert r.sha(dest / 'predictions.npz') == done['prediction_sha256']
    assert done['identity'] == r.load(dest / 'refit/identity.json')
    tr, _, te = r.split(d, f, None)
    z = arrays(dest / 'predictions.npz')
    np.testing.assert_array_equal(z['row_index'], np.flatnonzero(te))
    for key in ('y', 'person_id', 'video_id', 'fold', 'cohort', 'original_row_index'):
        np.testing.assert_array_equal(z[key], d[key][te])
    assert z['predictions'].shape == z['predictions_raw'].shape == (int(te.sum()), 1, 2)
    pred[te], raw[te] = z['predictions'][:, 0], z['predictions_raw'][:, 0]
    seen[te] += 1
    close(z['calibration'], h[-1]['calibration'])
    close(pred[te], r.corrected(raw[te], z['calibration']))
    close(r.metric(d['y'][te], pred[te], d['person_id'][te]), done['evaluation_person_mae'])
    close(r.metric(d['y'][te], raw[te], d['person_id'][te]), done['raw_evaluation_person_mae'])
    ck = torch.load(dest / 'refit/latest.pt', map_location='cpu', weights_only=True)
    assert ck['identity'] == done['identity'] and ck['history'] == h
    assert ck['lo'] == float(np.float32(d['y'][tr].min()))
    assert ck['span'] == float(np.float32(max(float(np.ptp(d['y'][tr])), 1e-6)))
    m = r.new_model()
    m.load_state_dict(ck['model'], strict=True)
    cpu_raw = r.predict(m, d['x'][te], ck['lo'], ck['span'], 'cpu')
    cpu_train = r.predict(m, d['x'][tr], ck['lo'], ck['span'], 'cpu')
    saved_train = arrays(dest / 'refit/raw_predictions' / ('epoch%04d.npz' % epoch))['train_raw']
    close(cpu_raw, raw[te], 1e-4)
    close(cpu_train, saved_train, 1e-4)
    cpu_coef = r.calibration(cpu_train, d['y'][tr])
    cpu_pred = r.corrected(cpu_raw, cpu_coef)
    close(cpu_pred, pred[te], 1e-4)
    replay.append(dict(fold=f, selected_epoch=epoch, slope=float(z['calibration'][0]), intercept=float(z['calibration'][1]),
        raw_max_error=float(np.max(np.abs(cpu_raw - raw[te]))), corrected_max_error=float(np.max(np.abs(cpu_pred-pred[te]))),
        train_raw_max_error=float(np.max(np.abs(cpu_train - saved_train))), calibration_max_error=float(np.max(np.abs(cpu_coef-z['calibration'])))))
    print(json.dumps(dict(audited_fold=f, selected_epoch=epoch, replay=replay[-1])), flush=True)

assert (seen == 1).all() and len(fits) == 25 and len(all_curves) == 4702 and steps == 523725
oof = arrays(results / 'oof_predictions.npz')
for key in ('y', 'person_id', 'video_id', 'fold', 'cohort', 'original_row_index'):
    np.testing.assert_array_equal(oof[key], d[key])
np.testing.assert_array_equal(oof['predictions'][:, 0], pred)
np.testing.assert_array_equal(oof['predictions_raw'][:, 0], raw)
for label, value in [('metrics', pred), ('raw_diagnostic_metrics', raw)]:
    computed = r.metrics(d['y'], value, d['person_id'])
    for key, v in computed.items():
        close(v, summary[label][key])
for f in range(5):
    mask = d['fold'] == f
    for key, v in r.metrics(d['y'][mask], pred[mask], d['person_id'][mask]).items():
        close(v, summary['fold_metrics'][f][key])
assert summary['optimizer_steps'] == steps and summary['fits'] == 25 and summary['diagnostic_rows'] == len(all_curves)
assert summary['bundle_sha256'] == r.sha(H / 'bundle/manifest.json')
assert summary['independent_final_evaluation'] is False
with (results / 'diagnostic_curves.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(all_curves)
for a, b in zip(rows, all_curves):
    for key, val in a.items():
        if key in ('fit', 'raw_predictions_sha256'):
            assert val == b[key]
        elif val == '':
            assert key not in b
        else:
            close(float(val), b[key])
with (results / 'predictions.csv').open(newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 2122
for k, row in enumerate(rows):
    i, side = divmod(k, 2)
    assert row['person_id'] == str(d['person_id'][i]) and row['video_id'] == str(d['video_id'][i])
    assert int(row['fold']) == d['fold'][i] and row['side'] == ['left', 'right'][side]
    close([float(row[key]) for key in ('actual_gdi', 'raw_prediction', 'corrected_prediction')], [d['y'][i, side], raw[i, side], pred[i, side]])

comparison = {'STT_corrected': summary['metrics'], 'STT_raw_diagnostic': summary['raw_diagnostic_metrics']}
paired = {}
for label, name in [('single_CNN', 'cnn-paper-input695'), ('multiwindow_CNN', 'cnn-multiwindow695'), ('single_STT', 'stt692/full')]:
    z = arrays(H.parent.parent / name / ('analysis/unpacked/results/oof_predictions.npz' if label=='single_STT' else 'analysis/cnn_oof.npz'))
    # Align by unique video id; do not assume reference arrays share row order.
    assert len(set(z['video_id'])) == 1061
    index = {v: i for i, v in enumerate(z['video_id'])}
    ix = np.array([index[v] for v in d['video_id']])
    for key in ('person_id', 'y', 'fold'):
        np.testing.assert_array_equal(z[key][ix], d[key])
    p = z['predictions'][ix, 0]
    comparison[label] = r.metrics(d['y'], p, d['person_id'])
    delta = [float(np.abs(pred[d['person_id']==q]-d['y'][d['person_id']==q]).mean()-np.abs(p[d['person_id']==q]-d['y'][d['person_id']==q]).mean()) for q in np.unique(d['person_id'])]
    paired[label] = dict(stt_minus_reference_mae=float(np.mean(delta)), relative_error_increase_pct=float(np.mean(delta)/comparison[label]['mae']*100),
        people_STT_better=int(np.sum(np.array(delta)<0)), people_STT_worse=int(np.sum(np.array(delta)>0)),
        reference_fold_mae=[r.metric(d['y'][d['fold']==f], p[d['fold']==f], d['person_id'][d['fold']==f]) for f in range(5)])
r.save(out / 'comparison.json', dict(metrics=comparison, paired=paired, fits=fits, replay=replay))
r.save(out / 'verification.json', dict(passed=True, people=692, records=1061, fits=25, epochs_checked=len(all_curves), optimizer_steps=steps,
    all_person_splits_checked=True, all_training_only_calibration_recomputed=True, inner_epoch_selection_recomputed=True,
    all_raw_and_corrected_metrics_recomputed=True, fixed_data_manifest_verified=True, outer_checkpoints_replayed=5,
    original_audit_fit_local=original_audit_checks, inner_checkpoint_downloaded=True,
    all_epoch_calibration_max_error=max(calibration_errors), coefficient_metric_tolerance=1e-10,
    inner_audit_scope='all 25 checkpoint hashes locally matched; identities, raw predictions, curves, selection and steps independently recomputed; five refits re-inferred',
    cpu_replay_tolerance=1e-4, cpu_replay_max_error=max(x['corrected_max_error'] for x in replay),
    protected_final_evaluation_opened=False, local_optimizer_steps=0))
print(json.dumps(dict(passed=True, comparison=comparison, paired=paired, replay=replay)), flush=True)
