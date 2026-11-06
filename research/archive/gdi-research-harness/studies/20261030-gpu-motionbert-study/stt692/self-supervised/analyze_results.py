"""Read-only audit of multiwindow STT receipts; writes derived evidence in analysis only."""
from pathlib import Path
import csv, json, math, shutil, sys, tarfile
import numpy as np
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H / 'bundle'))
import run as r
import torch

received = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-gpu-motionbert-study/stt692/self-supervised/received-full-20261031')
out=H/'analysis';out.mkdir(exist_ok=True)
proof=r.load(out/'transfer_verification.json')
assert proof['passed'] and r.sha(received/'receipt.json')==proof['receipt_sha256']
receipt=r.load(received/'receipt.json')
for name,digest in receipt['files'].items():
    path=received/Path(name).name if name.endswith('.tar.gz') else received/'checkpoints'/name
    assert r.sha(path)==digest,name
results=received/'unpacked/results'
for p in (received/'unpacked/bundle').iterdir():assert r.sha(p)==r.sha(H/'bundle'/p.name)
assert 'STT_TRAINING_AND_SUMMARY_FINISHED' in (received/'checkpoints/logs/worker-32844.log').read_text()
import pretrain as ssl
ssl_replays=[]
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,rtol=0,atol=tol)
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def macro_mae(y,p,ids):
    _,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    return float(np.sum(np.abs(y-p).mean(axis=1)/count[inv])/len(count))
def predict64(model,x,lo,span):
    model.double().eval();parts=[]
    with torch.inference_mode():
        for start in range(0,len(x.values),64):
            t=torch.from_numpy(x.values[start:start+64].reshape(-1,124,4,2)).double()
            parts.append((model(t)*span+lo).reshape(-1,2).numpy())
    return x.aggregate(np.concatenate(parts))
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
    # Independently replay every training-only SSL checkpoint on its stored probe.
    ss=ssl.audit(d['x'][tr],d['person_id'][tr],c,folder/'pretrain',ident['seed'])
    sk=torch.load(folder/'pretrain/latest.pt',map_location='cpu',weights_only=True)
    sm=ssl.SSL(r.new_model()).double();sm.load_state_dict(sk['model']);sm.eval()
    with torch.inference_mode():
        t=torch.from_numpy(d['x'][tr].values[:4].reshape(-1,124,4,2)).double()
        lc,fc,_,_=ssl.masked_inputs(t,42,c['ssl']['mask_frames'],c['ssl']['prefix_frames'])
        lp=sm(lc,'local').numpy();fp=sm(fc,'future').numpy()
    z=arrays(folder/'pretrain/probe.npz')
    err=max(float(abs(lp-z['local']).max()),float(abs(fp-z['future']).max()))
    assert err<1e-4,err
    transfer=r.load(folder/'pretrain/transfer.json')
    assert transfer['checkpoint_sha256']==ss['checkpoint_sha256'] and not transfer['decoder_transferred']
    assert set(transfer['transferred_keys'])=={k for k in r.new_model().state_dict() if ssl.encoder_key(k)}
    assert set(ss['identity']['train_people'])==set(d['person_id'][tr])
    ssl_replays.append(dict(fold=f,inner=j,probe_max_error=err,checkpoint_sha256=ss['checkpoint_sha256'],steps=ss['optimizer_steps'],epochs=ss['epochs']))
    ckfit=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    assert ckfit['identity']==ident
    assert ckfit['lo']==float(np.float32(d['y'][tr].min()))
    assert ckfit['span']==float(np.float32(max(float(np.ptp(d['y'][tr])),1e-6)))
    done = r.load(folder / 'fit_complete.json')
    h = r.load(folder / 'curve.json')
    assert ident == done['identity'] == r.load(folder / 'identity.json')
    assert len(h) == done['epochs'] == epochs and ckfit['history']==h
    assert r.sha(folder/'latest.pt')==done['checkpoint_sha256']
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
    cpu_raw = predict64(m, d['x'][te], ck['lo'], ck['span'])
    cpu_train = predict64(m, d['x'][tr], ck['lo'], ck['span'])
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

assert (seen == 1).all() and len(fits) == 25 and len(all_curves) == 4629 and steps == 513462
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


assert summary['pretraining_fits']==len(ssl_replays)==25
assert all(v['epochs']==25 for v in ssl_replays)
r.save(out/'verification.json',dict(passed=True,people=692,records=1061,windows=11261,fits=25,epochs_checked=len(all_curves),optimizer_steps=steps,
    pretraining_fits=25,pretraining_epochs=625,pretraining_steps=sum(x['steps'] for x in ssl_replays),ssl_replays=ssl_replays,
    all_person_splits_checked=True,training_only_pretraining=True,training_only_scale_calibration=True,
    inner_epoch_selection_recomputed=True,all_epoch_raw_calibrated_metrics_verified=True,
    original_audit_fit_local=original_audit_checks,calibration_max_error=max(calibration_errors),coefficient_metric_tolerance=1e-10,
    replay_arithmetic='float64 reference of unchanged float32 weights, as approved pilot',cpu_replay_tolerance=1e-4,
    replays=replay,cpu_replay_max_error=max(x['corrected_max_error'] for x in replay),
    protected_final_evaluation_opened=False,local_optimizer_steps=0,receipt_sha256=proof['receipt_sha256']))
r.save(out/'diagnostics.json',dict(fits=fits,replay=replay,metrics=summary['metrics'],raw_metrics=summary['raw_diagnostic_metrics']))
print(json.dumps(dict(passed=True,metrics=summary['metrics'],ssl_probes=25,replayed_gdi=5)),flush=True)
