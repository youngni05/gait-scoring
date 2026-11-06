"""Read-only review of a completed variant; original results remain unchanged."""
from pathlib import Path
import argparse, csv, json, shutil, sys, tarfile
import numpy as np
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H/'bundle'))
import run as r
import audit
import torch

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['window_control', 'record_mean'], required=True)
p.add_argument('--receipt-sha256', required=True)
p.add_argument('--source-audit', type=Path)
p.add_argument('--source-audit-sha256')
p.add_argument('--cpu64-reference', action='store_true')
p.add_argument('--received', type=Path)
p.add_argument('--working-root', type=Path)
a = p.parse_args()
v = a.variant
received = a.received or H/('received-'+v.replace('_','-')+'-20261031')
out = H/'analysis'/v
out.mkdir(parents=True, exist_ok=True)
receipt_path = received/('receipt-'+v+'-20261031.json')
assert r.sha(receipt_path) == a.receipt_sha256
receipt = r.load(receipt_path)
work = a.working_root or out
work.mkdir(parents=True, exist_ok=True)
unpacked, checkpoints = work/'unpacked', work/'checkpoints'
for remote, digest in receipt['files'].items():
    if remote.endswith('.tar.gz'):
        path = received/Path(remote).name
        assert r.sha(path) == digest
        dest = checkpoints if path.name.startswith('checkpoints-') else unpacked
        dest.mkdir(exist_ok=True)
        # Previously extracted files are checked below against the receipt,
        # bundle, checkpoint and epoch hashes before any result is accepted.
        ready = (dest/'results/summary.json').exists() if dest==unpacked else len(list((dest/'runs/main'/v).glob('fold*/*/latest.pt')))==25
        if not ready:
            with tarfile.open(path) as tf:
                tf.extractall(dest, filter='data')
for remote, digest in receipt['files'].items():
    if not remote.endswith('.tar.gz'):
        assert r.sha(checkpoints/remote) == digest
results = unpacked/'results'
for path in (checkpoints/'runs/main'/v).glob('fold*/*/latest.pt'):
    dest = results/path.relative_to(checkpoints/'runs/main'/v)
    shutil.copyfile(path, dest)
for path in (unpacked/'bundle').iterdir():
    assert r.sha(path) == r.sha(H/'bundle'/path.name)
r.save(out/'transfer_verification.json', dict(passed=True, receipt_sha256=a.receipt_sha256, files=receipt['files']))
d = r.data()
c = r.config()
c.update(loss_variant=v, name='cnn_'+v+'_packed_records')
r.setup(c['seed'], 'cpu')
summary = r.load(results/'summary.json')
assert summary == receipt['summary']
assert r.sha(results/'oof_predictions.npz') == summary['oof_sha256']

def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def close(x, y, tol=1e-10):
    np.testing.assert_allclose(x, y, atol=tol, rtol=0)

def predict64(model,x,lo,span):
    model.double().eval()
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(x.values),128):
            values=r.channels(torch.from_numpy(x.values[start:start+128]).double())
            batch=values.reshape(-1,12,124)
            scaled=model(batch)[:,0].reshape(-1,2)*float(span)+float(lo)
            parts.append(scaled.numpy())
    return x.aggregate(np.concatenate(parts))

pred = np.full(d['y'].shape, np.nan)
raw = np.full_like(pred, np.nan)
seen = np.zeros(len(pred), int)
fits, replays = [], []
source_audit = None
if a.source_audit:
    assert r.sha(a.source_audit) == a.source_audit_sha256
    source_audit = r.load(a.source_audit)
    assert source_audit['passed'] and source_audit['original_audit_fit_unchanged']
    assert source_audit['summary_sha256'] == r.sha(results/'summary.json')
    assert len(source_audit['fits']) == 25

def audit_history(f, j, folder, epochs, purpose):
    if source_audit is None:
        return audit.audit_fit(r,d,c,f,j,folder,epochs,purpose)
    # Keep the failed local strict calculation. Verify that the unmodified
    # auditor passed on THESE files in the original NumPy environment.
    proof = [x for x in source_audit['fits'] if x['fold']==f and x['inner']==j]
    assert len(proof)==1 and proof[0]['passed'] and proof[0]['epochs']==epochs
    done = r.load(folder/'fit_complete.json')
    h = r.load(folder/'curve.json')
    assert r.sha(folder/'curve.json') == done['curve_sha256'] == proof[0]['curve_sha256']
    assert r.sha(folder/'latest.pt') == done['checkpoint_sha256'] == proof[0]['checkpoint_sha256']
    ck = torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
    assert ck['history']==h and ck['identity']==done['identity']==r.load(folder/'identity.json')
    for key,val in r.identity_for(d,c,f,j,epochs,purpose).items():
        assert ck['identity'][key]==val
    tr,va,_=r.split(d,f,j)
    lo,span=r.scale_from_train(d['y'][tr])
    assert torch.equal(lo,ck['lo']) and torch.equal(span,ck['span'])
    assert len(h)==done['epochs']==epochs
    assert done['optimizer_steps']==epochs*int(np.ceil(tr.sum()/c['batch_records']))
    for k,row in enumerate(h,1):
        assert row['epoch']==k
        path=folder/'raw_predictions'/('epoch%04d.npz'%k)
        assert r.sha(path)==row['raw_predictions_sha256']
        z=arrays(path)
        assert z['train_raw'].shape==d['y'][tr].shape and np.isfinite(z['train_raw']).all()
        # Metric replay uses stored coefficients; their training-only fitting
        # was verified with unchanged 1e-10 tolerance by the source auditor.
        close(r.person_mae(d['y'][tr],z['train_raw'],d['person_id'][tr]),row['train_raw_person_mae'])
        close(r.person_mae(d['y'][tr],r.corrected(z['train_raw'],row['calibration']),d['person_id'][tr]),row['train_person_mae'])
        if j is None:
            assert z['validation_raw'].shape==(0,2) and 'validation_person_mae' not in row
        else:
            assert row['validation_people']==len(np.unique(d['person_id'][va]))
            close(r.person_mae(d['y'][va],z['validation_raw'],d['person_id'][va]),row['validation_raw_person_mae'])
            close(r.person_mae(d['y'][va],r.corrected(z['validation_raw'],row['calibration']),d['person_id'][va]),row['validation_person_mae'])
    return h

for f in range(5):
    dest = results/('fold%d'%f)
    histories = []
    for j in range(5):
        if j == f:
            continue
        folder = dest/('inner%d'%j)
        histories.append(audit_history(f,j,folder,c['max_epochs'],'inner_epoch_selection'))
        fits.append(r.load(folder/'fit_complete.json'))
    epoch, means = r.select_epoch(histories)
    selection, done = r.load(dest/'selection.json'), r.load(dest/'complete.json')
    assert epoch == selection['selected_epoch'] == done['selected_epoch']
    assert selection['outer_evaluation_used_for_selection'] is False
    close(means, selection['inner_mean_person_mae_by_epoch'])
    assert r.sha(dest/'selection.json') == done['selection_sha256']
    assert r.sha(dest/'predictions.npz') == done['prediction_sha256']
    assert r.sha(dest/'refit/latest.pt') == done['checkpoint_sha256']
    h = audit_history(f,None,dest/'refit',epoch,'outer_refit')
    fits.append(r.load(dest/'refit/fit_complete.json'))
    assert done['identity'] == r.identity_for(d,c,f,None,epoch,'outer_refit')
    tr, _, te = r.split(d,f)
    z = arrays(dest/'predictions.npz')
    np.testing.assert_array_equal(z['row_index'], np.flatnonzero(te))
    for key in ('person_id','video_id','fold','cohort','y'):
        np.testing.assert_array_equal(z[key], d[key][te])
    assert z['predictions'].shape == z['predictions_raw'].shape == (int(te.sum()),1,2)
    pred[te], raw[te] = z['predictions'][:,0], z['predictions_raw'][:,0]
    seen[te] += 1
    close(z['calibration'], h[-1]['calibration'])
    close(pred[te], r.corrected(raw[te],z['calibration']))
    close(r.person_mae(d['y'][te],pred[te],d['person_id'][te]), done['evaluation_person_mae'])
    close(r.person_mae(d['y'][te],raw[te],d['person_id'][te]), done['raw_evaluation_person_mae'])
    ck = torch.load(dest/'refit/latest.pt', map_location='cpu', weights_only=True)
    model = r.GdiCNN(c['dropout'])
    model.load_state_dict(ck['model'], strict=True)
    cpu_raw = r.predict(model,d['x'][te],ck['lo'],ck['span'],'cpu')
    cpu_train = r.predict(model,d['x'][tr],ck['lo'],ck['span'],'cpu')
    saved_train = arrays(dest/'refit/raw_predictions'/('epoch%04d.npz'%epoch))['train_raw']
    coef = r.calibrate(cpu_train,d['y'][tr])
    cpu_pred = r.corrected(cpu_raw,coef)
    close(cpu_raw,raw[te],1e-4)
    close(cpu_train,saved_train,1e-4)
    float32_error=float(np.max(np.abs(cpu_pred-pred[te])))
    replay_precision='float32'
    try:
        close(cpu_pred,pred[te],1e-4)
    except AssertionError:
        r.save(out/('float32-replay-failure-fold%d.json'%f),dict(passed=False,fold=f,
            corrected_max_error=float32_error,tolerance=1e-4,checkpoint_sha256=r.sha(dest/'refit/latest.pt')))
        if not a.cpu64_reference:
            raise
        cpu_raw=predict64(model,d['x'][te],ck['lo'],ck['span'])
        cpu_train=predict64(model,d['x'][tr],ck['lo'],ck['span'])
        coef=r.calibrate(cpu_train,d['y'][tr])
        cpu_pred=r.corrected(cpu_raw,coef)
        close(cpu_raw,raw[te],1e-4)
        close(cpu_train,saved_train,1e-4)
        close(cpu_pred,pred[te],1e-4)
        replay_precision='float64_reference'
    replay = dict(fold=f, selected_epoch=epoch, raw_max_error=float(np.max(np.abs(cpu_raw-raw[te]))),
        corrected_max_error=float(np.max(np.abs(cpu_pred-pred[te]))), calibration=z['calibration'].tolist(),
        last_train_mae=h[-1]['train_person_mae'], outer_mae=done['evaluation_person_mae'],
        replay_precision=replay_precision,float32_corrected_max_error=float32_error)
    replays.append(replay)
    print(json.dumps(dict(replayed=replay)), flush=True)
assert (seen == 1).all() and len(fits)==25
oof = arrays(results/'oof_predictions.npz')
for key in ('person_id','video_id','fold','cohort','y','original_row_index'):
    np.testing.assert_array_equal(oof[key],d[key])
np.testing.assert_array_equal(oof['predictions'][:,0],pred)
np.testing.assert_array_equal(oof['predictions_raw'][:,0],raw)
for metric, values in [(summary['metrics'][0],pred),(summary['raw_diagnostic_metrics'],raw)]:
    for key,val in r.metrics(d['y'],values,d['person_id']).items():
        close(val,metric[key])
for f in range(5):
    mask=d['fold']==f
    for key,val in r.metrics(d['y'][mask],pred[mask],d['person_id'][mask]).items():
        close(val,summary['fold_metrics'][f][key])
assert sum(x['optimizer_steps'] for x in fits)==summary['optimizer_steps']
assert summary['bundle_sha256']==r.sha(H/'bundle/manifest.json')
assert summary['independent_final_evaluation'] is False
comparison = {v:summary['metrics'][0], v+'_raw':summary['raw_diagnostic_metrics']}
for label,path in [('multiwindow_CNN',H.parent/'cnn-multiwindow695/analysis/cnn_oof.npz'),
                   ('multiwindow_STT',H.parent/'stt692/multiwindow/analysis/unpacked/results/oof_predictions.npz')]:
    z=arrays(path); index={x:i for i,x in enumerate(z['video_id'])}
    assert len(index)==1061
    ix=np.array([index[x] for x in d['video_id']])
    for key in ('person_id','fold','y'):
        np.testing.assert_array_equal(z[key][ix],d[key])
    comparison[label]=r.metrics(d['y'],z['predictions'][ix,0],d['person_id'])
high = np.unravel_index(np.argmax(pred),pred.shape)
maximum = dict(fold=int(d['fold'][high[0]]), predicted=float(pred[high]),raw=float(raw[high]),actual=float(d['y'][high]))
r.save(out/'comparison.json',dict(metrics=comparison,replay=replays,max_prediction_diagnostic=maximum))
r.save(out/'verification.json',dict(passed=True,variant=v,people=692,records=1061,windows=11261,fits=25,
    epochs_checked=sum(x['epochs'] for x in fits),optimizer_steps=summary['optimizer_steps'],
    original_audit_fit_passed=25,original_audit_environment='source NumPy '+source_audit['numpy'] if source_audit else 'local',
    source_audit_sha256=a.source_audit_sha256,source_environment_audit_used=bool(source_audit),
    all_training_only_normalization_calibration_checked=True,
    fixed_person_splits_checked=True,inner_epoch_selection_recomputed=True,person_equal_metrics_recomputed=True,
    outer_checkpoints_replayed=5,cpu_replay_tolerance=1e-4,cpu_replay_max_error=max(x['corrected_max_error'] for x in replays),
    protected_final_evaluation_opened=False,local_optimizer_steps=0))
print(json.dumps(dict(passed=True,metrics=comparison,max_prediction=maximum)),flush=True)
