"""Frozen MotionBERT temporal + geometry single-window CNN, retained 692-person cohort; Python 3.8+, NumPy, PyTorch 2.4+.

check: real data, no optimization. smoke: synthetic optimization only.
pilot: 3 epochs on outer-0/inner-1 training people (retained subset), not a performance result.
train: nested epoch selection, then outer refit; summarize: complete OOF only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import sys
import tarfile
import time
import numpy as np
import torch
from model import GdiCNN, channels, training_feature_stats
from features import validate_motion_inputs, read_cache

ROOT = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(str(tmp), str(path))


def save_torch(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    os.replace(str(tmp), str(path))


def setup(seed, device):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA unavailable; no silent CPU fallback')
        torch.cuda.manual_seed_all(seed)
        print('CUDA probe: allocating and synchronizing', flush=True)
        torch.empty(1, device='cuda')
        torch.cuda.synchronize()
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': np.__version__, 'device': device, 'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None}


def verify_manifest():
    manifest = load(ROOT / 'manifest.json')
    for name, digest in manifest.items():
        require(sha(ROOT / name) == digest, 'Bundle changed: ' + name)
    return sha(ROOT / 'manifest.json')


def validate_data(d, fused=True):
    n=len(d['y'])
    require(d['x'].shape==(n,2,124,524 if fused else 12) and d['x'].dtype==np.float32, 'Wrong side/time/feature layout')
    require(d['y'].shape==(n,2) and np.isfinite(d['x']).all() and np.isfinite(d['y']).all(), 'Invalid numbers')
    require(n==1061 and len(set(d['person_id']))==692 and len(set(d['video_id']))==n, 'Wrong retained cohort')
    for key in ('person_id','video_id','fold','cohort','original_row_index'):
        require(d[key].shape==(n,), 'Wrong metadata shape')
    require(len(np.unique(d['original_row_index']))==n, 'Duplicate source rows')
    require(set(d['fold'])==set(range(5)), 'Missing fold')
    for f in range(5):
        tr,_,te=split(d,f)
        require(len(set(d['person_id'][te]))==[138,138,138,139,139][f], 'Wrong fold membership')
        for j in range(5):
            if j!=f:split(d,f,j)


def split(d, outer, inner=None):
    require(outer in range(5) and (inner is None or inner in range(5) and inner != outer), 'Invalid fold')
    test = d['fold'] == outer
    valid = np.zeros(len(test), dtype=bool) if inner is None else d['fold'] == inner
    train = ~(test | valid)
    groups = [set(d['person_id'][m]) for m in (train, valid, test)]
    require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), 'Person leakage')
    return train, valid, test


def data(with_features=True):
    verify_manifest()
    with np.load(ROOT / 'data.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    validate_data(d,fused=False)
    validate_motion_inputs(ROOT,d)
    if with_features:
        features,cache_hash=read_cache(ROOT,ROOT/'cache/temporal',len(d['y']))
        tiled=np.broadcast_to(features[:,None,:,:],(len(features),2,124,512))
        d['x']=np.concatenate([d['x'],tiled],axis=3)
        d['_feature_cache_sha256']=cache_hash
        validate_data(d)
    return d


def config():
    c = load(ROOT / 'protocol.json')
    require(c['stage'] == 'frozen_motionbert_temporal_fusion_no_search', 'Baseline scope changed')
    for k in ('batch_size', 'max_epochs', 'lr_drop_every', 'pilot_epochs', 'wall_time_seconds'):
        require(type(c[k]) is int and c[k] > 0, 'Invalid ' + k)
    require(c['outer_folds'] == list(range(5)), 'Incomplete folds')
    require(type(c['seed']) is int and c['seed'] >= 0, 'Invalid seed')
    require(c['optimizer'] == 'RMSprop', 'Wrong optimizer')
    for k in ('learning_rate', 'lr_drop_factor', 'rmsprop_eps'):
        require(math.isfinite(c[k]) and c[k] > 0, 'Invalid ' + k)
    require(0 <= c['dropout'] < 1 and 0 <= c['rmsprop_alpha'] < 1
            and math.isfinite(c['kernel_l2']) and c['kernel_l2'] >= 0, 'Invalid regularization')
    require(c['target_scaling'] == 'training_only_pooled_sides_minmax'
            and c['selection'] == 'earliest_minimum_inner_person_weighted_calibrated_MAE'
            and c['input'] == 'matched124_geometry12_plus_motionbert512', 'Contract changed')
    return c


def person_mae(y, pred, people):
    require(y.shape == pred.shape and y.shape == (len(people), 2), 'Metric axes mismatch')
    require(np.isfinite(pred).all(), 'Nonfinite predictions')
    e = np.abs(pred.astype(np.float64) - y.astype(np.float64)).mean(axis=1)
    return float(np.mean([e[people == p].mean() for p in np.unique(people)]))


def scale_from_train(y):
    return torch.tensor(float(y.min()),dtype=torch.float32), torch.tensor(max(float(y.max()-y.min()),1e-6),dtype=torch.float32)


def make_optimizer(model, c):
    return torch.optim.RMSprop(model.parameters(), lr=c['learning_rate'],
                              alpha=c['rmsprop_alpha'], eps=c['rmsprop_eps'],
                              weight_decay=0, momentum=0, centered=False)


def step(model, opt, x, target, c):
    model.train()
    opt.zero_grad(set_to_none=True)
    mse = (model(x)[:,0] - target).square().mean()
    loss = mse + c['kernel_l2'] * model.penalty()
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Invalid gradients')
    opt.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite weights')
    return float(loss.detach()), float(mse.detach())


def predict(model,x,lo,span,device,batch=128):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            chunk=x[start:start+batch]
            p=model(chunk.reshape(-1,524,124).to(device))[:,0]*span.to(device)+lo.to(device)
            result.append(p.reshape(-1,2).cpu().numpy())
    return np.concatenate(result)


def calibrate(train_predictions,train_labels):
    # One shared slope/intercept, pooling left and right as in the source OLS code.
    design=np.column_stack([train_predictions.reshape(-1).astype(np.float64),np.ones(train_predictions.size)])
    coef=np.linalg.lstsq(design,train_labels.reshape(-1).astype(np.float64),rcond=None)[0]
    require(np.isfinite(coef).all(),'Invalid training-only calibration')
    return coef


def corrected(raw,coef):
    return raw.astype(np.float64)*coef[0]+coef[1]


def rng_state():
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(r):
    torch.set_rng_state(r['cpu'].cpu())
    if r['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in r['cuda']])


def fit(x, y, people, valid_x, valid_y, valid_people, c, epochs, folder, identity,
        device, deadline=None):
    """No outer-test arguments: optimizer, scaling and epoch curves cannot read them."""
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'identity.json').exists():
        require(load(folder / 'identity.json') == identity, 'Existing output belongs to a different fit')
    else:
        save(folder / 'identity.json', identity)
    setup(identity['fit_seed'], device)
    model = GdiCNN(c['dropout']).to(device)
    mean,scale=training_feature_stats(x)
    model.feature_mean.copy_(mean);model.feature_scale.copy_(scale)
    opt = make_optimizer(model, c)
    lo, span = scale_from_train(y)
    xt = channels(torch.from_numpy(x))
    yt = ((torch.from_numpy(y.astype(np.float32)) - lo) / span).reshape(-1)
    train_examples=xt.reshape(-1,524,124)
    vx = channels(torch.from_numpy(valid_x)) if valid_x is not None else None
    ckpath = folder / 'latest.pt'
    history = []
    if ckpath.exists():
        ck = torch.load(ckpath, map_location=device, weights_only=True)
        require(ck['identity'] == identity, 'Checkpoint belongs to a different run')
        require(torch.equal(ck['lo'].cpu(), lo) and torch.equal(ck['span'].cpu(), span), 'Training scaling changed')
        require(torch.equal(ck['model']['feature_mean'].cpu(),mean) and torch.equal(ck['model']['feature_scale'].cpu(),scale),'Feature normalization changed')
        model.load_state_dict(ck['model'], strict=True)
        opt.load_state_dict(ck['optimizer'])
        restore_rng(ck['rng'])
        history = ck['history']
        require(len(history) <= epochs, 'Requested fewer epochs than existing checkpoint')
    for epoch in range(len(history), epochs):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Wall-time budget reached; resume with the same command')
        started = time.perf_counter()
        lr = c['learning_rate'] * c['lr_drop_factor'] ** ((epoch + 1) // c['lr_drop_every'])
        for g in opt.param_groups:
            g['lr'] = lr
        order = np.random.default_rng(identity['fit_seed'] + epoch).permutation(len(yt))
        totals = np.zeros(2)
        for start in range(0, len(order), c['batch_size']):
            idx = order[start:start + c['batch_size']]
            totals += np.asarray(step(model, opt, train_examples[idx].to(device), yt[idx].to(device), c)) * len(idx)
        train_raw = predict(model, xt, lo, span, device)
        correction=calibrate(train_raw,y)
        train_pred=corrected(train_raw,correction)
        row = {'epoch': epoch + 1, 'lr': lr, 'loss': float(totals[0] / len(yt)),
               'scaled_mse': float(totals[1] / len(yt)),
               'train_person_mae': person_mae(y, train_pred, people),
               'train_raw_person_mae':person_mae(y,train_raw,people), 'calibration':correction.tolist()}
        if vx is not None:
            val_raw=predict(model,vx,lo,span,device)
            row['validation_person_mae']=person_mae(valid_y,corrected(val_raw,correction),valid_people)
            row['validation_raw_person_mae']=person_mae(valid_y,val_raw,valid_people)
            row['validation_people']=len(np.unique(valid_people))
        row['seconds'] = time.perf_counter() - started
        history.append(row)
        save_torch(ckpath, {'identity': identity, 'model': model.state_dict(),
                          'optimizer': opt.state_dict(), 'rng': rng_state(),
                          'history': history, 'lo': lo, 'span': span})
        save(folder / 'curve.json', history)
        print(json.dumps({'fit': folder.name, **row}), flush=True)
    # curve.json is derivable; restore it even if a previous interrupt followed checkpoint save.
    save(folder / 'curve.json', history)
    save(folder / 'fit_complete.json', {
        'identity': identity, 'epochs': len(history),
        'optimizer_steps': len(history) * math.ceil(len(yt) / c['batch_size']),
        'checkpoint_sha256': sha(ckpath), 'curve_sha256': sha(folder / 'curve.json')})
    return model, lo, span, history


def identity_for(d, c, outer, inner, epochs, purpose):
    tr, va, te = split(d, outer, inner)
    seed = c['seed'] + outer * 100 + (inner if inner is not None else 50)
    return {'bundle_sha256': sha(ROOT / 'manifest.json'), 'feature_cache_sha256':d['_feature_cache_sha256'], 'purpose': purpose,
            'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs, 'fit_seed': seed,
            'train_people': np.unique(d['person_id'][tr]).tolist(),
            'validation_people': np.unique(d['person_id'][va]).tolist(),
            'outer_evaluation_people': np.unique(d['person_id'][te]).tolist()}


def select_epoch(histories):
    require(len(histories) == 4 and len(set(map(len, histories))) == 1, 'Incomplete inner curves')
    curves = np.array([[r['validation_person_mae'] for r in h] for h in histories])
    require(np.isfinite(curves).all(), 'Nonfinite selection scores')
    # Subset fold sizes differ: weight by validation people, not by record count.
    weights=np.array([h[0]['validation_people'] for h in histories],dtype=float)
    means = np.average(curves,axis=0,weights=weights)
    return int(np.argmin(means)) + 1, means.tolist()


def train_fold(d, c, outer, out, device, deadline):
    dest = out / ('fold%d' % outer)
    histories = []
    for inner in range(5):
        if inner == outer:
            continue
        tr, va, _ = split(d, outer, inner)
        identity = identity_for(d, c, outer, inner, c['max_epochs'], 'inner_epoch_selection')
        _, _, _, hist = fit(d['x'][tr], d['y'][tr], d['person_id'][tr],
                            d['x'][va], d['y'][va], d['person_id'][va], c, c['max_epochs'],
                            dest / ('inner%d' % inner), identity, device, deadline)
        histories.append(hist)
    epoch, means = select_epoch(histories)
    save(dest / 'selection.json', {'selected_epoch': epoch, 'inner_mean_person_mae_by_epoch': means,
                                  'outer_evaluation_used_for_selection': False})
    tr, _, te = split(d, outer)
    identity = identity_for(d, c, outer, None, epoch, 'outer_refit')
    model, lo, span, _ = fit(d['x'][tr], d['y'][tr], d['person_id'][tr], None, None, None,
                            c, epoch, dest / 'refit', identity, device, deadline)
    train_raw=predict(model,channels(torch.from_numpy(d['x'][tr])),lo,span,device)
    coef=calibrate(train_raw,d['y'][tr])
    raw=predict(model,channels(torch.from_numpy(d['x'][te])),lo,span,device)
    pred=corrected(raw,coef)
    np.savez_compressed(dest / 'predictions.npz', predictions=pred[:, None, :], predictions_raw=raw[:,None,:], calibration=coef, y=d['y'][te],
                        row_index=np.flatnonzero(te), **{k: d[k][te] for k in ('person_id', 'video_id', 'fold', 'cohort')})
    save(dest / 'complete.json', {'identity': identity, 'evaluation_person_mae': person_mae(d['y'][te], pred, d['person_id'][te]),
                                'prediction_sha256': sha(dest / 'predictions.npz'),
                                'checkpoint_sha256': sha(dest / 'refit/latest.pt'),
                                'selection_sha256': sha(dest / 'selection.json'),
                                'calibration':coef.tolist(), 'raw_evaluation_person_mae':person_mae(d['y'][te],raw,d['person_id'][te]),
                                'selected_epoch': epoch, 'torch': str(torch.__version__), 'device': device})
    print(json.dumps({'complete': str(dest), 'selected_epoch': epoch}), flush=True)


def metrics(y, p, people):
    ids = np.unique(people)
    def macro(v):
        return float(np.mean([v[people == k].mean() for k in ids]))
    distance = np.abs(np.floor(p / 10) - np.floor(y / 10))
    return {'people': len(ids), 'records': len(y), 'mae': person_mae(y, p, people),
            'bias': macro(p - y), 'exact_10point_bin_rate': macro(distance == 0),
            'two_or_more_bins_rate': macro(distance >= 2),
            'prediction_min': float(p.min()), 'prediction_max': float(p.max())}


def summarize(d, c, out):
    pred = np.full((len(d['y']), 2), np.nan)
    seen = np.zeros(len(d['y']), dtype=int)
    raw_all=np.full_like(pred,np.nan)
    for f in range(5):
        dest = out / ('fold%d' % f)
        done = load(dest / 'complete.json')
        require(done['identity'] == identity_for(d, c, f, None, done['selected_epoch'], 'outer_refit'), 'Completion identity mismatch')
        require(sha(dest / 'predictions.npz') == done['prediction_sha256'], 'Prediction hash mismatch')
        require(sha(dest / 'selection.json') == done['selection_sha256'], 'Selection hash mismatch')
        histories = []
        for inner in range(5):
            if inner == f:
                continue
            inner_dir = dest / ('inner%d' % inner)
            completed = load(inner_dir / 'fit_complete.json')
            require(completed['identity'] == identity_for(d, c, f, inner, c['max_epochs'], 'inner_epoch_selection'), 'Inner fit identity mismatch')
            require(completed['epochs'] == c['max_epochs'] and sha(inner_dir / 'curve.json') == completed['curve_sha256'], 'Incomplete inner fit')
            histories.append(load(inner_dir / 'curve.json'))
        chosen, _ = select_epoch(histories)
        require(chosen == done['selected_epoch'], 'Wrong selected epoch')
        with np.load(dest / 'predictions.npz', allow_pickle=False) as z:
            idx = z['row_index']
            require(np.array_equal(idx, np.flatnonzero(d['fold'] == f)), 'Wrong evaluation rows')
            for k in ('person_id', 'video_id', 'fold', 'cohort', 'y'):
                require(np.array_equal(z[k], d[k][idx]), 'Prediction metadata mismatch: ' + k)
            require(z['predictions'].shape == (len(idx), 1, 2), 'Prediction axes mismatch')
            pred[idx] = z['predictions'][:, 0]
            raw_all[idx]=z['predictions_raw'][:,0]
            np.testing.assert_allclose(pred[idx],corrected(raw_all[idx],z['calibration']),rtol=0,atol=1e-10)
            seen[idx] += 1
    require((seen == 1).all() and np.isfinite(pred).all(), 'Incomplete OOF predictions')
    np.savez_compressed(out / 'oof_predictions.npz', predictions=pred[:, None, :], predictions_raw=raw_all[:,None,:], original_row_index=d['original_row_index'], y=d['y'],
                        **{k: d[k] for k in ('person_id', 'video_id', 'fold', 'cohort')})
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'seed', 'fold', 'role', 'person_id', 'video_id', 'side', 'actual_gdi', 'predicted_gdi'])
        for i in range(len(pred)):
            for j, side in enumerate(('left', 'right')):
                writer.writerow([c['name'], c['seed'], int(d['fold'][i]), 'outer_evaluation',
                                 d['person_id'][i], d['video_id'][i], side, d['y'][i, j], pred[i, j]])
    rows = []
    for name, mask in [('retained_all', np.ones(len(d['y']), dtype=bool)), ('retained_old174', d['cohort'] == 0), ('retained_new518', d['cohort'] == 1)]:
        rows.append({'group': name, **metrics(d['y'][mask], pred[mask], d['person_id'][mask])})
    reference = load(ROOT / 'reference_metrics.json')
    summary = {'method': c['name'], 'metrics': rows, 'reference_metrics': reference,
               'raw_diagnostic_metrics':metrics(d['y'],raw_all,d['person_id']),
               'primary_output':'training_only_OLS_corrected; raw is diagnostic',
               'single_cnn_mae_reduction':reference['single_cnn_mae']-rows[0]['mae'],
               'frozen_motionbert':True, 'feature_cache_sha256':d['_feature_cache_sha256'],
               'ridge_mae_reduction': reference['ridge_mae'] - rows[0]['mae'],
               'pls_mae_reduction': reference['pls_mae'] - rows[0]['mae'],
               'bundle_sha256': sha(ROOT / 'manifest.json'), 'independent_final_evaluation': False,
               'oof_sha256': sha(out / 'oof_predictions.npz')}
    summary['optimizer_steps'] = sum(load(p)['optimizer_steps'] for p in out.glob('fold*/*/fit_complete.json'))
    summary['fits'] = len(list(out.glob('fold*/*/fit_complete.json')))
    summary['fold_metrics'] = [dict(fold=f, **metrics(d['y'][d['fold'] == f], pred[d['fold'] == f], d['person_id'][d['fold'] == f])) for f in range(5)]
    save(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def smoke(c, out, device):
    """Only synthetic data; includes nonzero updates and exact checkpoint continuation."""
    out.mkdir(parents=True, exist_ok=True)
    env = setup(1234, device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    x = channels(torch.rand(c['batch_size']//2,2,124,524)).reshape(-1,524,124).to(device)
    y = torch.rand(c['batch_size']).to(device)
    m = GdiCNN(c['dropout']).to(device)
    opt = make_optimizer(m, c)
    initial = {n: p.detach().clone() for n, p in m.named_parameters()}
    start = time.perf_counter()
    loss1 = step(m, opt, x, y, c)
    ckpath = out / 'synthetic.pt'
    save_torch(ckpath, {'model': m.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng_state()})
    loss2 = step(m, opt, x, y, c)
    expected = {k: v.detach().clone() for k, v in m.state_dict().items()}
    restored = GdiCNN(c['dropout']).to(device)
    opt2 = make_optimizer(restored, c)
    ck = torch.load(ckpath, map_location=device, weights_only=True)
    restored.load_state_dict(ck['model'])
    opt2.load_state_dict(ck['optimizer'])
    restore_rng(ck['rng'])
    replay_loss = step(restored, opt2, x, y, c)
    max_error = max(float((expected[k] - v).abs().max()) for k, v in restored.state_dict().items())
    require(max_error == 0 and loss2 == replay_loss, 'Checkpoint continuation mismatch')
    changed = sum(not torch.equal(initial[n], p) for n, p in m.named_parameters())
    require(changed > 0, 'No parameter updates')
    m.eval()
    restored.eval()
    with torch.no_grad():
        require(torch.equal(m(x), restored(x)), 'Restored prediction mismatch')
    result = {'passed': True, 'environment': env, 'synthetic_optimizer_steps': 3,
              'real_data_optimizer_steps': 0, 'parameters': sum(p.numel() for p in m.parameters()),
              'changed_parameter_tensors': changed, 'resume_max_error': max_error,
              'loss1': loss1, 'loss2': loss2, 'seconds': time.perf_counter() - start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None}
    save(out / 'smoke.json', result)
    print(json.dumps(result), flush=True)


def export_review(out):
    require((out / 'summary.json').exists(), 'Summarize all five folds first')
    archive = out.parent / (out.name + '-review.tar.gz')
    with tarfile.open(archive, 'w:gz') as tf:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.npz', '.csv'):
                tf.add(str(p), arcname='results/' + p.relative_to(out).as_posix())
        for p in (ROOT / 'manifest.json', ROOT / 'protocol.json', ROOT / 'run.py', ROOT / 'model.py', ROOT / 'features.py', ROOT / 'input_contract.json', ROOT / 'cache/temporal/complete.json'):
            tf.add(str(p), arcname='bundle/' + p.name)
    print(json.dumps({'review_archive': str(archive), 'sha256': sha(archive)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'smoke', 'pilot', 'train', 'summarize', 'export'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/main')
    parser.add_argument('--fold', type=int, choices=range(5))
    args = parser.parse_args()
    print('CNN command: ' + args.command, flush=True)
    verify_manifest()
    c = config()
    if args.command == 'smoke':
        return smoke(c, args.output, args.device)
    if args.command == 'export':
        return export_review(args.output)
    d = data(with_features=args.command != 'check')
    if args.command == 'check':
        result = {'passed': True, 'people': len(set(d['person_id'])), 'records': len(d['y']), 'input_shape': list(d['x'].shape),
                  'cnn_input_shape_after_feature_extraction': [len(d['y'])*2, 524, 124], 'temporal_cache_required_for_training':True, 'targets': ['left_GDI', 'right_GDI'],
                  'person_disjoint': True, 'all_20_inner_splits_checked': True,
                  'real_data_optimizer_steps': 0, 'bundle_sha256': sha(ROOT / 'manifest.json')}
        save(args.output / 'input_check.json', result)
        print(json.dumps(result), flush=True)
        return
    if args.command == 'summarize':
        return summarize(d, c, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    env = setup(c['seed'], args.device)
    save(args.output / ('environment-%d.json' % time.time_ns()), env)
    deadline = time.monotonic() + c['wall_time_seconds']
    if args.command == 'pilot':
        tr, va, _ = split(d, 0, 1)
        ident = identity_for(d, c, 0, 1, c['pilot_epochs'], 'pilot_not_performance')
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        _, _, _, hist = fit(d['x'][tr], d['y'][tr], d['person_id'][tr], d['x'][va], d['y'][va],
                            d['person_id'][va], c, c['pilot_epochs'], args.output / 'pilot', ident, args.device, deadline)
        mean_seconds = float(np.mean([r['seconds'] for r in hist]))
        result = {'pilot_only': True, 'epochs': len(hist), 'train_people': len(set(d['person_id'][tr])), 'validation_people':len(set(d['person_id'][va])),
                  'outer_evaluation_used': False, 'mean_epoch_seconds': mean_seconds,
                  'rough_max_training_seconds': mean_seconds*c['max_epochs']*sum(math.ceil(2*int(split(d,f,j)[0].sum())/c['batch_size']) for f in range(5) for j in ([None]+[k for k in range(5) if k!=f]))/math.ceil(2*int(tr.sum())/c['batch_size']),
                  'estimate_note': 'rough extrapolation; selected refit epochs, IO and hardware vary',
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None}
        save(args.output / 'pilot_summary.json', result)
        print(json.dumps(result), flush=True)
        return
    for outer in ([args.fold] if args.fold is not None else range(5)):
        train_fold(d, c, outer, args.output, args.device, deadline)
    if args.fold is None:
        summarize(d, c, args.output)
    print('CNN_TRAINING_FINISHED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(124)
