"""Matched baseline versus knee-auxiliary CNN, nested participant-level selection."""
import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import baseline as b
from model import GdiCNN, channels
from windows import weighted_mse

ROOT = Path(__file__).resolve().parent


def verify_manifest():
    for name, digest in b.load(ROOT / 'manifest.json').items():
        b.require(b.sha(ROOT / name) == digest, 'Changed sealed input: ' + name)


def config():
    c = b.load(ROOT / 'protocol.json')
    b.require(c['auxiliary_target'] == 'KneeFlex_maxExtension', 'Wrong target')
    b.require(c['auxiliary_weights'] == [0.1, 0.3, 1.0], 'Changed weight search')
    b.require(c['expected_fits'] == 90, 'Changed training budget')
    return c


def data():
    verify_manifest()
    with np.load(ROOT / 'data.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    d['x'] = b.WindowSet(d['x'], d.pop('window_record'), len(d['y']),
                       d.pop('window_start'), d.pop('canonical_fallback'))
    b.validate_data(d)
    with np.load(ROOT / 'auxiliary-targets.npz', allow_pickle=False) as z:
        for key in ['person_id', 'video_id', 'fold']:
            b.require(np.array_equal(z[key], d[key]), 'Auxiliary linkage changed: ' + key)
        d['aux'] = z['aux'].copy()
    b.require(d['aux'].shape == d['y'].shape and np.isfinite(d['aux']).all(), 'Invalid knee targets')
    return d


def step(model, opt, x, y, aux, weight, c, record_weights):
    model.train()
    opt.zero_grad(set_to_none=True)
    output = model(x)
    mse = weighted_mse(output[:, 0], y, record_weights)
    # Skip entirely for weight zero: baseline is the exact original computation.
    auxiliary = weighted_mse(output[:, 1], aux, record_weights) if weight else mse.new_zeros(())
    loss = mse + c['kernel_l2'] * model.penalty()
    if weight:
        loss = loss + weight * auxiliary
    b.require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    b.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Invalid gradients')
    opt.step()
    b.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite weights')
    return float(loss.detach()), float(mse.detach()), float(auxiliary.detach())


def fit(x, y, aux, people, vx, vy, vp, c, epochs, folder, identity, device):
    """Only train and inner validation data enter fitting; no outer outcomes."""
    folder.mkdir(parents=True, exist_ok=True)
    b.setup(identity['fit_seed'], device)
    model = GdiCNN(c['dropout']).to(device)
    optimizer = b.make_optimizer(model, c)
    lo, span = b.scale_from_train(y)
    aux_lo, aux_span = b.scale_from_train(aux)
    yt = ((torch.from_numpy(y[x.owner].astype(np.float32)) - lo) / span).reshape(-1)
    at = ((torch.from_numpy(aux[x.owner].astype(np.float32)) - aux_lo) / aux_span).reshape(-1)
    inputs = channels(torch.from_numpy(x.values)).reshape(-1, 12, 124)
    weights = torch.from_numpy(x.side_weights())
    weight = identity['auxiliary_weight']
    checkpoint = folder / 'latest.pt'
    history = []
    if (folder / 'identity.json').exists():
        b.require(b.load(folder / 'identity.json') == identity, 'Existing fit identity mismatch')
    else:
        b.save(folder / 'identity.json', identity)
    if checkpoint.exists():
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        b.require(ck['identity'] == identity, 'Resume identity mismatch')
        for key, value in [('lo', lo), ('span', span), ('aux_lo', aux_lo), ('aux_span', aux_span)]:
            b.require(torch.equal(ck[key].cpu(), value), 'Resume target scaling mismatch')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        b.restore_rng(ck['rng'])
        history = ck['history']
        b.require(len(history) <= epochs, 'Cannot shorten existing training')
    for epoch in range(len(history), epochs):
        started = time.perf_counter()
        lr = c['learning_rate'] * c['lr_drop_factor'] ** ((epoch + 1) // c['lr_drop_every'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        order = np.random.default_rng(identity['fit_seed'] + epoch).permutation(len(yt))
        totals = np.zeros(3)
        for start in range(0, len(order), c['batch_size']):
            idx = order[start:start + c['batch_size']]
            values = step(model, optimizer, inputs[idx].to(device), yt[idx].to(device), at[idx].to(device),
                          weight, c, weights[idx].to(device))
            totals += np.asarray(values) * len(idx)
        train_raw = b.predict(model, x, lo, span, device)
        coef = b.calibrate(train_raw, y)
        row = dict(epoch=epoch + 1, lr=lr, loss=float(totals[0] / len(yt)),
                   scaled_mse=float(totals[1] / len(yt)), scaled_auxiliary_mse=float(totals[2] / len(yt)),
                   train_person_mae=b.person_mae(y, b.corrected(train_raw, coef), people),
                   train_raw_person_mae=b.person_mae(y, train_raw, people), calibration=coef.tolist())
        arrays = dict(train_raw=train_raw, calibration=coef)
        if vx is not None:
            val_raw = b.predict(model, vx, lo, span, device)
            row.update(validation_person_mae=b.person_mae(vy, b.corrected(val_raw, coef), vp),
                       validation_raw_person_mae=b.person_mae(vy, val_raw, vp),
                       validation_people=len(np.unique(vp)))
            arrays['validation_raw'] = val_raw
        row['seconds'] = time.perf_counter() - started
        history.append(row)
        # Preserve the predictions that determined every epoch's calibration/selection.
        np.savez_compressed(folder / ('epoch-%03d.npz' % (epoch + 1)), **arrays)
        b.save_torch(checkpoint, dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                     rng=b.rng_state(), history=history, lo=lo, span=span,
                                     aux_lo=aux_lo, aux_span=aux_span))
        b.save(folder / 'curve.json', history)
        print(b.json.dumps(dict(fit=str(folder), **row)), flush=True)
    b.save(folder / 'fit_complete.json', dict(identity=identity, epochs=len(history),
           optimizer_steps=len(history) * math.ceil(len(yt) / c['batch_size']),
           checkpoint_sha256=b.sha(checkpoint), curve_sha256=b.sha(folder / 'curve.json')))
    return model, lo, span, history


def identity(d, c, outer, inner, epochs, weight, purpose):
    tr, va, te = b.split(d, outer, inner)
    return dict(purpose=purpose, fit_seed=c['seed'] + outer * 100 + (50 if inner is None else inner),
                outer_fold=outer, inner_fold=inner, epochs=epochs, auxiliary_weight=weight,
                manifest_sha256=b.sha(ROOT / 'manifest.json'),
                train_people=np.unique(d['person_id'][tr]).tolist(),
                validation_people=np.unique(d['person_id'][va]).tolist(),
                outer_evaluation_people=np.unique(d['person_id'][te]).tolist())


def fit_subset(d, c, outer, inner, epochs, weight, folder, device, purpose):
    tr, va, _ = b.split(d, outer, inner)
    ident = identity(d, c, outer, inner, epochs, weight, purpose)
    return fit(d['x'][tr], d['y'][tr], d['aux'][tr], d['person_id'][tr],
               d['x'][va] if va.any() else None, d['y'][va] if va.any() else None,
               d['person_id'][va] if va.any() else None, c, epochs, folder, ident, device)


def tag(weight):
    return 'weight-' + str(weight).replace('.', 'p')


def select(candidates):
    """Minimum inner-person MAE; ties choose smaller weight then earlier epoch."""
    best = min(candidates, key=lambda row: (row['score'], row['weight'], row['epoch']))
    return best


def train_fold(d, c, variant, outer, out, device):
    dest = out / variant / ('fold%d' % outer)
    weights = [0.0] if variant == 'gdi_only' else c['auxiliary_weights']
    candidates = []
    for weight in weights:
        histories = []
        for inner in range(5):
            if inner == outer:
                continue
            _, _, _, hist = fit_subset(d, c, outer, inner, c['max_epochs'], weight,
                                       dest / tag(weight) / ('inner%d' % inner), device, 'inner_selection')
            histories.append(hist)
        epoch, curve = b.select_epoch(histories)
        candidates.append(dict(weight=weight, epoch=epoch, score=curve[epoch - 1], inner_curve=curve))
    chosen = select(candidates)
    b.save(dest / 'selection.json', dict(chosen=chosen, candidates=candidates, outer_outcomes_used=False))
    model, lo, span, _ = fit_subset(d, c, outer, None, chosen['epoch'], chosen['weight'],
                                   dest / 'refit', device, 'outer_refit')
    tr, _, te = b.split(d, outer)
    coef = b.calibrate(b.predict(model, d['x'][tr], lo, span, device), d['y'][tr])
    raw = b.predict(model, d['x'][te], lo, span, device)
    pred = b.corrected(raw, coef)
    np.savez_compressed(dest / 'predictions.npz', predictions=pred, predictions_raw=raw, calibration=coef,
                        row_index=np.flatnonzero(te), y=d['y'][te],
                        **{key: d[key][te] for key in ['person_id', 'video_id', 'fold']})
    b.save(dest / 'complete.json', dict(selected_epoch=chosen['epoch'], selected_weight=chosen['weight'],
            evaluation_person_mae=b.person_mae(d['y'][te], pred, d['person_id'][te]),
            prediction_sha256=b.sha(dest / 'predictions.npz'), checkpoint_sha256=b.sha(dest / 'refit/latest.pt'),
            selection_sha256=b.sha(dest / 'selection.json')))


def pilot(d, c, out, device):
    summaries = {}
    for weight in [0.0, 1.0]:
        folder = out / tag(weight)
        _, _, _, hist = fit_subset(d, c, 0, 1, 3, weight, folder, device, 'pilot_not_performance')
        summaries[str(weight)] = dict(epochs=len(hist), mean_epoch_seconds=float(np.mean([r['seconds'] for r in hist])),
                                     checkpoint_sha256=b.sha(folder / 'latest.pt'))
    b.save(out / 'pilot_summary.json', dict(pilot_only=True, train_people=416, validation_people=138,
           outer_evaluation_used=False, variants=summaries,
           peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else None))


def collect(d, out):
    predictions = {}
    for variant in ['gdi_only', 'knee_auxiliary']:
        full = np.full_like(d['y'], np.nan, dtype=np.float64)
        for outer in range(5):
            folder = out / variant / ('fold%d' % outer)
            done = b.load(folder / 'complete.json')
            b.require(b.sha(folder / 'predictions.npz') == done['prediction_sha256'], 'Prediction changed')
            with np.load(folder / 'predictions.npz') as z:
                rows = z['row_index']
                b.require(np.array_equal(rows, np.flatnonzero(d['fold'] == outer)), 'Wrong outer rows')
                for key in ['person_id', 'video_id', 'fold', 'y']:
                    b.require(np.array_equal(z[key], d[key][rows]), 'Outer metadata changed: ' + key)
                full[rows] = z['predictions']
        b.require(np.isfinite(full).all(), 'Incomplete predictions')
        predictions[variant] = full
    return predictions


def summarize(d, c, out):
    predictions = collect(d, out)
    people = np.unique(d['person_id'])
    errors = {k: np.array([np.abs(p[d['person_id'] == person] - d['y'][d['person_id'] == person]).mean()
                          for person in people]) for k, p in predictions.items()}
    delta = errors['knee_auxiliary'] - errors['gdi_only']
    draws = np.random.default_rng(c['uncertainty']['seed']).integers(0, len(people), (10000, len(people)))
    interval = np.percentile(delta[draws].mean(1), [2.5, 97.5])
    truth = np.array([d['y'][d['person_id'] == person].mean() for person in people])
    bands = []
    for low, high, name in [(-np.inf, 60, 'below60'), (60, 70, '60to70'), (70, 80, '70to80'), (80, 90, '80to90'), (90, np.inf, '90plus')]:
        mask = (truth >= low) & (truth < high)
        bands.append(dict(band=name, people=int(mask.sum()), mae={k: float(v[mask].mean()) if mask.any() else None for k, v in errors.items()}))
    folds = {k: [b.person_mae(d['y'][d['fold'] == f], p[d['fold'] == f], d['person_id'][d['fold'] == f]) for f in range(5)] for k, p in predictions.items()}
    report = dict(people=len(people), records=len(d['y']), expected_fits=90,
                  completed_fits=len(list(out.glob('*/*/*/inner*/fit_complete.json'))) + len(list(out.glob('*/*/refit/fit_complete.json'))),
                  metrics={k: dict(person_equal_mae=float(e.mean()), fold_mae=folds[k], fold_mae_sample_sd=float(np.std(folds[k], ddof=1)),
                                   pearson=float(np.corrcoef(d['y'].ravel(), predictions[k].ravel())[0, 1]),
                                   person_equal_bias=float(np.mean([(predictions[k][d['person_id'] == person] - d['y'][d['person_id'] == person]).mean() for person in people]))) for k, e in errors.items()},
                  comparison=dict(auxiliary_minus_gdi_only=float(delta.mean()), paired_person_bootstrap95=interval.tolist(),
                                  relative_mae_reduction_percent=float(-100 * delta.mean() / errors['gdi_only'].mean()),
                                  verdict='auxiliary_better' if interval[1] < 0 else 'gdi_only_better' if interval[0] > 0 else 'inconclusive',
                                  people_improved=int((delta < 0).sum()), people_worsened=int((delta > 0).sum())),
                  bands=bands, selected={variant: [b.load(out / variant / ('fold%d' % f) / 'selection.json')['chosen'] for f in range(5)] for variant in predictions},
                  independent_final_evaluation=False, reserved_holdout_used=False,
                  caveats=['Repeatedly explored development cohort, not a new independent confirmatory test.',
                           'Bootstrap holds trained predictions fixed; training/selection uncertainty is not fully included.',
                           'Five folds share training people; their sample SD is descriptive, not independent repetitions.'])
    b.require(report['completed_fits'] == 90, 'Missing fits')
    np.savez_compressed(out / 'oof_predictions.npz', **predictions, y=d['y'], person_id=d['person_id'], video_id=d['video_id'])
    b.save(out / 'summary.json', report)
    print(b.json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['check', 'pilot', 'train', 'summarize'])
    parser.add_argument('--variant', choices=['gdi_only', 'knee_auxiliary'])
    parser.add_argument('--fold', type=int, choices=range(5))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/main')
    args = parser.parse_args()
    d, c = data(), config()
    if args.command == 'check':
        b.save(args.output / 'input_check.json', dict(passed=True, people=692, records=1061,
               all_20_person_disjoint_splits_checked=True, optimizer_steps=0, reserved_holdout_used=False))
        return
    if args.command == 'summarize':
        return summarize(d, c, args.output)
    b.setup(c['seed'], args.device)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.command == 'pilot':
        return pilot(d, c, args.output, args.device)
    b.require(args.variant is not None and args.fold is not None, 'Specify variant and fold')
    train_fold(d, c, args.variant, args.fold, args.output, args.device)


if __name__ == '__main__':
    main()
