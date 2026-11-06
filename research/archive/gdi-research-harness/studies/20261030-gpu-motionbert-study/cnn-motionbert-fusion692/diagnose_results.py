"""Post-hoc saved-model audit; no fitting, model selection, or GPU allocation."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
B = HERE / 'bundle'
R = HERE / 'received/results'
OUT = HERE / 'analysis'
BASE = HERE.parent / 'cnn-paper-input695'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('fusion_model_audit', B / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(B / 'data.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    with np.load(R / 'oof_predictions.npz', allow_pickle=False) as z:
        fusion, raw = z['predictions'][:, 0].copy(), z['predictions_raw'][:, 0].copy()
    with np.load(BASE / 'analysis/cnn_oof.npz', allow_pickle=False) as z:
        single = z['predictions'][:, 0].copy()
    proof = load(HERE / 'received/bundle/complete.json')
    feature_path = HERE / 'received/cache/features.npy'
    assert sha(feature_path) == proof['features_sha256']
    feature = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    assert feature.shape == (1061, 124, 512) and np.isfinite(feature).all()
    with np.load(B / 'motion_input.npz', allow_pickle=False) as z:
        for k in ('person_id', 'video_id', 'fold', 'original_row_index'):
            np.testing.assert_array_equal(z[k], d[k])
        pooled_error = float(np.max(np.abs(feature.mean(axis=1) - z['pooled_reference'])))
        assert pooled_error <= 1e-4
    folds = []
    reproduced = np.empty_like(raw)
    for f in range(5):
        idx = np.flatnonzero(d['fold'] == f)
        tr = d['fold'] != f
        done = load(R / ('fold%d' % f) / 'complete.json')
        checkpoint_path = HERE / 'received/checkpoints' / ('fold%d.pt' % f)
        assert sha(checkpoint_path) == done['checkpoint_sha256']
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert ck['identity'] == done['identity']
        assert ck['history'] == load(R / ('fold%d' % f) / 'refit/curve.json')
        model = module.GdiCNN()
        model.load_state_dict(ck['model'], strict=True)
        model.requires_grad_(False)
        model.eval()
        assert float(ck['lo']) == float(np.float32(d['y'][tr].min()))
        assert float(ck['span']) == float(np.float32(d['y'][tr].max()-d['y'][tr].min()))
        vals = np.asarray(feature[tr], dtype=np.float64)
        mean = vals.mean(axis=(0, 1)).astype(np.float32)
        scale = vals.std(axis=(0, 1)).astype(np.float32)
        scale[scale < 1e-6] = 1
        del vals
        np.testing.assert_array_equal(model.feature_mean.detach().numpy(), mean)
        np.testing.assert_array_equal(model.feature_scale.detach().numpy(), scale)
        with torch.inference_mode():
            for start in range(0, len(idx), 32):
                rows = idx[start:start+32]
                motion = np.broadcast_to(feature[rows, None], (len(rows), 2, 124, 512))
                x = torch.from_numpy(np.concatenate([d['x'][rows], motion], axis=3)).permute(0, 1, 3, 2).contiguous().reshape(-1, 524, 124)
                p = model(x)[:, 0] * ck['span'] + ck['lo']
                reproduced[rows] = p.reshape(-1, 2).numpy()
            # Diagnose the first-layer signal on fixed training rows, without changing weights.
            rows = np.flatnonzero(tr)[:32]
            geom = torch.from_numpy(d['x'][rows]).permute(0, 1, 3, 2).contiguous().reshape(-1, 12, 124)
            mv = np.repeat(feature[rows], 2, axis=0)
            motion = (torch.from_numpy(mv).transpose(1, 2)-model.feature_mean[None, :, None])/model.feature_scale[None, :, None]
            w = model.convs[0].weight
            g = F.conv1d(F.pad(geom, (3, 4)), w[:, :12])
            m = F.conv1d(F.pad(motion, (3, 4)), w[:, 12:])
            grms, mrms = float(g.square().mean().sqrt()), float(m.square().mean().sqrt())
            gcenter = float((g-g.mean(dim=(0,2),keepdim=True)).square().mean().sqrt())
            mcenter = float((m-m.mean(dim=(0,2),keepdim=True)).square().mean().sqrt())
        # CPU/CUDA arithmetic may differ; use the existing bundle replay tolerance.
        error = float(np.max(np.abs(reproduced[idx]-raw[idx])))
        assert error <= 1e-4, (f, error)
        curve = ck['history']
        folds.append(dict(fold=f, selected_epoch=len(curve), cpu_raw_max_error=error,
            first_conv_geometry_rms=grms, first_conv_motion_rms=mrms,
            motion_over_geometry_rms=mrms/max(grms, 1e-12),
            motion_over_geometry_centered_rms=mcenter/max(gcenter, 1e-12),
            geometry_weight_norm=float(w[:, :12].norm()), motion_weight_norm=float(w[:, 12:].norm()),
            geometry_input_rms=float(geom.square().mean().sqrt()), normalized_motion_input_rms=float(motion.square().mean().sqrt())))
        print('MODEL_REPLAY_OK', f, error, flush=True)
    calibrated = np.empty_like(reproduced)
    for f in range(5):
        mask = d['fold'] == f
        a, b = load(R / ('fold%d' % f) / 'complete.json')['calibration']
        calibrated[mask] = reproduced[mask]*a+b
    replay_error = float(np.max(np.abs(calibrated-fusion)))
    assert replay_error < 2e-4
    ids, y = d['person_id'], d['y']
    def macro(values, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        return float(np.mean([values[(ids==p)&mask].mean() for p in np.unique(ids[mask])]))
    def distribution(p):
        return dict(min=float(p.min()), max=float(p.max()), std_record_sides=float(p.std()),
            correlation_record_sides=float(np.corrcoef(p.ravel(), y.ravel())[0, 1]),
            mean_abs_left_right_gap=macro(np.abs(p[:, 0]-p[:, 1])),
            left_right_gap_mae=macro(np.abs((p[:, 0]-p[:, 1])-(y[:, 0]-y[:, 1]))),
            bilateral_mean_mae=macro(np.abs(p.mean(axis=1)-y.mean(axis=1))),
            posthoc_same_bilateral_prediction_mae=macro(np.abs(p.mean(axis=1,keepdims=True)-y)))
    bands = []
    person_truth = {p: float(y[ids==p].mean()) for p in np.unique(ids)}
    means = np.array([person_truth[p] for p in ids])
    for low, high in [(0, 60), (60, 70), (70, 80), (80, 90), (90, 150)]:
        mask = (means >= low) & (means < high)
        if not mask.any(): continue
        bands.append(dict(person_mean_gdi_band=[low, high], people=len(np.unique(ids[mask])),
            single_mae=macro(np.abs(single-y),mask), fusion_mae=macro(np.abs(fusion-y),mask),
            single_bias=macro(single-y,mask),fusion_bias=macro(fusion-y,mask)))
    curve_rows = []
    for label, root in [('single', BASE/'received/results'), ('fusion', R)]:
        all_hist=[]
        for f in range(5):
            all_hist.extend(load(p) for p in (root/('fold%d'%f)).glob('inner*/curve.json'))
        assert len(all_hist)==20
        for ep in [1, 10, 20, 40, 60, 80, 100]:
            rows=[h[ep-1] for h in all_hist]
            curve_rows.append(dict(model=label, epoch=ep,
                mean_train_mae=float(np.mean([r['train_person_mae'] for r in rows])),
                mean_valid_mae=float(np.mean([r['validation_person_mae'] for r in rows])),
                mean_scaled_mse=float(np.mean([r['scaled_mse'] for r in rows])),
                negative_calibration_slopes=sum(r['calibration'][0]<0 for r in rows)))
    single_diag=load(BASE/'analysis/summary.json')['training_diagnostics']
    fusion_diag=load(OUT/'summary.json')['training_diagnostics']
    c1=load(BASE/'bundle/protocol.json'); c2=load(B/'protocol.json')
    differences={k:[c1.get(k),c2.get(k)] for k in sorted(set(c1)|set(c2)) if c1.get(k)!=c2.get(k)}
    result=dict(checkpoint_reinference_passed=True, cpu_torch=str(torch.__version__),
        checkpoint_count=5, records_reinferred=len(y), raw_max_error=float(np.max(np.abs(reproduced-raw))),
        calibrated_max_error=replay_error, feature_mean_matches_prior_pooled_max_error=pooled_error,
        feature_normalization_train_only_recomputed=True, no_optimizer_steps=True,
        model_parameters=sum(p.numel() for p in model.parameters()), folds=folds,
        distributions={'truth':distribution(y),'single':distribution(single),'fusion':distribution(fusion)},
        person_mean_score_bands=bands, inner_curve_means=curve_rows,
        single_refit=single_diag,fusion_refit=fusion_diag,protocol_differences=differences,
        interpretation_limit='Post-hoc diagnostics; architecture causes are not isolated by a controlled retraining ablation. No new model selected on outer outcomes.')
    save(OUT/'diagnostics.json',result)
    # Static diagnostic figure; these are saved outcomes, not new model selection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for label, root, color in [('Single CNN', BASE/'received/results','#1976a3'),('Fusion CNN', R,'#d55e00')]:
        histories=[load(p) for p in root.glob('fold*/inner*/curve.json')]
        for key,style in [('train_person_mae','--'),('validation_person_mae','-')]:
            curve=np.mean([[r[key] for r in h] for h in histories],axis=0)
            axes[0].plot(np.arange(1,101),curve,style,color=color,label=label+(' train' if style=='--' else ' validation'))
    axes[0].set(xlabel='Epoch',ylabel='Mean person MAE (GDI points)',title='20 inner fits: mean learning curves',ylim=(4,12))
    axes[0].legend(fontsize=8)
    for values,label,color in [(single,'Single CNN','#1976a3'),(fusion,'Fusion CNN','#d55e00')]:
        tx=np.array([y[ids==p].mean() for p in np.unique(ids)])
        py=np.array([values[ids==p].mean() for p in np.unique(ids)])
        axes[1].scatter(tx,py,s=9,alpha=.35,color=color,label=label)
    axes[1].plot([35,115],[35,115],':',color='gray')
    axes[1].set(xlabel='Actual GDI (person mean)',ylabel='Predicted GDI (person mean)',title='Held-out predictions: 692 people',xlim=(35,115),ylim=(35,115))
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'diagnostics.png',dpi=170)
    plt.close(fig)
    proof=load(OUT/'verification.json')
    proof.update(independent_model_reinference=True,checkpoint_hashes_match=True,
        cache_content_hash_matches=True,train_feature_normalization_recomputed=True,
        model_reinference_calibrated_max_error=replay_error)
    save(OUT/'verification.json',proof)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
