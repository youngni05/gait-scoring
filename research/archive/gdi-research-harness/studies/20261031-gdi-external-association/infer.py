"""Apply only existing GDI checkpoints; no target-score fitting anywhere."""
from common import *
import torch
import joblib


def linear(x, path, pls=False):
    a = arrays(path)
    coef, intercept = ('coef', 'intercept') if pls else ('combined_coef', 'combined_intercept')
    return ((x - a['scaler_mean']) / a['scaler_scale']) @ a[coef].T + a[intercept]


def encoder(x):
    sys.path.insert(0, str(MB / 'vendor'))
    from lib.model.DSTformer import DSTformer
    model = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8,
                      mlp_ratio=4, num_joints=17, maxlen=243, qkv_bias=True, att_fuse=True)
    path = source(MB / 'assets/MB_lite.bin', '6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461')
    state = torch.load(path, map_location='cpu', weights_only=True)['model_pos']
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
    model.requires_grad_(False).eval()
    frames, pooled = [], []
    with torch.inference_mode():
        for i, row in enumerate(x):
            rep = model.get_representation(torch.from_numpy(row[None]))
            frames.append(rep.mean(2).numpy()[0])
            pooled.append(rep.mean((1, 2)).numpy()[0])
            if i % 10 == 0:
                print(json.dumps({'encoder_records': i + 1, 'total': len(x)}), flush=True)
    return np.stack(pooled), np.stack(frames)


def cnn_models():
    raw = module(FUSION / 'cnn_source/model.py', 'association_raw_model')
    # Reuse the exact saved temporal architecture without importing training code.
    path = source(FUSION / 'train_cnn.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'TemporalCNN')
    ns = {'GdiCNN': raw.GdiCNN, 'SameConv': raw.SameConv, 'torch': torch,
          'F': torch.nn.functional}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), 'exec'), ns)
    return raw, ns['TemporalCNN']


def cnn_predict(x, path, cls, channels, expected):
    ck = torch.load(source(path, expected), map_location='cpu', weights_only=True)
    m = cls(.5)
    m.load_state_dict(ck['model'], strict=True)
    m.requires_grad_(False).eval()
    tx = channels(torch.from_numpy(np.asarray(x)))
    with torch.inference_mode():
        return (m(tx) * ck['span'] + ck['lo']).numpy()


def h36m_predictions(x, pooled, frames, replay=False):
    raw, temporal_cls = cnn_models()
    summary = functions(FUSION / 'prepare.py', ['summaries'])['summaries']
    leg_fn = functions(CPU / 'side-gap-learning695/run.py', ['raw_features'], flexible_batch=True)['raw_features']
    time_fn = functions(CPU / 'mean-leg-followup695/run.py', ['temporal_features'], flexible_batch=True)['temporal_features']
    kin = summary(x)[0]
    leg = leg_fn(x)[0]
    temporal = time_fn(x)
    combined = np.column_stack([pooled, leg, temporal])
    result = {name: [] for name in ['mb_ridge', 'mb_pls32', 'rf_coordinate', 'rf_motionbert',
              'rf_concatenated', 'raw_cnn', 'temporal_cnn', 'mb_leg_time_ridge']}
    checks = []
    for fold in range(5):
        ridge_path = CPU / f'expanded695-pls-study/models/fold{fold}_ridge.npz'
        pls_path = CPU / f'expanded695-pls64-study/models/fold{fold}_selected.npz'
        source(ridge_path, read(CPU / 'expanded695-pls-study/artifact_manifest.json')[f'models/fold{fold}_ridge.npz'])
        source(pls_path, read(CPU / 'expanded695-pls64-study/artifact_manifest.json')[f'models/fold{fold}_selected.npz'])
        values = {'mb_ridge': linear(pooled, ridge_path), 'mb_pls32': linear(pooled, pls_path, True)}
        for name, features in [('rf_coordinate', kin), ('rf_motionbert', pooled),
                               ('rf_concatenated', np.column_stack([kin, pooled]))]:
            folder = FUSION / 'rf' / name / f'fold{fold}'
            meta = read(source(folder / 'complete.json'))
            model = joblib.load(source(folder / 'model.joblib', meta['model_sha256']))
            values[name] = model.predict(features)
        for name, folder, features, cls, ch in [
            ('raw_cnn', FUSION / f'cnn/raw/fold{fold}', x, raw.GdiCNN, raw.channels),
            ('temporal_cnn', FUSION / f'cnn/temporal/fold{fold}', frames, temporal_cls,
             lambda a: a.transpose(1, 2).contiguous())]:
            complete = read(source(folder / 'complete.json'))
            values[name] = cnn_predict(features, folder / 'refit/latest.pt', cls, ch, complete['checkpoint_sha256'])
        mean_gap = []
        for target in ['mean', 'gap']:
            path = CPU / f'mean-leg-followup695/models/fold{fold}_{target}_combined_temporal_ridge.npz'
            source(path, read(CPU / 'mean-leg-followup695/artifact_manifest.json')[path.relative_to(CPU / 'mean-leg-followup695').as_posix()])
            a = arrays(path)
            mean_gap.append(((combined - a['mean']) / a['scale']) @ a['coef'] + a['intercept'])
        mean, gap = mean_gap
        values['mb_leg_time_ridge'] = np.column_stack([mean + gap / 2, mean - gap / 2])
        for name, pred in values.items():
            assert pred.shape == (len(x), 2) and np.isfinite(pred).all(), name
            result[name].append(pred)
        if replay:
            # Caller supplies one original held-out record for each fold.
            for name in ['rf_coordinate', 'rf_motionbert', 'rf_concatenated', 'raw_cnn', 'temporal_cnn']:
                folder = (FUSION / f'rf/{name}/fold{fold}' if name.startswith('rf_') else
                          FUSION / f'cnn/{"raw" if name == "raw_cnn" else "temporal"}/fold{fold}')
                saved = arrays(folder / ('prediction.npz' if name.startswith('rf_') else 'predictions.npz'))
                index = int(REPLAY_INDICES[fold])
                local = np.flatnonzero(saved['indices'] == index) if 'indices' in saved else np.array([0])
                expected = saved['prediction' if name.startswith('rf_') else 'predictions'][local[0]]
                delta = float(abs(values[name][fold] - expected).max())
                tolerance = 1e-8 if name.startswith('rf_') else 1e-4
                assert delta <= tolerance, (name, fold, delta)
                checks.append({'model': name, 'fold': fold, 'max_difference': delta, 'tolerance': tolerance})
            for name, cube in [('mb_ridge', SOURCE_LINEAR['ridge']), ('mb_pls32', SOURCE_LINEAR['pls']),
                                ('mb_leg_time_ridge', SOURCE_LINEAR['time'])]:
                expected = cube[fold, REPLAY_INDICES[fold]]
                delta = float(abs(values[name][fold] - expected).max())
                assert delta <= 1e-8, (name, fold, delta)
                checks.append({'model': name, 'fold': fold, 'max_difference': delta, 'tolerance': 1e-8})
        print(json.dumps({'h36m_fold_complete': fold, 'records': len(x), 'source_replay': replay}), flush=True)
    result = {k: np.stack(v) for k, v in result.items()}
    for name, a, b in [('rf_coordinate_pls_equal', 'rf_coordinate', 'mb_pls32'),
                       ('rf_concatenated_pls_equal', 'rf_concatenated', 'mb_pls32'),
                       ('raw_cnn_pls_equal', 'raw_cnn', 'mb_pls32'),
                       ('raw_cnn_rf_coordinate_equal', 'raw_cnn', 'rf_coordinate'),
                       ('temporal_cnn_pls_equal', 'temporal_cnn', 'mb_pls32')]:
        result[name] = (result[a] + result[b]) / 2
    return result, checks


def body_predictions(x):
    sys.path.insert(0, str(BUNDLE))
    bundle_infer = module(BUNDLE / 'infer.py', 'association_bundle_infer')
    from model import GdiCNN, channels
    bundle_infer.verify_bundle()
    meta = read(source(BUNDLE / 'models.json'))
    results = {'multiwindow_cnn': [], 'multiwindow_high90': []}
    for fold in range(5):
        item = meta['folds'][fold]
        source(BUNDLE / item['file'], item['sha256'])
        a, b = item['calibration_slope_intercept']
        results['multiwindow_cnn'].append(bundle_infer.predict_windows(x, fold, meta).astype(float) * a + b)
        folder = STUDIES / f'20261031-cnn-high-gdi/high90_x3/received/results/fold{fold}'
        done = read(source(folder / 'complete.json'))
        ck = torch.load(source(folder / 'refit/latest.pt', done['checkpoint_sha256']), map_location='cpu', weights_only=True)
        model = GdiCNN(.5)
        model.load_state_dict(ck['model'], strict=True)
        model.requires_grad_(False).eval()
        with torch.inference_mode():
            value = (model(channels(torch.from_numpy(x)).reshape(-1, 12, 124))[:, 0] * ck['span'] + ck['lo']).reshape(-1, 2).numpy()
        a, b = done['calibration']
        results['multiwindow_high90'].append(value.astype(float) * a + b)
    return {k: np.stack(v) for k, v in results.items()}


def main():
    global REPLAY_INDICES, SOURCE_LINEAR
    assert not (HERE / 'inference-verification.json').exists(), 'Preserve verified inference'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    d = arrays(HERE / 'input.npz')
    # Only pose arrays are passed to predictors; clinical targets are not passed.
    previous = arrays(TRANSFER / 'prepared.npz')
    names = read(HERE / 'protocol.json')['models']
    if not (HERE / 'predictions.npz').exists():
        x = np.concatenate([d['x17'], d['h36m_x17']])
        if (HERE / 'features.npz').exists():
            cached = arrays(HERE / 'features.npz')
            pooled, frames = cached['pooled'], cached['frames']
        else:
            pooled, frames = encoder(x)
        # Small derived cache permits source-verification fixes without recomputing the encoder.
        np.savez_compressed(HERE / 'features.npz', pooled=pooled, frames=frames)
        result, _ = h36m_predictions(x, pooled, frames)
        body = body_predictions(d['cnn_x12'])
        primary = {name: value[:, :len(d['ids'])] for name, value in result.items()}
        lookup = {str(p): i for i, p in enumerate(d['cnn_ids'])}
        for name, value in body.items():
            primary[name] = value[:, [lookup[str(p)] for p in d['ids']]]
        assert set(primary) == set(names)
        np.savez_compressed(HERE / 'predictions.npz', models=np.array(names),
            primary=np.stack([primary[n] for n in names]), ids=d['ids'],
            h36m_models=np.array(list(result)), h36m=np.stack([v[:, len(d['ids']):] for v in result.values()]),
            h36m_owner=d['h36m_owner'], cnn_models=np.array(list(body)),
            cnn=np.stack(list(body.values())), cnn_ids=d['cnn_ids'])
    else:
        # Existing predictions are immutable; verification does not overwrite or choose them.
        print('Resuming source verification; preserving existing external predictions', flush=True)
        if (HERE / 'inference-sources-pending.json').exists():
            SOURCES.update(read(HERE / 'inference-sources-pending.json'))
    cached = arrays(HERE / 'features.npz')
    old = arrays(TORONTO / 'features.npz')
    np.testing.assert_allclose(cached['pooled'][len(d['ids']):], old['record_features'], rtol=1e-5, atol=2e-6)
    saved_predictions = arrays(HERE / 'predictions.npz')
    index = saved_predictions['cnn_models'].tolist().index('multiwindow_cnn')
    np.testing.assert_allclose(saved_predictions['cnn'][index].mean(0), previous['gdi_score'], rtol=0, atol=1e-5)
    save('inference-sources-pending.json', SOURCES)
    print('External predictions fixed; replaying source predictors', flush=True)
    original = arrays(FUSION / 'data.npz')
    REPLAY_INDICES = np.array([np.flatnonzero(original['fold'] == f)[0] for f in range(5)])
    source(MB / 'assets/MB_lite.bin')
    frame_path = source(FUSION / 'frame_features.npy', read(FUSION / 'frame_features.json')['sha256'])
    source_frames = np.asarray(np.load(frame_path, mmap_mode='r')[REPLAY_INDICES])
    ridge = arrays(CPU / 'expanded695-pls-study/fold_predictions.npz')
    pls = arrays(CPU / 'expanded695-pls64-study/fold_predictions.npz')
    time = arrays(CPU / 'mean-leg-followup695/fold_predictions.npz')
    time_names = arrays(CPU / 'mean-leg-followup695/oof_predictions.npz')['methods'].tolist()
    SOURCE_LINEAR = {'ridge': ridge['predictions'][ridge['methods'].tolist().index('ridge')],
        'pls': pls['predictions'], 'time': time['predictions'][time_names.index('joint_combined_temporal')]}
    _, checks = h36m_predictions(original['x'][REPLAY_INDICES], original['motionbert'][REPLAY_INDICES], source_frames, True)
    multi = arrays(GPU / 'cnn-multiwindow695/bundle/data.npz')
    records = np.array([np.flatnonzero(multi['fold'] == f)[0] for f in range(5)])
    use = np.isin(multi['window_record'], records)
    replay = body_predictions(multi['x'][use])
    owner = multi['window_record'][use]
    for name, path in [('multiwindow_cnn', GPU / 'cnn-multiwindow695/analysis/cnn_oof.npz'),
                       ('multiwindow_high90', STUDIES / '20261031-cnn-high-gdi/high90_x3/received/results/oof_predictions.npz')]:
        saved = arrays(path)
        for f, record in enumerate(records):
            actual = replay[name][f, owner == record].mean(0)
            expected = saved['predictions'][record].reshape(-1, 2)[0]
            delta = float(abs(actual - expected).max())
            assert delta <= 1e-4, (name, f, delta)
            checks.append({'model': name, 'fold': f, 'max_difference': delta, 'tolerance': 1e-4})
    for path, expected in SOURCES.items():
        assert sha(path) == expected, path
    save('inference-verification.json', {'passed': True, 'source_replays': checks,
        'toronto_cached_encoder_replayed_records': len(old['record_features']),
        'previous_multiwindow_predictions_replayed_people': len(previous['ids']),
        'models': names, 'training_performed': False, 'target_calibration_fitted': False,
        'not_included': {'single_cnn_original': 'Original verification says checkpoints_downloaded=false; no final weights at the recorded refit path.',
                        'motionbert_partial_finetuning': 'No local encoder checkpoints in inspected received folder.',
                        'rare_sqrt_cap4': 'No completed local final checkpoint set in inspected received folder.'},
        'prediction_sha256': sha(HERE / 'predictions.npz')})
    save('inference-sources.json', SOURCES)
    print('FROZEN_EXTERNAL_INFERENCE_AND_SOURCE_REPLAY_FINISHED', flush=True)


if __name__ == '__main__':
    main()
