"""Independent sample-weight calculations; no model fitting or encoder inference."""
import copy
from pathlib import Path
import diagnose as a
import numpy as np

d, heads, baseline, sources = a.m.source_data()
s = a.m.load_json(a.HERE / 'summary.json')
old = a.m.load_json(a.HERE.parent / 'cpu_comparison/summary.json')
first = a.m.load_json(a.m.HERE / 'runs/main/verification.json')
checks = []
people = np.unique(d['person_id'])
truth = d['y']
person_truth = np.array([truth[d['person_id'] == pid].mean() for pid in people])
truth_sd = float(person_truth.std(ddof=0))
for name in ['ridge', 'cpu_head_only']:
    path = a.HERE / (name + '_fold_predictions.npz')
    cube, meta = a.load_cube(path, a.HERE / (name + '.metadata.json'), d, sources)
    oof = cube[d['fold'], np.arange(203)]
    expected = baseline
    if name == 'cpu_head_only':
        with np.load(a.m.HERE / 'runs/main/all_predictions.npz', allow_pickle=False) as z:
            expected = z['predictions'].copy()
    assert float(np.max(abs(oof - expected))) < 1e-10
    for fold in range(5):
        tr = d['fold'] != fold
        mse = float(np.square(cube[fold, tr, 0] - truth[tr]).mean())
        key = 'training_record_equal_MSE_before' if name == 'ridge' else 'training_record_equal_MSE_after'
        assert abs(mse - first['folds'][fold][key]) < 1e-10
    for role in ['train_people', 'held_people']:
        views = 4 if role == 'train_people' else 1
        weights = np.zeros((5, 203, 2))
        for fold in range(5):
            chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
            for i in np.flatnonzero(chosen):
                records = np.count_nonzero(d['person_id'] == d['person_id'][i])
                weights[fold, i, :] = 1 / (175 * records * 2 * views)
        assert abs(weights.sum() - 1) < 1e-12
        for wi, window in enumerate(['front', 'back']):
            row = next(x for x in s['metrics'] if x['model'] == name and x['role'] == role and x['window'] == window)
            p = cube[:, :, wi]
            error = p - truth[None, :, :]
            gap = abs(np.floor(p / 10) - np.floor(truth[None] / 10))
            values = {'mae': np.sum(weights * abs(error)), 'bias': np.sum(weights * error),
                      'mse': np.sum(weights * error**2), 'exact_bin_rate': np.sum(weights * (gap == 0)),
                      'within_one_bin_rate': np.sum(weights * (gap <= 1)),
                      'two_or_more_bin_rate': np.sum(weights * (gap >= 2))}
            for key, value in values.items():
                assert abs(row[key] - value) < 1e-10, (name, role, window, key)
            assert abs(row['truth_person_sd_ddof0'] - truth_sd) < 1e-12
            scores = []
            for pid in people:
                for fold in range(5):
                    include = d['fold'][d['person_id'] == pid][0] != fold if role == 'train_people' else d['fold'][d['person_id'] == pid][0] == fold
                    if include:
                        scores.append(cube[fold, d['person_id'] == pid, wi].mean())
            assert len(scores) == 175 * views
            assert abs(np.std(scores, ddof=0) - row['prediction_person_sd_ddof0']) < 1e-12
            if role == 'held_people':
                prior_name = 'ridge' if name == 'ridge' else 'cpu_head_only'
                prior = next(x for x in old['metrics'] if x['model'] == prior_name and x['window'] == window)
                for new_key, old_key in [('mae', 'person_macro_mae'), ('bias', 'person_macro_bias'),
                                         ('exact_bin_rate', 'exact_bin_rate'), ('within_one_bin_rate', 'within_one_bin_rate')]:
                    assert abs(row[new_key] - prior[old_key]) < 1e-12
            for bandrow in [x for x in s['score_bands'] if x['model'] == name and x['role'] == role and x['window'] == window]:
                bandmask = (truth >= bandrow['actual_bin_low']) & (truth < bandrow['actual_bin_high_exclusive'])
                present = [pid for pid in people if np.any(bandmask & (d['person_id'] == pid)[:, None])]
                bw = np.zeros_like(weights)
                for fold in range(5):
                    chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
                    for pid in present:
                        use = bandmask & (d['person_id'] == pid)[:, None] & chosen[:, None]
                        count = np.count_nonzero(bandmask & (d['person_id'] == pid)[:, None])
                        bw[fold][use] = 1 / (len(present) * count * views)
                assert abs(bw.sum() - 1) < 1e-12
                for key, value in [('mae', np.sum(bw * abs(error))), ('bias', np.sum(bw * error)),
                                   ('exact_bin_rate', np.sum(bw * (gap == 0)))]:
                    assert abs(bandrow[key] - value) < 1e-10
            checks.append({'model': name, 'role': role, 'window': window,
                           'independent_weights_sum': float(weights.sum()), 'metrics_and_spread_checked': True})

# Verify that a valid completed CPU artifact can use the future GPU-input route.
cube, meta = a.load_cube(a.HERE / 'cpu_head_only_fold_predictions.npz', a.HERE / 'cpu_head_only.metadata.json', d, sources)
rejections = []
for case in ['wrong_dataset', 'wrong_model_order', 'wrong_sides', 'wrong_window', 'missing_fold', 'person_leakage']:
    bad = copy.deepcopy(meta)
    if case == 'wrong_dataset': bad['dataset_sha256'] = '0' * 64
    if case == 'wrong_model_order': bad['fold_model_order'] = list(reversed(range(5)))
    if case == 'wrong_sides': bad['sides'] = list(reversed(bad['sides']))
    if case == 'wrong_window': bad['windows'] = list(reversed(bad['windows']))
    if case == 'missing_fold': bad['fold_models'] = bad['fold_models'][:-1]
    if case == 'person_leakage': bad['fold_models'][0]['training_people'][0] = bad['fold_models'][0]['evaluation_people'][0]
    target = a.HERE / 'rejection_fixtures' / (case + '.json')
    a.write_json(target, bad)
    try:
        a.load_cube(a.HERE / 'cpu_head_only_fold_predictions.npz', target, d, sources)
    except ValueError:
        rejections.append(case)
    else:
        raise AssertionError('bad metadata accepted: ' + case)
assert len(rejections) == 6
route = a.m.load_json(a.HERE / 'external_route_check/summary.json')
for section, selectors in [('metrics', ['role', 'window']),
                           ('score_bands', ['role', 'window', 'actual_bin_low']),
                           ('fold_metrics', ['role', 'window', 'fold'])]:
    for row in s[section]:
        model = 'ridge' if row['model'] == 'ridge' else 'saved_head_route_check'
        other = next(x for x in route[section] if x['model'] == model and all(x[k] == row[k] for k in selectors))
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert abs(value - other[key]) < 1e-10, (section, key)
for path, digest in a.m.load_json(a.HERE.parent / 'preserved_primary_files.json').items():
    assert a.m.digest(a.m.HERE / path) == digest
assert a.m.source_data()[3] == sources
assert len(s['metrics']) == 8 and len(s['score_bands']) == 64 and len(s['fold_metrics']) == 40
result = {'passed': True, 'checks': checks, 'all_64_band_rows_checked': True,
    'OOF_predictions_reproduce_prior_results': True, 'training_record_MSE_reproduces_first_study': True,
    'person_weighting_checks_training_model_views': 4, 'person_weighting_checks_held_model_views': 1,
    'no_700_person_claim': True, 'GPU_metadata_refusal_cases': rejections,
    'source_and_primary_artifacts_unchanged': True, 'additional_training_updates': 0,
    'external_cli_route_checked_with_completed_CPU_artifact': True,
    'pose_encoder_loaded': False, 'gpu_used': False, 'source_hashes': sources}
a.write_json(a.HERE / 'verification.json', result)
print(__import__('json').dumps({'passed': True, 'metric_rows': 8, 'band_rows': 64, 'fold_rows': 40,
                               'metadata_refusal_cases': len(rejections)}, ensure_ascii=False))
