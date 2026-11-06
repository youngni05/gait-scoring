"""Independent Ridge fits, coefficient replay and selected person-weight checks."""
import compare as a
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

d, heads, baseline, sources = a.diag.m.source_data()
s = a.diag.m.load_json(a.HERE / 'summary.json')
assert s['protocol_sha256'] == a.diag.m.digest(a.HERE / 'protocol.json')
assert s['source_hashes'] == sources
expected_fold_rows = {(alpha, fold, role, window) for alpha in s['alphas'] for fold in range(5)
                      for role in ['train_people', 'held_people'] for window in ['front', 'back']}
actual_fold_rows = {(r['alpha'], r['fold'], r['role'], r['window']) for r in s['fold_metrics']}
assert actual_fold_rows == expected_fold_rows and len(s['fold_metrics']) == len(expected_fold_rows)
with np.load(a.HERE / 'all_fold_predictions.npz', allow_pickle=False) as z:
    cube = z['predictions'].copy()
    assert np.array_equal(z['alphas'], s['alphas'])
    for key in ['person_id', 'video_id', 'fold']:
        assert np.array_equal(z[key], d[key])
with np.load(a.HERE / 'fitted_heads.npz', allow_pickle=False) as z:
    fitted = {key: z[key].copy() for key in z.files}
assert cube.shape == (9, 5, 203, 2, 2) and np.isfinite(cube).all()
fit_checks = []
with threadpool_limits(limits=2):
    for fold in range(5):
        tr = d['fold'] != fold
        mu, scale = heads[f'fold{fold}__scaler_mean'], heads[f'fold{fold}__scaler_scale']
        z = (d['features'].astype(np.float64) - mu) / scale
        mse, norms = [], []
        for ai, alpha in enumerate(s['alphas']):
            p = f'alpha{ai}_fold{fold}__'
            assert np.array_equal(fitted[p + 'scaler_mean'], mu)
            assert np.array_equal(fitted[p + 'scaler_scale'], scale)
            independent = Ridge(alpha=alpha, solver='svd', fit_intercept=True).fit(z[tr, 0], d['y'][tr])
            prediction = independent.predict(z.reshape(-1, 512)).reshape(203, 2, 2)
            max_error = float(np.max(abs(prediction - cube[ai, fold])))
            if max_error > 1e-6:
                raise ValueError('independent solver prediction mismatch')
            replay = z @ fitted[p + 'coef'].T + fitted[p + 'intercept']
            assert float(np.max(abs(replay - cube[ai, fold]))) < 1e-10
            mse.append(float(np.square(cube[ai, fold, tr, 0] - d['y'][tr]).mean()))
            norms.append(float(np.linalg.norm(fitted[p + 'coef'])))
            fit_checks.append({'alpha': alpha, 'fold': fold, 'independent_prediction_max_error': max_error})
        assert (np.diff(mse) >= -1e-8).all(), 'training MSE not monotone with alpha'
        assert (np.diff(norms) <= 1e-8).all(), 'coefficient norm not monotone with alpha'

alpha_one = s['alphas'].index(1)
oof_alpha_one = cube[alpha_one, d['fold'], np.arange(203)]
baseline_error = float(np.max(abs(oof_alpha_one - baseline)))
assert baseline_error < 1e-8
previous = a.diag.m.load_json(a.HERE.parent / 'train_eval_diagnostic/summary.json')
for row in s['metrics']:
    if row['alpha'] == 1:
        old = next(x for x in previous['metrics'] if x['model'] == 'ridge' and x['role'] == row['role'] and x['window'] == row['window'])
        for key in ['mae', 'bias', 'mse', 'exact_bin_rate', 'prediction_person_sd_ddof0']:
            assert abs(row[key] - old[key]) < 1e-9

# Independently recompute all 36 aggregate rows using record-side weights.
for ai, alpha in enumerate(s['alphas']):
    for role in ['train_people', 'held_people']:
        views = 4 if role == 'train_people' else 1
        weights = np.zeros((5, 203, 2))
        for fold in range(5):
            chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
            for i in np.flatnonzero(chosen):
                n = int((d['person_id'] == d['person_id'][i]).sum())
                weights[fold, i] = 1 / (175 * n * 2 * views)
        assert abs(weights.sum() - 1) < 1e-12
        for wi, window in enumerate(['front', 'back']):
            row = next(x for x in s['metrics'] if x['alpha'] == alpha and x['role'] == role and x['window'] == window)
            delta = cube[ai, :, :, wi] - d['y'][None]
            gap = abs(np.floor(cube[ai, :, :, wi] / 10) - np.floor(d['y'][None] / 10))
            for key, value in [('mae', np.sum(weights * abs(delta))), ('bias', np.sum(weights * delta)),
                               ('mse', np.sum(weights * delta**2)), ('exact_bin_rate', np.sum(weights * (gap == 0)))]:
                assert abs(row[key] - value) < 1e-9

# Independently check the two informative tail bands for all alpha/model-role windows.
tail_rows = 0
for row in s['score_bands']:
    if row['actual_bin_low'] not in [40, 90]:
        continue
    ai = s['alphas'].index(row['alpha'])
    wi = ['front', 'back'].index(row['window'])
    views = 4 if row['role'] == 'train_people' else 1
    band = (d['y'] >= row['actual_bin_low']) & (d['y'] < row['actual_bin_high_exclusive'])
    people = [pid for pid in np.unique(d['person_id']) if (band & (d['person_id'] == pid)[:, None]).any()]
    w = np.zeros((5, 203, 2))
    for fold in range(5):
        chosen = d['fold'] != fold if row['role'] == 'train_people' else d['fold'] == fold
        for pid in people:
            count = np.count_nonzero(band & (d['person_id'] == pid)[:, None])
            use = band & (d['person_id'] == pid)[:, None] & chosen[:, None]
            w[fold][use] = 1 / (len(people) * count * views)
    delta = cube[ai, :, :, wi] - d['y'][None]
    assert abs(row['mae'] - np.sum(w * abs(delta))) < 1e-9
    assert abs(row['bias'] - np.sum(w * delta)) < 1e-9
    tail_rows += 1

for relative, h in a.diag.m.load_json(a.HERE.parent / 'preserved_primary_files.json').items():
    assert a.diag.m.digest(a.diag.m.HERE / relative) == h
old_diag = a.HERE.parent / 'train_eval_diagnostic'
for relative, h in a.diag.m.load_json(old_diag / 'artifact_manifest.json').items():
    assert a.diag.m.digest(old_diag / relative) == h
assert a.diag.m.source_data()[3] == sources
result = {'passed': True, 'primary_experimental_models': 45,
    'independent_sklearn_solver_checks': fit_checks,
    'independent_solver_max_prediction_error': max(x['independent_prediction_max_error'] for x in fit_checks),
    'alpha1_baseline_max_error': baseline_error, 'all_36_aggregate_rows_independently_checked': True,
    'tail_band_rows_independently_checked': tail_rows, 'saved_coefficients_replayed': True,
    'training_MSE_monotone_with_regularization': True, 'coefficient_norm_monotone_with_regularization': True,
    'source_and_previous_results_unchanged': True, 'fixed_grid_not_extended': True,
    'all_180_fold_role_window_combinations_present_once': True,
    'protected_cohorts_used': False, 'encoder_reinference': False, 'GPU_used': False,
    'independent_final_evaluation': False}
a.write_json(a.HERE / 'verification.json', result)
print(__import__('json').dumps({k: v for k, v in result.items() if k != 'independent_sklearn_solver_checks'}, ensure_ascii=False))
