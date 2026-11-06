"""Independent sklearn refits, person splits, weighted metrics, and source preservation."""
import os
import sys
import json
import csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
BRANCH = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(BRANCH))
import cpu_head as m
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def fit(method, k, x, y):
    if method == 'pca_ridge':
        return make_pipeline(StandardScaler(), PCA(n_components=k, svd_solver='full', whiten=False),
                             Ridge(alpha=1.0, solver='svd')).fit(x, y)
    if method == 'pls':
        return make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False,
                             max_iter=500, tol=1e-6)).fit(x, y)
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver='svd')).fit(x, y)


def weighted_metrics(d, pred, rows, band=None):
    # Construct explicit record-side weights, independent of the fitting script's aggregation.
    mask = np.broadcast_to(rows[:, None], pred.shape).copy()
    if band is not None:
        mask &= np.floor(d['y'] / 10) == band
    weight = np.zeros(pred.shape)
    unique = np.unique(d['person_id'][mask.any(1)])
    truth_mean, pred_mean = [], []
    for person in unique:
        local = mask & (d['person_id'] == person)[:, None]
        weight[local] = 1.0 / len(unique) / local.sum()
        truth_mean.append(d['y'][local].mean())
        pred_mean.append(pred[local].mean())
    np.testing.assert_allclose(weight.sum(), 1, rtol=0, atol=1e-14)
    error = pred - d['y']
    gap = abs(np.floor(pred / 10) - np.floor(d['y'] / 10))
    ts, ps = np.std(truth_mean), np.std(pred_mean)
    return dict(people=len(unique), person_MAE=float((weight * abs(error)).sum()),
        bias=float((weight * error).sum()), person_RMSE=float(np.sqrt((weight * error ** 2).sum())),
        exact_bin_rate=float((weight * (gap == 0)).sum()),
        within_one_bin_rate=float((weight * (gap <= 1)).sum()),
        two_or_more_bin_rate=float((weight * (gap >= 2)).sum()),
        truth_person_mean=float(np.mean(truth_mean)), prediction_person_mean=float(np.mean(pred_mean)),
        truth_person_sd=float(ts), prediction_person_sd=float(ps), prediction_truth_sd_ratio=float(ps / ts))


def main():
    protocol = m.load_json(HERE / 'protocol.json')
    provenance = m.load_json(HERE / 'data_provenance.json')
    summary = m.load_json(HERE / 'summary.json')
    assert m.digest(HERE / 'protocol.json') == summary['protocol_sha256']
    assert m.digest(HERE / 'development695.npz') == provenance['dataset_sha256']
    for name, expected in provenance['source_hashes'].items():
        assert m.digest(m.WORKSPACE / name) == expected, name
    d = load_npz(HERE / 'development695.npz')
    cube = load_npz(HERE / 'fold_predictions.npz')['predictions']
    out = load_npz(HERE / 'oof_predictions.npz')
    methods = out['methods'].tolist()
    assert methods == ['old140_ridge', 'ridge', 'pca_ridge', 'pls']
    assert cube.shape == (4, 5, 1064, 2) and np.isfinite(cube).all()
    for key in ['person_id', 'video_id', 'fold']:
        np.testing.assert_array_equal(out[key], d[key])
    np.testing.assert_array_equal(out['predictions'], cube[:, d['fold'], np.arange(1064)])
    meta = m.load_json(HERE / 'oof.metadata.json')
    assert meta['predictions_sha256'] == m.digest(HERE / 'oof_predictions.npz')
    assert meta['dataset_sha256'] == provenance['dataset_sha256']
    old, heads, baseline, _ = m.source_data()
    np.testing.assert_array_equal(d['X'][:203], old['features'][:, 0])
    np.testing.assert_array_equal(d['y'][:203], old['y'])
    np.testing.assert_array_equal(d['fold'][:203], old['fold'])
    new = load_npz(m.ROOT / 'studies/20261027-gdi-linkage/development_features.npz')
    np.testing.assert_array_equal(d['X'][203:], new['X'])
    np.testing.assert_array_equal(d['y'][203:], new['Y'])
    np.testing.assert_array_equal(d['person_id'][203:], new['patient_ids'])
    np.testing.assert_array_equal(d['video_id'][203:], new['video_ids'])
    split = m.load_json(m.ROOT / 'studies/20261027-window-ablation/new_person_split.json')
    exclude = m.load_json(m.ROOT / 'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')
    reserved = {r['patient_id'] for r in split['people'] if r['reserved_role'] == 'final_evaluation_reserved'}
    protected37 = set(exclude['development_patient_ids']) - set(old['person_id'].astype(str))
    protected38 = set(exclude['original_test_patient_ids'])
    assert len(reserved) == 223 and len(protected37) == 37 and len(protected38) == 38
    assert not set(d['person_id']) & (reserved | protected37 | protected38)
    np.testing.assert_allclose(out['predictions'][0, :203], baseline[:, 0], rtol=0, atol=1e-10)
    perm = np.random.default_rng(20261030).permutation(np.unique(d['person_id'][203:]))
    for fold, group in enumerate(np.array_split(perm, 5)):
        assert set(d['person_id'][203:][d['fold'][203:] == fold]) == set(group)
    max_inner, max_outer, max_latent = 0.0, 0.0, 0.0
    inner_count, outer_count, selections = 0, 0, []
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr, te = np.flatnonzero(d['fold'] != fold), np.flatnonzero(d['fold'] == fold)
            assert len(set(d['person_id'][tr])) == 556 and len(set(d['person_id'][te])) == 139
            assert not set(d['person_id'][tr]) & set(d['person_id'][te])
            prefix = f'fold{fold}__'
            reference = ((d['X'] - heads[prefix + 'scaler_mean']) / heads[prefix + 'scaler_scale']) @ heads[prefix + 'coef'].T + heads[prefix + 'intercept']
            np.testing.assert_array_equal(reference, cube[0, fold])
            saved = load_npz(HERE / f'models/fold{fold}_inner.npz')
            np.testing.assert_array_equal(tr, saved['outer_train_indices'])
            groups = np.array_split(np.random.default_rng(20261030 + fold).permutation(np.unique(d['person_id'][tr])), 4)
            assignments = np.full(len(tr), -1)
            for inner, group in enumerate(groups):
                assignments[np.isin(d['person_id'][tr], group)] = inner
            np.testing.assert_array_equal(assignments, saved['inner_assignment'])
            selected = {}
            for method, counts in [('pca_ridge', [8, 16, 32, 64]), ('pls', [2, 4, 8, 16])]:
                independent_oof = np.empty_like(saved[method + '_inner_oof'])
                for inner in range(4):
                    it, iv = tr[assignments != inner], tr[assignments == inner]
                    assert len(set(d['person_id'][it])) == 417 and len(set(d['person_id'][iv])) == 139
                    assert not set(d['person_id'][it]) & set(d['person_id'][iv])
                    assert not set(d['person_id'][it]) & set(d['person_id'][te])
                    np.testing.assert_array_equal(saved[f'inner{inner}__train_indices'], it)
                    np.testing.assert_array_equal(saved[f'inner{inner}__validation_indices'], iv)
                    for ci, k in enumerate(counts):
                        model = fit(method, k, d['X'][it], d['y'][it])
                        pred = model.predict(d['X'][iv])
                        independent_oof[ci, assignments == inner] = pred
                        key = f'inner{inner}__{method}_k{k}__'
                        np.testing.assert_allclose(saved[key + 'scaler_mean'], model[0].mean_, rtol=0, atol=1e-13)
                        np.testing.assert_allclose(saved[key + 'scaler_scale'], model[0].scale_, rtol=0, atol=1e-13)
                        replay = ((d['X'][iv] - saved[key + 'scaler_mean']) / saved[key + 'scaler_scale']) @ saved[key + 'combined_coef'].T + saved[key + 'combined_intercept']
                        np.testing.assert_allclose(replay, pred, rtol=0, atol=1e-8)
                        max_inner = max(max_inner, float(abs(replay - pred).max()))
                        inner_count += 1
                np.testing.assert_allclose(independent_oof, saved[method + '_inner_oof'], rtol=0, atol=1e-8)
                errors = []
                for ci, k in enumerate(counts):
                    value = np.mean([np.abs(independent_oof[ci, d['person_id'][tr] == p] - d['y'][tr][d['person_id'][tr] == p]).mean() for p in np.unique(d['person_id'][tr])])
                    errors.append(value)
                selected[method] = min(k for k, value in zip(counts, errors) if value <= min(errors) + 1e-12)
            for mi, method in enumerate(methods[1:], 1):
                k = selected.get(method, 512)
                saved_model = load_npz(HERE / f'models/fold{fold}_{method}.npz')
                assert saved_model['selected_k'] == k
                np.testing.assert_array_equal(saved_model['train_indices'], tr)
                np.testing.assert_array_equal(saved_model['evaluation_indices'], te)
                model = fit(method, k, d['X'][tr], d['y'][tr])
                pred = model.predict(d['X'])
                np.testing.assert_allclose(pred, cube[mi, fold], rtol=0, atol=1e-8)
                replay = ((d['X'] - saved_model['scaler_mean']) / saved_model['scaler_scale']) @ saved_model['combined_coef'].T + saved_model['combined_intercept']
                np.testing.assert_array_equal(replay, cube[mi, fold])
                if method == 'pls':
                    # Independent latent-score identity, without using combined_coef.
                    z = (d['X'] - saved_model['scaler_mean']) / saved_model['scaler_scale']
                    latent = (z - saved_model['x_mean']) @ saved_model['x_rotations']
                    pred2 = latent @ saved_model['y_loadings'].T + saved_model['y_mean']
                    np.testing.assert_allclose(pred2, pred, rtol=0, atol=1e-8)
                    max_latent = max(max_latent, float(abs(pred2 - pred).max()))
                max_outer = max(max_outer, float(abs(pred - cube[mi, fold]).max()))
                outer_count += 1
            row = summary['selected_dimensions'][fold]
            assert row['pca_components'] == selected['pca_ridge'] and row['pls_components'] == selected['pls']
            selections.append(selected)
            print(f'Independent outer fold {fold}: 32 inner and 3 outer refits checked', flush=True)
    nmetrics = 0
    candidate_rows = list(csv.DictReader((HERE / 'inner_candidate_scores.csv').open(encoding='utf-8-sig')))
    assert len(candidate_rows) == 40
    for fold in range(5):
        inner = load_npz(HERE / f'models/fold{fold}_inner.npz')
        tr = inner['outer_train_indices']
        for method, counts in [('pca_ridge', [8, 16, 32, 64]), ('pls', [2, 4, 8, 16])]:
            for ci, k in enumerate(counts):
                pred = inner[method + '_inner_oof'][ci]
                value = np.mean([abs(pred[d['person_id'][tr] == p] - d['y'][tr][d['person_id'][tr] == p]).mean() for p in np.unique(d['person_id'][tr])])
                row = next(r for r in candidate_rows if int(r['outer_fold']) == fold and r['method'] == method and int(r['components']) == k)
                np.testing.assert_allclose(value, float(row['inner_person_MAE']), rtol=0, atol=1e-10)
    for row in summary['metrics']:
        use = np.ones(1064, dtype=bool) if row['group'] == 'all695' else d['source_group'] == row['group']
        calculated = weighted_metrics(d, out['predictions'][methods.index(row['method'])], use)
        for key, value in calculated.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-10)
        nmetrics += 1
    for row in summary['score_bands']:
        calculated = weighted_metrics(d, out['predictions'][methods.index(row['method'])], np.ones(1064, dtype=bool), row['actual_bin_low'] // 10)
        for key, value in calculated.items():
            np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-10)
        nmetrics += 1
    for filename, expected_rows in [('metrics.csv', summary['metrics']), ('score_bands.csv', summary['score_bands'])]:
        rows = list(csv.DictReader((HERE / filename).open(encoding='utf-8-sig')))
        assert len(rows) == len(expected_rows)
        for row, expected in zip(rows, expected_rows):
            assert set(row) == set(expected)
            for key, value in expected.items():
                if isinstance(value, str):
                    assert row[key] == value
                else:
                    np.testing.assert_allclose(float(row[key]), value, rtol=0, atol=1e-10)
    errors = np.array([[abs(pred[d['person_id'] == p] - d['y'][d['person_id'] == p]).mean() for p in np.unique(d['person_id'])] for pred in out['predictions']])
    draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
    for row in summary['comparisons']:
        delta = errors[methods.index(row['baseline'])] - errors[methods.index(row['candidate'])]
        np.testing.assert_allclose(delta.mean(), row['MAE_reduction'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.percentile(delta[draws].mean(1), [2.5, 97.5]), row['bootstrap95'], rtol=0, atol=1e-12)
    preserved = 0
    for manifest_path, base in [(BRANCH / 'followup/preserved_primary_files.json', BRANCH),
        (BRANCH / 'followup/pca-ridge-study/artifact_manifest.json', BRANCH / 'followup/pca-ridge-study'),
        (BRANCH / 'followup/ridge-alpha-study/artifact_manifest.json', BRANCH / 'followup/ridge-alpha-study'),
        (BRANCH / 'followup/train_eval_diagnostic/artifact_manifest.json', BRANCH / 'followup/train_eval_diagnostic')]:
        for path, expected in m.load_json(manifest_path).items():
            assert m.digest(base / path) == expected, str(base / path)
            preserved += 1
    for name, expected in provenance['source_hashes'].items():
        assert m.digest(m.WORKSPACE / name) == expected
    result = dict(verified=True, independent_inner_refits=inner_count, independent_outer_refits=outer_count,
        max_inner_prediction_difference=max_inner, max_outer_prediction_difference=max_outer,
        max_PLS_latent_identity_difference=max_latent, weighted_metric_rows_checked=nmetrics,
        candidate_score_rows_checked=40, paired_bootstrap_comparisons_checked=4,
        protected_cohort_ID_overlap=0, protected_numeric_data_read=False, preserved_previous_artifacts=preserved,
        original_folds_preserved=True, inner_outer_person_overlap=0, selected_dimensions=selections,
        source_hashes_rechecked=len(provenance['source_hashes']), GPU_used=False)
    (HERE / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
