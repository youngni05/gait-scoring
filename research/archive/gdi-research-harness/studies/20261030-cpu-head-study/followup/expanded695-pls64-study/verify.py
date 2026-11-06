"""Independent PLS pipeline refits and explicit person-weighted metric checks."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'expanded695-pls-study'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
import importlib.util
import json
import csv
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
spec = importlib.util.spec_from_file_location('parent_verify', PARENT / 'verify.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
load_npz = parent.load_npz


def main():
    provenance = m.load_json(HERE / 'provenance.json')
    protocol = m.load_json(HERE / 'protocol.json')
    summary = m.load_json(HERE / 'summary.json')
    assert m.digest(PARENT / 'artifact_manifest.json') == provenance['parent_artifact_manifest_sha256']
    assert m.digest(HERE / 'protocol.json') == provenance['protocol_sha256'] == summary['protocol_sha256']
    for name, expected in provenance['parent_artifacts'].items():
        assert m.digest(PARENT / name) == expected, name
    d = load_npz(PARENT / 'development695.npz')
    assert d['X'].shape == (1064, 512) and len(set(d['person_id'])) == 695
    out = load_npz(HERE / 'oof_predictions.npz')
    previous = load_npz(PARENT / 'oof_predictions.npz')
    np.testing.assert_array_equal(out['predictions'][:4], previous['predictions'])
    methods = out['methods'].tolist()
    assert methods == ['old140_ridge', 'ridge', 'pca_ridge', 'pls16', 'pls64']
    for key in ['person_id', 'video_id', 'fold']:
        np.testing.assert_array_equal(out[key], d[key])
        np.testing.assert_array_equal(out[key], previous[key])
    cube = load_npz(HERE / 'fold_predictions.npz')['predictions']
    assert cube.shape == (5, 1064, 2) and np.isfinite(cube).all()
    np.testing.assert_array_equal(cube[d['fold'], np.arange(1064)], out['predictions'][4])
    meta = m.load_json(HERE / 'oof.metadata.json')
    assert meta['predictions_sha256'] == m.digest(HERE / 'oof_predictions.npz')
    assert meta['dataset_sha256'] == m.digest(PARENT / 'development695.npz') == provenance['dataset_sha256']
    inner_count, max_inner, max_outer, max_latent = 0, 0., 0., 0.
    scores, selections = [], []
    max_iter = 0
    with threadpool_limits(limits=2):
        for outer in range(5):
            tr, te = np.flatnonzero(d['fold'] != outer), np.flatnonzero(d['fold'] == outer)
            assert len(set(d['person_id'][tr])) == 556 and len(set(d['person_id'][te])) == 139
            assert not set(d['person_id'][tr]) & set(d['person_id'][te])
            saved = load_npz(HERE / f'models/fold{outer}_inner.npz')
            old = load_npz(PARENT / f'models/fold{outer}_inner.npz')
            np.testing.assert_array_equal(saved['outer_train_indices'], tr)
            np.testing.assert_array_equal(saved['inner_assignment'], old['inner_assignment'])
            expected_groups = np.array_split(np.random.default_rng(20261030 + outer).permutation(np.unique(d['person_id'][tr])), 4)
            assignment = saved['inner_assignment']
            inner_oof = np.full((6, len(tr), 2), np.nan)
            for inner, group in enumerate(expected_groups):
                it, iv = tr[assignment != inner], tr[assignment == inner]
                assert set(d['person_id'][iv]) == set(group)
                assert len(set(d['person_id'][it])) == 417 and len(set(d['person_id'][iv])) == 139
                assert not set(d['person_id'][it]) & set(d['person_id'][iv])
                assert not set(d['person_id'][it]) & set(d['person_id'][te])
                for suffix, indices in [('train_indices', it), ('validation_indices', iv)]:
                    np.testing.assert_array_equal(saved[f'inner{inner}__{suffix}'], indices)
                    np.testing.assert_array_equal(old[f'inner{inner}__{suffix}'], indices)
                for ci, k in enumerate(protocol['pls_components']):
                    model = make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False,
                                          max_iter=500, tol=1e-6)).fit(d['X'][it], d['y'][it])
                    pred = model.predict(d['X'][iv])
                    inner_oof[ci, assignment == inner] = pred
                    prefix = f'inner{inner}__k{k}__'
                    for key, expected in [('scaler_mean', model[0].mean_), ('scaler_scale', model[0].scale_)]:
                        np.testing.assert_array_equal(saved[prefix + key], expected)
                    replay = ((d['X'][iv] - saved[prefix + 'scaler_mean']) / saved[prefix + 'scaler_scale']) @ saved[prefix + 'coef'].T + saved[prefix + 'intercept']
                    np.testing.assert_allclose(pred, replay, rtol=0, atol=1e-8)
                    max_inner = max(max_inner, float(abs(pred - replay).max()))
                    max_iter = max(max_iter, max(model[1].n_iter_))
                    inner_count += 1
            np.testing.assert_allclose(inner_oof, saved['inner_oof_predictions'], rtol=0, atol=1e-8)
            np.testing.assert_allclose(inner_oof[:4], old['pls_inner_oof'], rtol=0, atol=1e-8)
            values = [np.mean([abs(pred[d['person_id'][tr] == p] - d['y'][tr][d['person_id'][tr] == p]).mean() for p in np.unique(d['person_id'][tr])]) for pred in inner_oof]
            selected = min(k for k, value in zip(protocol['pls_components'], values) if value <= min(values) + 1e-12)
            selections.append(selected)
            scores.extend(dict(outer_fold=outer, components=k, inner_person_MAE=float(value)) for k, value in zip(protocol['pls_components'], values))
            outer_model = load_npz(HERE / f'models/fold{outer}_selected.npz')
            assert int(outer_model['selected_k']) == selected == summary['selected_dimensions'][outer]['selected_components']
            np.testing.assert_array_equal(tr, outer_model['train_indices'])
            np.testing.assert_array_equal(te, outer_model['evaluation_indices'])
            model = make_pipeline(StandardScaler(), PLSRegression(n_components=selected, scale=False,
                                  max_iter=500, tol=1e-6)).fit(d['X'][tr], d['y'][tr])
            pred = model.predict(d['X'])
            np.testing.assert_allclose(pred, cube[outer], rtol=0, atol=1e-8)
            z = (d['X'] - outer_model['scaler_mean']) / outer_model['scaler_scale']
            replay = z @ outer_model['coef'].T + outer_model['intercept']
            np.testing.assert_array_equal(replay, cube[outer])
            latent = (z - outer_model['x_mean']) @ outer_model['x_rotations'] @ outer_model['y_loadings'].T + outer_model['y_mean']
            np.testing.assert_allclose(latent, pred, rtol=0, atol=1e-8)
            max_outer = max(max_outer, float(abs(pred - cube[outer]).max()))
            max_latent = max(max_latent, float(abs(latent - pred).max()))
            print(f'Independent fold {outer}: k={selected}, 24 inner and 1 outer refit verified', flush=True)
    rows = list(csv.DictReader((HERE / 'inner_candidate_scores.csv').open(encoding='utf-8-sig')))
    assert len(rows) == len(scores) == 30
    for expected, row in zip(scores, rows):
        for key, value in expected.items():
            np.testing.assert_allclose(float(row[key]), value, rtol=0, atol=1e-10)
    nmetrics = 0
    for filename, rows in [('metrics.csv', summary['metrics']), ('score_bands.csv', summary['score_bands'])]:
        exported = list(csv.DictReader((HERE / filename).open(encoding='utf-8-sig')))
        assert len(exported) == len(rows)
        for row, saved_csv in zip(rows, exported):
            if 'group' in row:
                use = np.ones(1064, dtype=bool) if row['group'] == 'all695' else d['source_group'] == row['group']
                band = None
            else:
                use, band = np.ones(1064, dtype=bool), row['actual_bin_low'] // 10
            calculated = parent.weighted_metrics(d, out['predictions'][methods.index(row['method'])], use, band)
            for key, value in calculated.items():
                np.testing.assert_allclose(row[key], value, rtol=0, atol=1e-10)
                np.testing.assert_allclose(float(saved_csv[key]), value, rtol=0, atol=1e-10)
            nmetrics += 1
    errors = np.array([[abs(pred[d['person_id'] == p] - d['y'][d['person_id'] == p]).mean() for p in np.unique(d['person_id'])] for pred in out['predictions']])
    draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
    for row in summary['comparisons']:
        delta = errors[methods.index(row['baseline'])] - errors[4]
        np.testing.assert_allclose(delta.mean(), row['MAE_reduction'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.percentile(delta[draws].mean(1), [2.5, 97.5]), row['exploratory_bootstrap95'], rtol=0, atol=1e-12)
    split = m.load_json(m.ROOT / 'studies/20261027-window-ablation/new_person_split.json')
    exclude = m.load_json(m.ROOT / 'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json')
    original175 = set(d['person_id'][d['source_group'] == 'original175'])
    excluded = set(exclude['development_patient_ids']) - original175
    excluded |= set(exclude['original_test_patient_ids'])
    excluded |= {r['patient_id'] for r in split['people'] if r['reserved_role'] == 'final_evaluation_reserved'}
    assert len(excluded) == 37 + 38 + 223 and not set(d['person_id']) & excluded
    for name, expected in provenance['parent_artifacts'].items():
        assert m.digest(PARENT / name) == expected
    for name, expected in provenance['source_hashes'].items():
        assert m.digest(m.WORKSPACE / name) == expected
    result = dict(verified=True, independent_inner_refits=inner_count, independent_outer_refits=5,
        selected_components=selections, candidate_score_rows_checked=30, metric_rows_checked=nmetrics,
        paired_bootstrap_comparisons_checked=3, max_inner_prediction_difference=max_inner,
        max_outer_prediction_difference=max_outer, max_latent_prediction_difference=max_latent,
        old_candidate_reproduction=True, old_reference_predictions_unchanged=True,
        outer_and_inner_splits_unchanged=True, person_overlap=0, protected_ID_overlap=0,
        protected_numeric_data_read=False, parent_artifacts_preserved=len(provenance['parent_artifacts']),
        original_source_hashes_verified=len(provenance['source_hashes']), max_inner_component_iterations=max_iter,
        GPU_used=False, independent_final_evaluation=False)
    (HERE / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
