"""Matched 695-person CPU comparison. Nested selection receives training slices only."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
BRANCH = HERE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
sys.path.insert(0, str(BRANCH))
import cpu_head as m
import csv
import importlib.util
import json
import time
import warnings
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('previous_pca', HERE.parent / 'pca-ridge-study/run.py')
pca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pca)
LINK = m.ROOT / 'studies/20261027-gdi-linkage'
METHODS = ['old140_ridge', 'ridge', 'pca_ridge', 'pls']


def js(name, obj):
    path = (HERE / name).resolve()
    path.relative_to(HERE)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def csvfile(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dataset(protocol):
    old, heads, baseline, oldhash = m.source_data()
    names = ['development_features.npz', 'feature_preparation.json', 'feature_sources.json',
             'development_linkage.json', 'feature_pilot.json']
    receipt = m.load_json(LINK / 'study_receipt.json')
    receipt_hashes = {r['path']: r['sha256'] for r in receipt['artifacts']}
    fingerprints = {str((m.BUNDLE / k).relative_to(m.WORKSPACE)): v for k, v in oldhash.items()}
    for name in names:
        path = LINK / name
        actual = m.digest(path)
        assert actual == receipt_hashes[path.relative_to(m.ROOT).as_posix()], name
        fingerprints[path.relative_to(m.WORKSPACE).as_posix()] = actual
    prep = m.load_json(LINK / 'feature_preparation.json')
    assert prep['verified'] and prep['people'] == 520 and prep['records'] == 861
    assert prep['sha256'] == m.digest(LINK / 'development_features.npz')
    sources = m.load_json(LINK / 'feature_sources.json')
    assert prep['feature_sources_sha256'] == m.digest(LINK / 'feature_sources.json')
    weight = 'outputs/motionbert-smoke-20261015/assets/MB_lite.bin'
    assert sources['sha256'][weight] == m.load_json(m.BUNDLE / 'manifest.json')['MB_lite.bin']
    for path, expected in sources['sha256'].items():
        assert m.digest(m.WORKSPACE / path) == expected, path
        fingerprints[path] = expected
    with np.load(LINK / 'development_features.npz', allow_pickle=False) as z:
        new = {k: z[k].copy() for k in z.files}
    rows = m.load_json(LINK / 'development_linkage.json')['records']
    assert new['X'].shape == (861, 512) and new['Y'].shape == (861, 2)
    assert np.isfinite(new['X']).all() and np.isfinite(new['Y']).all()
    for i, row in enumerate(rows):
        assert row['eligible'] and row['reserved_role'] == 'development_candidate'
        assert str(new['patient_ids'][i]) == row['patient_id']
        assert str(new['video_ids'][i]) == row['video_id']
        assert int(new['sequence'][i]) == row['sequence']
        np.testing.assert_array_equal(new['Y'][i], row['GDI_LR'])
        path = LINK / f"development_features/row_{row['sequence']:04d}.npz"
        proof = m.load_json(path.with_suffix('.json'))
        assert m.digest(path) == proof['sha256']
        assert proof['input_sha256'] == row['sha256']
        assert proof['feature_sources_sha256'] == prep['feature_sources_sha256']
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(new['X'][i], z['feature'].astype(np.float64))
    split_path = m.ROOT / 'studies/20261027-window-ablation/new_person_split.json'
    exclude_path = m.ROOT / 'control/desktop-sessions/20261026T150518Z-87c0f4/exclusion_identities.json'
    split, exclude = m.load_json(split_path), m.load_json(exclude_path)
    for path in [split_path, exclude_path, LINK / 'study_receipt.json']:
        fingerprints[path.relative_to(m.WORKSPACE).as_posix()] = m.digest(path)
    candidate = {r['patient_id'] for r in split['people'] if r['reserved_role'] == 'development_candidate'}
    reserved = {r['patient_id'] for r in split['people'] if r['reserved_role'] == 'final_evaluation_reserved'}
    op, nperson = old['person_id'].astype(str), new['patient_ids'].astype(str)
    assert len(candidate) == 520 and len(reserved) == 223 and set(nperson) == candidate
    existing212 = set(exclude['development_patient_ids'])
    original38 = set(exclude['original_test_patient_ids'])
    protected37 = existing212 - set(op)
    assert len(protected37) == 37 and len(original38) == 38
    assert not (set(op) & candidate)
    all_people = set(op) | candidate
    assert not (all_people & (protected37 | original38 | reserved))
    new_folds = np.full(861, -1, dtype=np.int64)
    perm = np.random.default_rng(protocol['new_person_seed']).permutation(np.unique(nperson))
    for fold, people in enumerate(np.array_split(perm, 5)):
        assert len(people) == 104
        new_folds[np.isin(nperson, people)] = fold
    d = dict(X=np.concatenate([old['features'][:, 0].astype(np.float64), new['X']]),
             y=np.concatenate([old['y'], new['Y']]), person_id=np.concatenate([op, nperson]),
             video_id=np.concatenate([old['video_id'].astype(str), new['video_ids'].astype(str)]),
             fold=np.concatenate([old['fold'], new_folds]),
             source_group=np.array(['original175'] * 203 + ['additional520'] * 861))
    assert d['X'].shape == (1064, 512) and len(set(d['person_id'])) == 695
    assert len(set(d['video_id'])) == 1064
    for f in range(5):
        assert len(set(d['person_id'][d['fold'] == f])) == 139
        assert not set(d['person_id'][d['fold'] == f]) & set(d['person_id'][d['fold'] != f])
    np.savez_compressed(HERE / 'development695.npz', **d)
    js('data_provenance.json', {'source_hashes': fingerprints, 'additional_record_proofs_verified': 861,
       'protected_person_counts_excluded': [37, 38, 223], 'protected_numeric_data_read': False,
       'dataset_sha256': m.digest(HERE / 'development695.npz'), 'people': 695, 'records': 1064,
       'front_only': True, 'weight_sha256': sources['sha256'][weight]})
    return d, heads, baseline


def person_errors(y, pred, people):
    return np.array([np.abs(pred[people == p] - y[people == p]).mean() for p in np.unique(people)])


def linear_ridge(x, y, k=None):
    if k is None:
        scaler = StandardScaler().fit(x)
        design = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
        model = Ridge(alpha=1.0, solver='svd').fit(scaler.transform(x), y)
        return {**design, 'combined_coef': model.coef_, 'combined_intercept': model.intercept_}
    design = pca.pca_design(x, k)
    head = pca.fit_output(x, y, design, k, 1.0)
    coef = head['coef'] @ design['components']
    return {**design, 'combined_coef': coef,
            'combined_intercept': head['intercept'] - design['pca_mean'] @ coef.T}


def fit_pls(x, y, k, protocol):
    scaler = StandardScaler().fit(x)
    cfg = protocol['pls']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = PLSRegression(n_components=k, scale=cfg['scale'], max_iter=cfg['max_iter'], tol=cfg['tol'])
        model.fit(scaler.transform(x), y)
    if caught:
        js('fit_warning.json', {'k': k, 'messages': [str(w.message) for w in caught]})
        raise RuntimeError('PLS warning; inspect fit_warning.json before continuing')
    out = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, combined_coef=model.coef_,
               combined_intercept=model.intercept_ - model._x_mean @ model.coef_.T,
               x_mean=model._x_mean, y_mean=model._y_mean, x_weights=model.x_weights_,
               x_loadings=model.x_loadings_, y_loadings=model.y_loadings_,
               x_rotations=model.x_rotations_, n_iter=np.array(model.n_iter_))
    np.testing.assert_allclose(predict(x, out), model.predict(scaler.transform(x)), rtol=0, atol=1e-9)
    return out


def predict(x, model):
    return ((x - model['scaler_mean']) / model['scaler_scale']) @ model['combined_coef'].T + model['combined_intercept']


def choose(x, y, people, global_indices, outer_fold, protocol):
    """Outer held data are deliberately absent from this function's arguments."""
    assert len(set(people)) == 556
    perm = np.random.default_rng(protocol['inner_split_seed_base'] + outer_fold).permutation(np.unique(people))
    assignment = np.full(len(x), -1, dtype=int)
    for inner_fold, group in enumerate(np.array_split(perm, 4)):
        assignment[np.isin(people, group)] = inner_fold
    candidates = {'pca_ridge': protocol['pca_components'], 'pls': protocol['pls_components']}
    oof = {method: np.full((4, len(x), 2), np.nan) for method in candidates}
    saved = dict(outer_train_indices=global_indices, inner_assignment=assignment)
    audit, details = [], []
    for fold in range(4):
        tr, va = assignment != fold, assignment == fold
        assert len(set(people[tr])) == 417 and len(set(people[va])) == 139
        assert not set(people[tr]) & set(people[va])
        prefix = f'inner{fold}__'
        saved[prefix + 'train_indices'], saved[prefix + 'validation_indices'] = global_indices[tr], global_indices[va]
        design = pca.pca_design(x[tr], 64)
        for method, counts in candidates.items():
            for ci, k in enumerate(counts):
                if method == 'pca_ridge':
                    head = pca.fit_output(x[tr], y[tr], design, k, 1.0)
                    coef = head['coef'] @ design['components'][:k]
                    model = {**{v: design[v] for v in ['scaler_mean', 'scaler_scale']},
                             'combined_coef': coef, 'combined_intercept': head['intercept'] - design['pca_mean'] @ coef.T}
                else:
                    model = fit_pls(x[tr], y[tr], k, protocol)
                pred = predict(x[va], model)
                oof[method][ci, va] = pred
                for key, value in model.items():
                    saved[prefix + f'{method}_k{k}__' + key] = value
                details.append(dict(outer_fold=outer_fold, inner_fold=fold, method=method, components=k,
                    train_people=417, validation_people=139, validation_records=int(va.sum()),
                    person_MAE=float(person_errors(y[va], pred, people[va]).mean())))
        audit.append(dict(outer_fold=outer_fold, inner_fold=fold, train_people=417,
                          validation_people=139, train_records=int(tr.sum()), validation_records=int(va.sum())))
    selections, scores = {}, []
    for method, counts in candidates.items():
        assert np.isfinite(oof[method]).all()
        errors = [float(person_errors(y, pred, people).mean()) for pred in oof[method]]
        selections[method] = min(k for k, value in zip(counts, errors) if value <= min(errors) + 1e-12)
        scores.extend(dict(outer_fold=outer_fold, method=method, components=k, inner_person_MAE=value)
                      for k, value in zip(counts, errors))
        saved[method + '_inner_oof'] = oof[method]
    np.savez_compressed(HERE / f'models/fold{outer_fold}_inner.npz', **saved)
    return selections, scores, details, audit


def evaluate(d, pred, use=None, band=None):
    if use is None:
        use = np.ones(len(pred), dtype=bool)
    values = []
    for person in np.unique(d['person_id'][use]):
        mask = np.broadcast_to((use & (d['person_id'] == person))[:, None], pred.shape).copy()
        if band is not None:
            mask &= np.floor(d['y'] / 10) == band
        if not mask.any():
            continue
        truth, prediction = d['y'][mask], pred[mask]
        err = prediction - truth
        gap = abs(np.floor(prediction / 10) - np.floor(truth / 10))
        values.append([abs(err).mean(), err.mean(), (err ** 2).mean(), (gap == 0).mean(),
                       (gap <= 1).mean(), (gap >= 2).mean(), truth.mean(), prediction.mean()])
    a = np.array(values)
    result = dict(people=len(a), person_MAE=float(a[:, 0].mean()), bias=float(a[:, 1].mean()),
                  person_RMSE=float(np.sqrt(a[:, 2].mean())), exact_bin_rate=float(a[:, 3].mean()),
                  within_one_bin_rate=float(a[:, 4].mean()), two_or_more_bin_rate=float(a[:, 5].mean()),
                  truth_person_mean=float(a[:, 6].mean()), prediction_person_mean=float(a[:, 7].mean()),
                  truth_person_sd=float(a[:, 6].std()), prediction_person_sd=float(a[:, 7].std()))
    result['prediction_truth_sd_ratio'] = result['prediction_person_sd'] / result['truth_person_sd']
    return result


def main():
    protocol = m.load_json(HERE / 'protocol.json')
    assert protocol['methods'] == METHODS and protocol['ridge_alpha'] == 1.0
    assert not (HERE / 'summary.json').exists(), 'preserve completed experiment; no overwrite'
    (HERE / 'models').mkdir(exist_ok=True)
    d, old_heads, old_oof = dataset(protocol)
    js('execution_state.json', {'status': 'fitting', 'people': 695, 'records': 1064})
    cube = np.empty((4, 5, 1064, 2))
    selections, scores, details, audits = [], [], [], []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr, te = np.flatnonzero(d['fold'] != fold), np.flatnonzero(d['fold'] == fold)
            selected, s, det, audit = choose(d['X'][tr], d['y'][tr], d['person_id'][tr], tr, fold, protocol)
            scores.extend(s); details.extend(det); audits.extend(audit)
            prefix = f'fold{fold}__'
            old_model = {key: old_heads[prefix + key] for key in ['scaler_mean', 'scaler_scale']}
            old_model.update(combined_coef=old_heads[prefix + 'coef'], combined_intercept=old_heads[prefix + 'intercept'])
            cube[0, fold] = predict(d['X'], old_model)
            for mi, method in enumerate(METHODS[1:], 1):
                if method == 'pls':
                    model = fit_pls(d['X'][tr], d['y'][tr], selected['pls'], protocol)
                else:
                    model = linear_ridge(d['X'][tr], d['y'][tr], selected.get(method))
                cube[mi, fold] = predict(d['X'], model)
                np.savez_compressed(HERE / f'models/fold{fold}_{method}.npz', **model,
                    train_indices=tr, evaluation_indices=te, selected_k=np.array(selected.get(method, 512)))
            selections.append(dict(outer_fold=fold, pca_components=selected['pca_ridge'],
                pls_components=selected['pls'], train_people=556, evaluation_people=139,
                train_records=len(tr), evaluation_records=len(te)))
            print(json.dumps({'outer_fold_complete': fold, 'selected': selected,
                              'elapsed_seconds': round(time.perf_counter() - tick, 1)}), flush=True)
            js('execution_state.json', {'status': 'fitting', 'completed_outer_folds': fold + 1, 'selected': selections})
    assert np.isfinite(cube).all()
    oof = cube[:, d['fold'], np.arange(1064)]
    np.testing.assert_allclose(oof[0, :203], old_oof[:, 0], rtol=0, atol=1e-10)
    np.savez_compressed(HERE / 'fold_predictions.npz', predictions=cube, methods=np.array(METHODS))
    np.savez_compressed(HERE / 'oof_predictions.npz', predictions=oof, methods=np.array(METHODS),
                        person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
    metrics, bands, folds = [], [], []
    for mi, method in enumerate(METHODS):
        pred = oof[mi]
        for group in ['all695', 'original175', 'additional520']:
            use = np.ones(1064, dtype=bool) if group == 'all695' else d['source_group'] == group
            metrics.append(dict(method=method, group=group, records=int(use.sum()),
                raw_prediction_min=float(pred[use].min()), raw_prediction_max=float(pred[use].max()),
                **evaluate(d, pred, use)))
        for band in np.unique(np.floor(d['y'] / 10).astype(int)):
            bands.append(dict(method=method, actual_bin_low=int(10 * band), actual_bin_high_exclusive=int(10 * (band + 1)),
                record_sides=int((np.floor(d['y'] / 10) == band).sum()), **evaluate(d, pred, band=band)))
        for fold in range(5):
            folds.append(dict(method=method, fold=fold, **evaluate(d, pred, use=d['fold'] == fold)))
    draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
    comparisons = []
    for base, other in [(0, 1), (1, 2), (1, 3), (2, 3)]:
        delta = person_errors(d['y'], oof[base], d['person_id']) - person_errors(d['y'], oof[other], d['person_id'])
        ci = np.percentile(delta[draws].mean(1), [2.5, 97.5])
        comparisons.append(dict(baseline=METHODS[base], candidate=METHODS[other],
            MAE_reduction=float(delta.mean()), bootstrap95=ci.tolist(), people_improved=int((delta > 0).sum()),
            people_worsened=int((delta < 0).sum())))
    csvfile('metrics.csv', metrics); csvfile('score_bands.csv', bands); csvfile('fold_metrics.csv', folds)
    csvfile('selected_dimensions.csv', selections); csvfile('inner_candidate_scores.csv', scores)
    csvfile('inner_fold_scores.csv', details); csvfile('split_audit.csv', audits)
    js('summary.json', {'protocol_sha256': m.digest(HERE / 'protocol.json'), 'people': 695, 'records': 1064,
        'metrics': metrics, 'score_bands': bands, 'selected_dimensions': selections, 'comparisons': comparisons,
        'fit_seconds': time.perf_counter() - tick, 'sklearn': sklearn.__version__, 'numpy': np.__version__,
        'truth_raw_range': [float(d['y'].min()), float(d['y'].max())],
        'fits': {'inner_PCA_Ridge': 80, 'outer_PCA_Ridge': 5, 'inner_PLS': 80, 'outer_PLS': 5, 'outer_Ridge': 5},
        'protected_numeric_data_read': False, 'GPU_used': False, 'independent_final_evaluation': False})
    js('oof.metadata.json', {'schema': 'gdi_development695_front_oof_v1',
        'shape': [4, 1064, 2], 'axes': ['method', 'record', 'side'], 'methods': METHODS,
        'sides': ['left_GDI', 'right_GDI'], 'windows': ['front124_only'],
        'dataset_sha256': m.digest(HERE / 'development695.npz'),
        'predictions_sha256': m.digest(HERE / 'oof_predictions.npz'),
        'protocol_sha256': m.digest(HERE / 'protocol.json'), 'selection_uses_outer_held_people': False,
        'note': 'New front-only 695-person contract; not the old 175-person front/back GPU prediction contract.'})
    js('execution_state.json', {'status': 'fit_complete_awaiting_independent_verification', 'selected': selections})
    print(json.dumps({'metrics': [r for r in metrics if r['group'] == 'all695'], 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
