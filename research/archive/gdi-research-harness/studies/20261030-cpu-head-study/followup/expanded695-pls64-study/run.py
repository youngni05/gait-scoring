"""Extend PLS candidates while preserving every parent data and split assignment."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'expanded695-pls-study'
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
import importlib.util
import json
import csv
import time
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

spec = importlib.util.spec_from_file_location('parent_run', PARENT / 'run.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
METHODS = ['old140_ridge', 'ridge', 'pca_ridge', 'pls16', 'pls64']


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def js(name, obj):
    path = (HERE / name).resolve()
    path.relative_to(HERE)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def preserved_parent():
    manifest = m.load_json(PARENT / 'artifact_manifest.json')
    for name, expected in manifest.items():
        assert m.digest(PARENT / name) == expected, name
    assert m.load_json(PARENT / 'verification.json')['verified']
    sources = m.load_json(PARENT / 'data_provenance.json')['source_hashes']
    for name, expected in sources.items():
        assert m.digest(m.WORKSPACE / name) == expected, name
    return manifest, sources


def predict(x, model):
    return ((x - model['scaler_mean']) / model['scaler_scale']) @ model['coef'].T + model['intercept']


def fit(x, y, k, cfg):
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = PLSRegression(n_components=k, **cfg).fit(scaler.transform(x), y)
    if caught:
        js('fit_warning.json', {'components': k, 'messages': [str(w.message) for w in caught]})
        raise RuntimeError('PLS warning, inspect fit_warning.json; no candidate dropped')
    result = dict(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, coef=model.coef_,
        intercept=model.intercept_ - model._x_mean @ model.coef_.T,
        x_mean=model._x_mean, y_mean=model._y_mean, x_rotations=model.x_rotations_,
        y_loadings=model.y_loadings_, n_iter=np.asarray(model.n_iter_))
    np.testing.assert_allclose(predict(x, result), model.predict(scaler.transform(x)), rtol=0, atol=1e-9)
    return result


def choose(x, y, people, indices, assignment, previous_oof, protocol, outer):
    """No outer evaluation features or labels are passed to this function."""
    assert len(set(people)) == 556
    counts = protocol['pls_components']
    oof = np.full((len(counts), len(x), 2), np.nan)
    saved = dict(outer_train_indices=indices, inner_assignment=assignment)
    details, audits = [], []
    for inner in range(4):
        tr, va = assignment != inner, assignment == inner
        assert len(set(people[tr])) == 417 and len(set(people[va])) == 139
        assert not set(people[tr]) & set(people[va])
        saved[f'inner{inner}__train_indices'] = indices[tr]
        saved[f'inner{inner}__validation_indices'] = indices[va]
        for ci, k in enumerate(counts):
            model = fit(x[tr], y[tr], k, protocol['pls'])
            pred = predict(x[va], model)
            oof[ci, va] = pred
            for key, value in model.items():
                saved[f'inner{inner}__k{k}__{key}'] = value
            details.append(dict(outer_fold=outer, inner_fold=inner, components=k,
                person_MAE=float(parent.person_errors(y[va], pred, people[va]).mean()),
                max_component_iterations=int(model['n_iter'].max())))
        audits.append(dict(outer_fold=outer, inner_fold=inner, train_people=417, validation_people=139,
                           train_records=int(tr.sum()), validation_records=int(va.sum())))
    assert np.isfinite(oof).all()
    # Refit the old candidates too, proving that the changed candidate list is the only change.
    np.testing.assert_allclose(oof[:4], previous_oof, rtol=0, atol=1e-9)
    errors = [float(parent.person_errors(y, pred, people).mean()) for pred in oof]
    selected = min(k for k, err in zip(counts, errors) if err <= min(errors) + 1e-12)
    saved['inner_oof_predictions'] = oof
    np.savez_compressed(HERE / f'models/fold{outer}_inner.npz', **saved)
    scores = [dict(outer_fold=outer, components=k, inner_person_MAE=err) for k, err in zip(counts, errors)]
    return selected, scores, details, audits, float(abs(oof[:4] - previous_oof).max())


def main():
    assert not (HERE / 'summary.json').exists(), 'preserve completed results'
    protocol = m.load_json(HERE / 'protocol.json')
    assert protocol['pls_components'] == [2, 4, 8, 16, 32, 64]
    # Parent has an explanatory outputs field; numerical parameters must match exactly.
    previous_config = m.load_json(PARENT / 'protocol.json')['pls']
    assert all(protocol['pls'][key] == previous_config[key] for key in ['scale', 'max_iter', 'tol'])
    manifest, sources = preserved_parent()
    js('provenance.json', {'parent_artifact_manifest_sha256': m.digest(PARENT / 'artifact_manifest.json'),
        'parent_artifacts': manifest, 'source_hashes': sources,
        'dataset_sha256': m.digest(PARENT / 'development695.npz'),
        'protocol_sha256': m.digest(HERE / 'protocol.json')})
    d = load_npz(PARENT / 'development695.npz')
    previous = load_npz(PARENT / 'oof_predictions.npz')
    assert d['X'].shape == (1064, 512) and len(set(d['person_id'])) == 695
    (HERE / 'models').mkdir(exist_ok=True)
    js('execution_state.json', {'status': 'fitting', 'candidates': protocol['pls_components']})
    cube = np.empty((5, 1064, 2))
    selections, candidate_scores, inside, audits, matches = [], [], [], [], []
    tick = time.perf_counter()
    with threadpool_limits(limits=2):
        for outer in range(5):
            tr, te = np.flatnonzero(d['fold'] != outer), np.flatnonzero(d['fold'] == outer)
            parent_inner = load_npz(PARENT / f'models/fold{outer}_inner.npz')
            np.testing.assert_array_equal(tr, parent_inner['outer_train_indices'])
            assert len(set(d['person_id'][te])) == 139
            assert not set(d['person_id'][tr]) & set(d['person_id'][te])
            selected, scores, det, audit, difference = choose(d['X'][tr], d['y'][tr], d['person_id'][tr],
                tr, parent_inner['inner_assignment'], parent_inner['pls_inner_oof'], protocol, outer)
            model = fit(d['X'][tr], d['y'][tr], selected, protocol['pls'])
            cube[outer] = predict(d['X'], model)
            np.savez_compressed(HERE / f'models/fold{outer}_selected.npz', **model, selected_k=np.array(selected),
                                train_indices=tr, evaluation_indices=te)
            selections.append(dict(outer_fold=outer, selected_components=selected,
                previous_components=16, selected_inner_MAE=next(r['inner_person_MAE'] for r in scores if r['components'] == selected),
                train_people=556, evaluation_people=139, train_records=len(tr), evaluation_records=len(te)))
            candidate_scores.extend(scores); inside.extend(det); audits.extend(audit); matches.append(difference)
            print(json.dumps({'outer_fold': outer, 'selected': selected, 'scores': scores,
                              'seconds': round(time.perf_counter() - tick, 2)}), flush=True)
    assert np.isfinite(cube).all()
    new_oof = cube[d['fold'], np.arange(1064)]
    predictions = np.concatenate([previous['predictions'], new_oof[None]])
    np.savez_compressed(HERE / 'fold_predictions.npz', predictions=cube)
    np.savez_compressed(HERE / 'oof_predictions.npz', predictions=predictions, methods=np.array(METHODS),
                        person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
    metrics, bands, fold_metrics = [], [], []
    for method, pred in zip(METHODS, predictions):
        for group in ['all695', 'original175', 'additional520']:
            use = np.ones(1064, dtype=bool) if group == 'all695' else d['source_group'] == group
            metrics.append(dict(method=method, group=group, records=int(use.sum()),
                raw_prediction_min=float(pred[use].min()), raw_prediction_max=float(pred[use].max()),
                **parent.evaluate(d, pred, use)))
        for band in np.unique(np.floor(d['y'] / 10).astype(int)):
            bands.append(dict(method=method, actual_bin_low=int(10 * band), actual_bin_high_exclusive=int(10 * (band + 1)),
                record_sides=int((np.floor(d['y'] / 10) == band).sum()), **parent.evaluate(d, pred, band=band)))
        for outer in range(5):
            fold_metrics.append(dict(method=method, fold=outer, **parent.evaluate(d, pred, d['fold'] == outer)))
    draws = np.random.default_rng(20261029).integers(0, 695, (10000, 695))
    comparisons = []
    for base in ['pls16', 'ridge', 'pca_ridge']:
        delta = parent.person_errors(d['y'], predictions[METHODS.index(base)], d['person_id']) - parent.person_errors(d['y'], new_oof, d['person_id'])
        comparisons.append(dict(baseline=base, candidate='pls64', MAE_reduction=float(delta.mean()),
            exploratory_bootstrap95=np.percentile(delta[draws].mean(1), [2.5, 97.5]).tolist(),
            people_improved=int((delta > 0).sum()), people_worsened=int((delta < 0).sum())))
    for name, rows in [('metrics.csv', metrics), ('score_bands.csv', bands), ('fold_metrics.csv', fold_metrics),
        ('selected_dimensions.csv', selections), ('inner_candidate_scores.csv', candidate_scores),
        ('inner_fold_scores.csv', inside), ('split_audit.csv', audits)]:
        table(name, rows)
    js('summary.json', {'protocol_sha256': m.digest(HERE / 'protocol.json'), 'people': 695, 'records': 1064,
        'selected_dimensions': selections, 'metrics': metrics, 'score_bands': bands,
        'comparisons': comparisons, 'old_inner_candidate_max_difference': max(matches),
        'fit_seconds': time.perf_counter() - tick, 'inner_PLS_fits': 120, 'outer_PLS_fits': 5,
        'GPU_used': False, 'protected_numeric_data_used': False, 'independent_final_evaluation': False})
    js('oof.metadata.json', {'schema': 'gdi_development695_front_oof_v1', 'shape': [5, 1064, 2],
        'axes': ['method', 'record', 'side'], 'methods': METHODS, 'windows': ['front124_only'],
        'dataset_path': '../expanded695-pls-study/development695.npz',
        'dataset_sha256': m.digest(PARENT / 'development695.npz'),
        'predictions_sha256': m.digest(HERE / 'oof_predictions.npz'),
        'protocol_sha256': m.digest(HERE / 'protocol.json'), 'selection_uses_outer_held_people': False})
    js('execution_state.json', {'status': 'fit_complete_awaiting_verification', 'selected': selections})
    print(json.dumps({'metrics': [r for r in metrics if r['group'] == 'all695'], 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
