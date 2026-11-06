"""Inner-person selection of PCA dimension; no encoder inference or outer-label tuning."""
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'train_eval_diagnostic'))
import diagnose as diag
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
import csv
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def write_json(path, obj):
    path = Path(path).resolve()
    path.relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pca_design(x, max_k):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    center = z.mean(0)
    _, singular, vt = np.linalg.svd(z - center, full_matrices=False)
    if max_k > len(singular):
        raise ValueError('component count exceeds training rank bound')
    return {'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_, 'pca_mean': center,
            'components': vt[:max_k], 'singular_values': singular}


def fit_output(x, y, design, k, alpha):
    z = (x - design['scaler_mean']) / design['scaler_scale']
    t = (z - design['pca_mean']) @ design['components'][:k].T
    tmean, ymean = t.mean(0), y.mean(0)
    tc, yc = t - tmean, y - ymean
    coef = np.linalg.solve(tc.T @ tc + alpha * np.eye(k), tc.T @ yc).T
    return {'coef': coef, 'intercept': ymean - tmean @ coef.T}


def predict(x, design, output, k):
    z = (x - design['scaler_mean']) / design['scaler_scale']
    return ((z - design['pca_mean']) @ design['components'][:k].T) @ output['coef'].T + output['intercept']


def person_mae(y, pred, people):
    return float(np.mean([abs(pred[people == pid] - y[people == pid]).mean() for pid in np.unique(people)]))


def choose_k(x, y, people, global_indices, outer_fold, protocol):
    """Accept outer-training arrays ONLY. No outer held features/labels are arguments."""
    if len(np.unique(people)) != 140 or len(x) != len(y) or len(x) != len(people):
        raise ValueError('unexpected outer training slice')
    candidates = protocol['candidate_components']
    shuffled = np.random.default_rng(protocol['inner_split_seed_base'] + outer_fold).permutation(np.unique(people))
    split_people = np.array_split(shuffled, protocol['inner_folds'])
    assignment = np.full(len(people), -1, dtype=int)
    for inner_fold, group in enumerate(split_people):
        assignment[np.isin(people, group)] = inner_fold
    if (assignment < 0).any():
        raise ValueError('uncovered inner records')
    oof = np.full((len(candidates), len(y), 2), np.nan)
    saved = {'outer_train_indices': global_indices, 'inner_assignment': assignment}
    audit, inner_rows = [], []
    for inner_fold in range(4):
        tr, va = assignment != inner_fold, assignment == inner_fold
        train_people, validation_people = set(people[tr]), set(people[va])
        if train_people & validation_people or len(train_people) != 105 or len(validation_people) != 35:
            raise ValueError('inner person leakage/count mismatch')
        design = pca_design(x[tr], max(candidates))
        prefix = f'inner{inner_fold}__'
        saved[prefix + 'train_indices'] = global_indices[tr]
        saved[prefix + 'validation_indices'] = global_indices[va]
        for key, value in design.items():
            saved[prefix + key] = value
        for ci, k in enumerate(candidates):
            head = fit_output(x[tr], y[tr], design, k, protocol['ridge_alpha'])
            pred = predict(x[va], design, head, k)
            oof[ci, va] = pred
            saved[prefix + f'k{k}_coef'] = head['coef']
            saved[prefix + f'k{k}_intercept'] = head['intercept']
            inner_rows.append({'outer_fold': outer_fold, 'inner_fold': inner_fold, 'components': k,
                'validation_people': 35, 'validation_records': int(va.sum()),
                'front_person_MAE': person_mae(y[va], pred, people[va])})
        audit.append({'outer_fold': outer_fold, 'inner_fold': inner_fold, 'train_people': 105,
                      'validation_people': 35, 'train_records': int(tr.sum()), 'validation_records': int(va.sum())})
    if not np.isfinite(oof).all():
        raise ValueError('incomplete inner prediction coverage')
    scores = [{'outer_fold': outer_fold, 'components': k, 'inner_front_person_MAE': person_mae(y, oof[ci], people)}
              for ci, k in enumerate(candidates)]
    best = min(row['inner_front_person_MAE'] for row in scores)
    selected = min(row['components'] for row in scores if row['inner_front_person_MAE'] <= best + 1e-12)
    saved['inner_oof_predictions'] = oof
    return selected, scores, inner_rows, audit, saved


def main():
    protocol = diag.m.load_json(HERE / 'protocol.json')
    assert protocol['candidate_components'] == [8, 16, 32, 64]
    assert protocol['ridge_alpha'] == 1 and protocol['whiten'] is False
    d, old_heads, baseline, sources = diag.m.source_data()
    root_models = HERE / 'models'
    root_models.mkdir(exist_ok=True)
    cube = np.empty((5, 203, 2, 2))
    ridge = np.empty_like(cube)
    selections, scores, inner_rows, audits = [], [], [], []
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr, te = np.flatnonzero(d['fold'] != fold), np.flatnonzero(d['fold'] == fold)
            selected, values, inside, audit, saved = choose_k(d['features'][tr, 0].astype(np.float64),
                d['y'][tr], d['person_id'][tr], tr, fold, protocol)
            scores.extend(values); inner_rows.extend(inside); audits.extend(audit)
            np.savez_compressed(root_models / f'fold{fold}_inner.npz', **saved)
            # Refit every transform after selection, still excluding the outer held people.
            design = pca_design(d['features'][tr, 0].astype(np.float64), selected)
            head = fit_output(d['features'][tr, 0].astype(np.float64), d['y'][tr], design, selected, 1.0)
            cube[fold] = predict(d['features'].astype(np.float64), design, head, selected)
            combined_coef = head['coef'] @ design['components'][:selected]
            combined_intercept = head['intercept'] - design['pca_mean'] @ combined_coef.T
            ratio = float(np.square(design['singular_values'][:selected]).sum() / np.square(design['singular_values']).sum())
            np.savez_compressed(root_models / f'fold{fold}_selected.npz', **design, ridge_coef=head['coef'],
                ridge_intercept=head['intercept'], combined_coef=combined_coef, combined_intercept=combined_intercept,
                selected_k=np.array(selected), train_indices=tr, evaluation_indices=te,
                retained_feature_variance_ratio=np.array(ratio))
            selections.append({'outer_fold': fold, 'selected_components': selected, 'train_people': 140,
                'evaluation_people': 35, 'train_records': len(tr), 'evaluation_records': len(te),
                'selected_inner_front_MAE': next(x['inner_front_person_MAE'] for x in values if x['components'] == selected),
                'retained_standardized_feature_variance_ratio': ratio})
            prefix = f'fold{fold}__'
            z = (d['features'].astype(np.float64) - old_heads[prefix + 'scaler_mean']) / old_heads[prefix + 'scaler_scale']
            ridge[fold] = z @ old_heads[prefix + 'coef'].T + old_heads[prefix + 'intercept']
            print(f'Outer fold {fold}: selected k={selected} using training people only', flush=True)
    if not np.isfinite(cube).all():
        raise ValueError('nonfinite outer predictions')
    np.savez_compressed(HERE / 'fold_predictions.npz', pca_ridge=cube, ridge=ridge,
                        person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
    oof = cube[d['fold'], np.arange(203)]
    np.savez_compressed(HERE / 'oof_predictions.npz', predictions=oof, baseline=baseline,
                        person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
    fit_seconds = time.perf_counter() - started
    metrics, bands, fold_metrics = [], [], []
    for name, values in [('ridge', ridge), ('pca_ridge_nested', cube)]:
        result = diag.analyze_cube(d, values, name)
        metrics.extend(result[0]); bands.extend(result[1]); fold_metrics.extend(result[2])
    old_metrics, old_people = diag.m.metrics(d, baseline)
    new_metrics, new_people = diag.m.metrics(d, oof)
    delta = old_people - new_people
    draws = np.random.default_rng(20261029).integers(0, 175, (10000, 175))
    intervals = np.percentile(delta[draws].mean(1), [2.5, 97.5], axis=0).T.tolist()
    write_csv(HERE / 'metrics.csv', metrics)
    write_csv(HERE / 'score_bands.csv', bands)
    write_csv(HERE / 'fold_metrics.csv', fold_metrics)
    write_csv(HERE / 'selected_dimensions.csv', selections)
    write_csv(HERE / 'inner_candidate_scores.csv', scores)
    write_csv(HERE / 'inner_fold_scores.csv', inner_rows)
    write_csv(HERE / 'split_audit.csv', audits)
    write_json(HERE / 'summary.json', {'protocol': protocol, 'protocol_sha256': diag.m.digest(HERE / 'protocol.json'),
        'source_hashes': sources, 'selected_dimensions': selections, 'inner_candidate_scores': scores,
        'metrics': metrics, 'score_bands': bands, 'fold_metrics': fold_metrics,
        'fit_and_export_seconds': fit_seconds, 'inner_ridge_fits': 80, 'outer_ridge_fits': 5, 'PCA_fits': 25,
        'baseline_OOF': old_metrics, 'nested_PCA_OOF': new_metrics,
        'paired_reduction_front_back_gap': delta.mean(0).tolist(),
        'exploratory_person_bootstrap95_front_back_gap': intervals,
        'people_improved_worsened_front': [int((delta[:, 0] > 0).sum()), int((delta[:, 0] < 0).sum())],
        'selection_uses_outer_held_people': False, 'independent_final_evaluation': False,
        'encoder_reinference': False, 'GPU_used': False})
    # The same figure and diagnostic formats can later consume fold-specific GPU features.
    diag.plots(d, {'ridge': ridge, 'pca_ridge_nested': cube}, metrics, bands, HERE)
    print(json.dumps({'selections': selections, 'baseline': old_metrics, 'nested_PCA': new_metrics,
                      'paired_reduction_front_back_gap': delta.mean(0).tolist()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
