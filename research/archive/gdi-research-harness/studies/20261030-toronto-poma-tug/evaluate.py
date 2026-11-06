"""Nested participant-disjoint prediction of the two separate clinical targets."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import time
import warnings

from prepare import HERE, ROOT, MB, sha, save
import sys
sys.path.insert(0, str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

TARGETS = ['POMA_Gait', 'TUG_seconds']
METHODS = ['mean', 'median', 'ridge', 'pls']


def predict(train_x, train_y, test_x, method, components=None):
    assert train_x.ndim == test_x.ndim == 2 and np.isfinite(train_x).all() and np.isfinite(test_x).all()
    scaler = StandardScaler().fit(train_x)
    np.testing.assert_allclose(scaler.mean_, train_x.mean(0), rtol=0, atol=1e-12)
    if np.ptp(train_y) == 0:
        return np.full(len(test_x), train_y[0]), {'constant_training_target': True, 'warnings': []}
    model = Ridge(alpha=1.0) if method == 'ridge' else PLSRegression(n_components=components, scale=False, max_iter=500, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(scaler.transform(train_x), train_y)
        result = model.predict(scaler.transform(test_x)).reshape(-1)
    assert np.isfinite(result).all()
    return result, {'constant_training_target': False, 'warnings': [str(x.message) for x in caught]}


def folds(people):
    result = []
    for outer in people:
        train = [int(p) for p in people if p != outer]
        inner = [{'evaluation': int(p), 'training': [x for x in train if x != p]} for p in train]
        assert int(outer) not in train and all(int(outer) not in x['training'] and int(outer) != x['evaluation'] for x in inner)
        result.append({'evaluation': int(outer), 'training': train, 'inner': inner})
    return result


def main():
    tick = time.monotonic()
    assert not (HERE/'results.json').exists(), 'Do not overwrite an evaluated experiment'
    protocol = json.loads((HERE/'protocol.json').read_text())
    receipt = json.loads((HERE/'feature_receipt.json').read_text())
    assert all(sha(HERE/name) == value for name, value in receipt['sha256'].items())
    with np.load(HERE/'features.npz', allow_pickle=False) as data:
        X, Y, people, counts = [data[k].copy() for k in ['X', 'Y', 'person', 'windows_per_person']]
    assert len(set(people)) == len(people) and X.shape == (len(people), 512) and Y.shape == (len(people), 2)
    split = folds(people)
    save('folds.json', {'unit': 'participant', 'outer': split, 'declared_before_head_fitting_utc': datetime.now(timezone.utc).isoformat()})
    predictions = {method: np.full(Y.shape, np.nan) for method in METHODS}
    selections, traces, warning_counts = [], [], Counter()
    components = protocol['models']['PLS']['components']
    with threadpool_limits(limits=2):
        for outer_i, outer in enumerate(split):
            train = np.asarray([i for i, p in enumerate(people) if p in outer['training']])
            test = np.asarray([outer_i])
            assert people[outer_i] == outer['evaluation'] and not set(people[train]) & set(people[test])
            for target_i, target in enumerate(TARGETS):
                predictions['mean'][outer_i, target_i] = Y[train, target_i].mean()
                predictions['median'][outer_i, target_i] = np.median(Y[train, target_i])
                ridge, info = predict(X[train], Y[train, target_i], X[test], 'ridge')
                predictions['ridge'][outer_i, target_i] = ridge[0]
                warning_counts.update(info['warnings'])
                candidate_results = []
                for component in components:
                    inner_values, inner_truth = [], []
                    for inner in outer['inner']:
                        inner_train = np.asarray([i for i, p in enumerate(people) if p in inner['training']])
                        inner_test = np.where(people == inner['evaluation'])[0]
                        assert not set(people[inner_train]) & (set(people[inner_test]) | {outer['evaluation']})
                        assert len(inner_train) == len(people)-2 and len(inner_test) == 1
                        assert component <= min(len(inner_train)-1, X.shape[1])
                        value, info = predict(X[inner_train], Y[inner_train, target_i], X[inner_test], 'pls', component)
                        warning_counts.update(info['warnings'])
                        inner_values.append(float(value[0]))
                        inner_truth.append(float(Y[inner_test[0], target_i]))
                        traces.append({'outer': outer['evaluation'], 'target': target, 'components': component,
                                       'stage': 'inner', 'train_people': people[inner_train].tolist(),
                                       'evaluation_people': people[inner_test].tolist()})
                    candidate_results.append({'components': component, 'inner_MAE': float(mean_absolute_error(inner_truth, inner_values)),
                                              'inner_truth': inner_truth, 'inner_predictions': inner_values})
                best_mae = min(x['inner_MAE'] for x in candidate_results)
                chosen = min(x['components'] for x in candidate_results if x['inner_MAE'] <= best_mae+1e-12)
                value, info = predict(X[train], Y[train, target_i], X[test], 'pls', chosen)
                warning_counts.update(info['warnings'])
                predictions['pls'][outer_i, target_i] = value[0]
                selections.append({'outer_person': outer['evaluation'], 'target': target, 'chosen_components': chosen, 'candidates': candidate_results})
                for method in ['ridge', 'pls']:
                    traces.append({'outer': outer['evaluation'], 'target': target, 'method': method, 'stage': 'outer',
                                   'train_people': people[train].tolist(), 'evaluation_people': people[test].tolist()})
            print(json.dumps({'evaluated_people': outer_i+1, 'total': len(people)}), flush=True)
    assert all(np.isfinite(value).all() for value in predictions.values())
    save('inner_selection.json', selections)
    save('fit_trace.json', traces)
    np.savez_compressed(HERE/'predictions.npz', truth=Y, person=people, **predictions)
    with (HERE/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['person', 'windows', 'target', 'method', 'actual', 'predicted', 'absolute_error'])
        writer.writeheader()
        for i, p in enumerate(people):
            for j, target in enumerate(TARGETS):
                for method in METHODS:
                    value = float(predictions[method][i, j])
                    writer.writerow({'person': int(p), 'windows': int(counts[i]), 'target': target, 'method': method,
                                     'actual': float(Y[i, j]), 'predicted': value, 'absolute_error': abs(value-Y[i, j])})
    rng = np.random.default_rng(protocol['uncertainty']['seed'])
    bootstrap = rng.integers(0, len(people), size=(protocol['uncertainty']['paired_person_bootstrap_draws'], len(people)))
    metrics = {}
    for j, target in enumerate(TARGETS):
        metrics[target] = {}
        median_error = np.abs(predictions['median'][:, j]-Y[:, j])
        for method in METHODS:
            pred, truth = predictions[method][:, j], Y[:, j]
            error = np.abs(pred-truth)
            difference = error-median_error
            rho = float(spearmanr(pred, truth).statistic) if np.ptp(pred) > 0 and np.ptp(truth) > 0 else None
            metrics[target][method] = {'MAE': float(error.mean()), 'RMSE': float(np.sqrt(mean_squared_error(truth, pred))),
                                      'R2': float(r2_score(truth, pred)), 'Spearman': rho,
                                      'prediction_range': [float(pred.min()), float(pred.max())],
                                      'MAE_difference_from_median': float(difference.mean()),
                                      'difference_bootstrap_95_percentiles': np.quantile(difference[bootstrap].mean(1), [.025, .975]).tolist(),
                                      'people_better_equal_worse_than_median': [int((difference < -1e-12).sum()), int((np.abs(difference) <= 1e-12).sum()), int((difference > 1e-12).sum())]}
    result = {'scope': 'Exploratory nested leave-one-person-out Toronto prediction; no independent clinical validation.',
              'people': len(people), 'windows': int(counts.sum()), 'training_people_per_outer': len(people)-1,
              'inner_training_people': len(people)-2, 'metrics': metrics,
              'target_distributions': {target: dict(Counter(map(str, Y[:, j]))) for j, target in enumerate(TARGETS)},
              'PLS_selected_components': {target: dict(Counter(x['chosen_components'] for x in selections if x['target'] == target)) for target in TARGETS},
              'fit_warnings': dict(warning_counts), 'all_predictions_finite': True,
              'scaler_and_selection_training_only': True, 'protected_GDI_numeric_data_used': False,
              'encoder_finetuned': False, 'sklearn': sklearn.__version__, 'elapsed_seconds': time.monotonic()-tick,
              'finished_utc': datetime.now(timezone.utc).isoformat(),
              'sha256': {name: sha(HERE/name) for name in ['protocol.json', 'features.npz', 'folds.json', 'inner_selection.json', 'fit_trace.json', 'predictions.npz', 'predictions.csv', 'prepare.py', 'evaluate.py']}}
    save('results.json', result)
    print(json.dumps({'people': len(people), 'windows': int(counts.sum()), 'metrics': metrics, 'fit_warnings': dict(warning_counts)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
