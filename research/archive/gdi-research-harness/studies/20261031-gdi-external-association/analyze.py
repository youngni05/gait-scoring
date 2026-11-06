"""Person-level association only; no score conversion or model selection."""
from common import *
import itertools
from scipy.stats import rankdata


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide((a * b).sum(axis=-1), denominator,
                     out=np.full_like(denominator, np.nan), where=denominator > 1e-14)


def rho(x, y):
    value = float(correlation(rankdata(x), rankdata(y)))
    return value if np.isfinite(value) else None


def cohorts():
    d = arrays(HERE / 'predictions.npz')
    targets = clinical()
    out = []
    for name, names, predictions, owner in [
        ('common6', d['models'], d['primary'], d['ids']),
        ('h36m12', d['h36m_models'], d['h36m'], d['h36m_owner']),
        ('cnn10', d['cnn_models'], d['cnn'], d['cnn_ids'])]:
        ids = np.array(sorted(set(owner), key=int))
        # Average records before source folds; each person has equal weight.
        per_fold = np.stack([predictions[:, :, owner == pid].mean(2) for pid in ids], axis=2)
        assert per_fold.shape == (len(names), 5, len(ids), 2)
        y = np.array([targets[str(pid)] for pid in ids])
        out.append((name, names, ids, per_fold, y))
    return out


def holm(values):
    valid = sorted((v, i) for i, v in enumerate(values) if v is not None)
    result, previous = [None] * len(values), 0.
    for j, (value, i) in enumerate(valid):
        previous = max(previous, min(1., value * (len(valid) - j)))
        result[i] = previous
    return result


def main():
    assert read(HERE / 'inference-verification.json')['passed']
    summaries, people_rows, loo_rows, fold_rows = [], [], [], []
    all_cohorts = cohorts()
    for cohort, names, ids, per_fold, y in all_cohorts:
        n = len(ids)
        predictions = per_fold.mean(1)
        scores = predictions.mean(2)
        rng = np.random.default_rng(20261031)
        draws = rng.integers(0, n, size=(10000, n))
        exact = cohort == 'common6'
        permutations = np.array(list(itertools.permutations(range(n)))) if exact else np.stack([rng.permutation(n) for _ in range(9999)])
        rows = []
        for i, name in enumerate(names):
            for j, target in enumerate(['TUG', 'POMA_Gait']):
                x, target_values = scores[i], y[:, j]
                value = rho(x, target_values)
                p_value, interval, valid_draws = None, None, 0
                if value is not None:
                    permuted = correlation(rankdata(x), rankdata(target_values)[permutations])
                    count = int((abs(permuted) >= abs(value) - 1e-12).sum())
                    p_value = count / len(permuted) if exact else (count + 1) / (len(permuted) + 1)
                    boot = correlation(rankdata(x[draws], axis=1), rankdata(target_values[draws], axis=1))
                    good = boot[np.isfinite(boot)]
                    valid_draws = len(good)
                    if valid_draws:
                        interval = np.quantile(good, [.025, .975]).tolist()
                leave = []
                for k, pid in enumerate(ids):
                    keep = np.arange(n) != k
                    r = rho(x[keep], target_values[keep])
                    loo_rows.append({'cohort': cohort, 'model': str(name), 'target': target,
                                     'removed_person': str(pid), 'rho': r})
                    if r is not None:
                        leave.append(r)
                slow = y[:, 0] == 25.
                without_slow = rho(x[~slow], target_values[~slow]) if slow.any() else None
                pearson = float(correlation(x, target_values))
                row = {'cohort': cohort, 'model': str(name), 'target': target, 'people': n,
                    'spearman_rho': value, 'pearson_r': pearson if np.isfinite(pearson) else None,
                    'bootstrap95_low': interval[0] if interval else None,
                    'bootstrap95_high': interval[1] if interval else None,
                    'bootstrap_valid': valid_draws, 'bootstrap_degenerate': 10000 - valid_draws,
                    'permutation_p_two_sided': p_value, 'permutation_count': len(permutations),
                    'exact_permutation': exact, 'loo_rho_min': min(leave) if leave else None,
                    'loo_rho_max': max(leave) if leave else None, 'loo_valid': len(leave),
                    'rho_without_TUG25_person': without_slow, 'TUG25_people_removed': int(slow.sum()),
                    'expected_direction': 'negative' if j == 0 else 'positive',
                    'target_min': float(target_values.min()), 'target_max': float(target_values.max()),
                    'gdi_mean_min': float(x.min()), 'gdi_mean_max': float(x.max())}
                rows.append(row)
                for fold in range(5):
                    fold_rows.append({'cohort': cohort, 'model': str(name), 'target': target,
                                      'source_fold': fold, 'spearman_rho': rho(per_fold[i, fold].mean(1), target_values)})
            for k, pid in enumerate(ids):
                people_rows.append({'cohort': cohort, 'model': str(name), 'person': str(pid),
                    'actual_TUG_seconds': float(y[k, 0]), 'actual_POMA_Gait': float(y[k, 1]),
                    'predicted_GDI_left': float(predictions[i, k, 0]),
                    'predicted_GDI_right': float(predictions[i, k, 1]), 'predicted_GDI_mean': float(scores[i, k])})
        corrected = holm([r['permutation_p_two_sided'] for r in rows])
        for row, p in zip(rows, corrected):
            row['holm_p_within_cohort_all_model_targets'] = p
        summaries.extend(rows)
    table('associations.csv', summaries)
    table('person-predictions.csv', people_rows)
    table('leave-one-person-out.csv', loo_rows)
    table('source-fold-correlations.csv', fold_rows)
    save('summary.json', {'rows': summaries, 'cohorts': {
        name: {'people': len(ids), 'models': len(names), 'TUG_range': [float(y[:, 0].min()), float(y[:, 0].max())],
               'POMA_counts': {str(v): int((y[:, 1] == v).sum()) for v in np.unique(y[:, 1])}}
        for name, names, ids, _, y in all_cohorts},
        'prediction_sha256': sha(HERE / 'predictions.npz'), 'protocol_sha256': sha(HERE / 'protocol.json'),
        'primary_output': 'left/right mean; all five frozen source folds averaged',
        'training_performed': False, 'clinical_success_threshold_defined': False,
        'scope': 'Exploratory association, not target-score accuracy, clinical validity or untouched final validation.'})
    print(json.dumps(read(HERE / 'summary.json')['cohorts'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
