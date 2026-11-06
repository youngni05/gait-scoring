"""Independently verify person aggregation, labels, statistics and source immutability."""
from common import *
from scipy.stats import spearmanr, pearsonr, permutation_test
import warnings


def same(a, b, tolerance=1e-12):
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        assert (a is None or not np.isfinite(a)) and (b is None or not np.isfinite(b)), (a, b)
    else:
        assert abs(a - b) <= tolerance, (a, b)


def main():
    protocol = read(HERE / 'protocol.json')
    prep = read(HERE / 'input-audit.json')
    inference = read(HERE / 'inference-verification.json')
    report = read(HERE / 'summary.json')
    assert prep['protocol_sha256'] == report['protocol_sha256'] == sha(HERE / 'protocol.json')
    assert prep['input_sha256'] == sha(HERE / 'input.npz')
    assert report['prediction_sha256'] == inference['prediction_sha256'] == sha(HERE / 'predictions.npz')
    assert inference['passed'] and not inference['training_performed'] and not inference['target_calibration_fitted']
    source_count = 0
    for name in ['preparation-sources.json', 'inference-sources.json']:
        for path, digest in read(HERE / name).items():
            assert sha(path) == digest, path
            source_count += 1
    p = arrays(HERE / 'predictions.npz')
    clinical_labels = clinical()
    input_arrays = arrays(HERE / 'input.npz')
    np.testing.assert_array_equal(input_arrays['labels'], [clinical_labels[str(v)] for v in p['ids']])
    assert p['ids'].tolist() == protocol['primary_people']
    assert p['models'].tolist() == protocol['models'] and p['primary'].shape == (len(protocol['models']), 5, 6, 2)
    with (HERE / 'person-predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    with (HERE / 'leave-one-person-out.csv').open(encoding='utf-8-sig', newline='') as stream:
        loo = list(csv.DictReader(stream))
    checks, exact_checks = 0, 0
    by_key = {}
    for cohort, models, pred, owner in [
        ('common6', p['models'], p['primary'], p['ids']),
        ('h36m12', p['h36m_models'], p['h36m'], p['h36m_owner']),
        ('cnn10', p['cnn_models'], p['cnn'], p['cnn_ids'])]:
        ids = sorted(set(owner), key=int)
        for mi, model in enumerate(models):
            person_scores = []
            for pid in ids:
                indices = [i for i, v in enumerate(owner) if v == pid]
                # Explicit record and fold loops, independent of analysis vector axes.
                left = np.mean([np.mean([pred[mi, f, i, 0] for i in indices]) for f in range(5)])
                right = np.mean([np.mean([pred[mi, f, i, 1] for i in indices]) for f in range(5)])
                row = next(r for r in records if r['cohort'] == cohort and r['model'] == model and r['person'] == pid)
                same(float(row['predicted_GDI_left']), left)
                same(float(row['predicted_GDI_right']), right)
                same(float(row['predicted_GDI_mean']), (left + right) / 2)
                np.testing.assert_array_equal([float(row['actual_TUG_seconds']), float(row['actual_POMA_Gait'])], clinical_labels[pid])
                person_scores.append((left + right) / 2)
            x = np.array(person_scores)
            for j, target in enumerate(['TUG', 'POMA_Gait']):
                y = np.array([clinical_labels[pid][j] for pid in ids])
                row = next(r for r in report['rows'] if r['cohort'] == cohort and r['model'] == model and r['target'] == target)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    value = spearmanr(x, y).statistic
                    same(row['spearman_rho'], value)
                    same(row['pearson_r'], pearsonr(x, y).statistic)
                    for k, pid in enumerate(ids):
                        keep = np.arange(len(ids)) != k
                        expected = spearmanr(x[keep], y[keep]).statistic
                        saved = next(r for r in loo if r['cohort'] == cohort and r['model'] == model
                                     and r['target'] == target and r['removed_person'] == pid)['rho']
                        same(float(saved) if saved else None, expected)
                    slow = np.array([clinical_labels[pid][0] == 25. for pid in ids])
                    same(row['rho_without_TUG25_person'], spearmanr(x[~slow], y[~slow]).statistic if slow.any() else None)
                    if cohort == 'common6' and np.isfinite(value):
                        independent = permutation_test((x,), lambda a: spearmanr(a, y).statistic,
                            permutation_type='pairings', n_resamples=np.inf, alternative='two-sided')
                        # scipy's doubled-tail convention differs from the predeclared absolute-statistic test.
                        expected = np.mean(np.abs(independent.null_distribution) >= abs(value) - 1e-12)
                        same(row['permutation_p_two_sided'], expected)
                        exact_checks += 1
                assert row['bootstrap_valid'] + row['bootstrap_degenerate'] == 10000
                by_key[(cohort, str(model), target)] = row
                checks += 1
        # Independently recompute Holm's step-down correction.
        local = [r for r in report['rows'] if r['cohort'] == cohort and r['permutation_p_two_sided'] is not None]
        local.sort(key=lambda r: r['permutation_p_two_sided'])
        prev = 0.
        for i, row in enumerate(local):
            prev = max(prev, min(1., (len(local) - i) * row['permutation_p_two_sided']))
            same(row['holm_p_within_cohort_all_model_targets'], prev)
    # Permissible preprocessing adaptations are batch assertions, never target fitting.
    for name in ['infer.py', 'prepare.py', 'analyze.py']:
        tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ['fit', 'fit_transform', 'backward', 'step']], name
    artifacts = {str(p.relative_to(HERE)): sha(p) for p in HERE.iterdir() if p.is_file() and p.name != 'verification.json'}
    save('verification.json', {'passed': True, 'source_entries_unchanged': source_count,
        'source_checkpoint_replays': len(inference['source_replays']), 'independent_model_target_checks': checks,
        'independent_exact_permutation_checks': exact_checks, 'clinical_label_and_person_aggregation_checks': True,
        'training_or_target_conversion': False, 'artifacts': artifacts})
    print(json.dumps({k: v for k, v in read(HERE / 'verification.json').items() if k != 'artifacts'}), flush=True)


if __name__ == '__main__':
    main()
