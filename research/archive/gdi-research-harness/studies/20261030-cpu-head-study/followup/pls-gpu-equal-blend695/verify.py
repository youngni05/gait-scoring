"""Independently reconstruct saved predictions and scalar person-weighted metrics."""
import csv
import math
from statistics import mean, pstdev
import run as r

np, m, HERE = r.np, r.m, r.HERE


def independent_metrics(d, pred, selected=None, band=None):
    selected = set(range(1064)) if selected is None else set(selected)
    groups = {}
    for i in sorted(selected):
        for s in range(2):
            actual, value = float(d['y'][i, s]), float(pred[i, s])
            if band is not None and math.floor(actual / 10) != band:
                continue
            groups.setdefault(str(d['person_id'][i]), []).append((actual, value))
    rows = []
    for pairs in groups.values():
        err = [p - t for t, p in pairs]
        gaps = [abs(math.floor(p / 10) - math.floor(t / 10)) for t, p in pairs]
        rows.append([mean(map(abs, err)), mean(err), mean(e * e for e in err),
                     mean(int(x == 0) for x in gaps), mean(int(x <= 1) for x in gaps),
                     mean(int(x >= 2) for x in gaps), mean(t for t, p in pairs), mean(p for t, p in pairs)])
    cols = list(zip(*rows))
    return dict(people=len(groups), person_MAE=mean(cols[0]), bias=mean(cols[1]), person_RMSE=math.sqrt(mean(cols[2])),
                exact_bin_rate=mean(cols[3]), within_one_bin_rate=mean(cols[4]), two_or_more_bin_rate=mean(cols[5]),
                truth_person_mean=mean(cols[6]), prediction_person_mean=mean(cols[7]),
                truth_person_sd=pstdev(cols[6]), prediction_person_sd=pstdev(cols[7]),
                prediction_truth_sd_ratio=pstdev(cols[7]) / pstdev(cols[6]))


def main():
    summary = m.load_json(HERE / 'summary.json')
    fingerprints = m.load_json(HERE / 'source_hashes.json')
    for path, digest in fingerprints.items():
        assert m.digest(m.WORKSPACE / path) == digest, path
    d = r.npz(r.DATA / 'development695.npz')
    out = r.npz(HERE / 'oof_predictions.npz')
    preds = out['predictions']
    for key in ['person_id', 'video_id', 'fold', 'y', 'source_group']:
        np.testing.assert_array_equal(out[key], d[key])
    raw_gpu = np.empty((1064, 2))
    reconstructed_pls = np.empty((1064, 2))
    max_pls = 0.0
    coverage = np.zeros(1064, dtype=int)
    for f in range(5):
        te = np.flatnonzero(d['fold'] == f)
        model = r.npz(r.PLS / f'models/fold{f}_selected.npz')
        assert not set(d['person_id'][model['train_indices']]) & set(d['person_id'][te])
        z = (d['X'][te] - model['scaler_mean']) / model['scaler_scale']
        reconstructed_pls[te] = z @ model['coef'].T + model['intercept']
        job = r.GPU / f'received/dis01-run/extracted/runs/main/last_stage_seed20261028_fold{f}'
        raw = r.npz(job / 'predictions.npz')
        assert m.digest(job / 'predictions.npz') == m.load_json(job / 'complete.json')['predictions_sha256']
        np.testing.assert_array_equal(te, raw['indices'])
        raw_gpu[te] = raw['predictions'][:, 0]
        coverage[te] += 1
    assert (coverage == 1).all()
    np.testing.assert_allclose(preds[1], reconstructed_pls, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(preds[2], raw_gpu)
    max_pls = float(abs(preds[1] - reconstructed_pls).max())
    np.testing.assert_array_equal(preds[3], (preds[1] + raw_gpu) / 2)
    # Absolute error is convex: equal blending cannot be worse than the average component error.
    assert np.all(abs(preds[3] - d['y']) <= (abs(preds[1] - d['y']) + abs(preds[2] - d['y'])) / 2 + 1e-12)
    worst = 0.0
    for table_name, saved_rows in [('metrics.csv', summary['metrics']), ('score_bands.csv', summary['score_bands']),
                                    ('fold_metrics.csv', summary['fold_metrics'])]:
        csv_rows = list(csv.DictReader((HERE / table_name).open(encoding='utf-8-sig')))
        assert len(csv_rows) == len(saved_rows)
        for saved, csvrow in zip(saved_rows, csv_rows):
            pred = preds[r.METHODS.index(saved['method'])]
            use, band = None, None
            if 'group' in saved and saved['group'] != 'all695':
                use = np.flatnonzero(d['source_group'] == saved['group'])
            if 'fold' in saved:
                use = np.flatnonzero(d['fold'] == saved['fold'])
            if 'actual_bin_low' in saved:
                band = saved['actual_bin_low'] // 10
            for key, value in independent_metrics(d, pred, use, band).items():
                worst = max(worst, abs(value - saved[key]))
                assert abs(value - saved[key]) < 1e-10, (table_name, key)
                assert abs(value - float(csvrow[key])) < 1e-10
    people = sorted(set(d['person_id']))
    errors = np.array([[mean(abs(float(pred[i, s]) - float(d['y'][i, s]))
                        for i in range(1064) if d['person_id'][i] == person for s in range(2))
                        for person in people] for pred in preds])
    rows = list(csv.DictReader((HERE / 'person_comparison.csv').open(encoding='utf-8-sig')))
    assert [x['person_id'] for x in rows] == people
    for j, row in enumerate(rows):
        for mi, method in enumerate(r.METHODS):
            assert abs(float(row[method + '_MAE']) - errors[mi, j]) < 1e-10
    rng = np.random.default_rng(20261029)
    draws = rng.integers(0, 695, (10000, 695))
    for comp in summary['comparisons']:
        delta = errors[r.METHODS.index(comp['baseline'])] - errors[r.METHODS.index(comp['candidate'])]
        ci = np.percentile([mean(delta[draw].tolist()) for draw in draws], [2.5, 97.5])
        np.testing.assert_allclose(ci, comp['exploratory_bootstrap95'], rtol=0, atol=1e-10)
        assert abs(mean(delta) - comp['MAE_reduction']) < 1e-10
        assert int((delta > 1e-12).sum()) == comp['people_improved']
        assert int((delta < -1e-12).sum()) == comp['people_worsened']
    c = summary['complementarity']
    assert abs(float(np.corrcoef(errors[1], errors[2])[0, 1]) - c['person_MAE_correlation']) < 1e-10
    assert int((errors[3] < np.minimum(errors[1], errors[2]) - 1e-12).sum()) == c['blend_better_than_both_people']
    signed, opposite = [], []
    for j, person in enumerate(people):
        pairs = [(float(preds[1, i, side] - d['y'][i, side]), float(preds[2, i, side] - d['y'][i, side]))
                 for i in range(1064) if d['person_id'][i] == person for side in range(2)]
        signed.append([mean(a for a, b in pairs), mean(b for a, b in pairs)])
        opposite.append(mean(int(a*b < 0) for a, b in pairs))
        assert abs(opposite[-1] - float(rows[j]['opposite_error_fraction'])) < 1e-10
    assert abs(mean(opposite) - c['person_weighted_opposite_error_fraction']) < 1e-10
    assert abs(float(np.corrcoef(np.array(signed).T)[0, 1]) - c['person_mean_signed_error_correlation']) < 1e-10
    record_rows = list(csv.DictReader((HERE / 'record_side_predictions.csv').open(encoding='utf-8-sig')))
    assert len(record_rows) == 2128
    for j, row in enumerate(record_rows):
        i, side = divmod(j, 2)
        assert row['person_id'] == str(d['person_id'][i]) and row['video_id'] == str(d['video_id'][i])
        assert row['side'] == ['left_GDI', 'right_GDI'][side] and int(row['fold']) == d['fold'][i]
        assert float(row['actual']) == d['y'][i, side]
        for mi, method in enumerate(r.METHODS):
            assert float(row[method]) == preds[mi, i, side]
    result = dict(verified=True, people=695, records=1064, source_files_preserved=len(fingerprints),
                  aligned_person_video_side_fold_truth=True, PLS_saved_models_repredicted=5,
                  raw_GPU_prediction_files_verified=5, GPU_model_reinference=False,
                  max_PLS_prediction_difference=max_pls, equal_blend_verified=True,
                  independent_metric_rows=len(summary['metrics']) + len(summary['score_bands']) + len(summary['fold_metrics']),
                  max_metric_difference=worst, paired_bootstrap_comparisons=4,
                  person_rows_checked=len(rows), record_side_rows_checked=len(record_rows),
                  complementarity_metrics_verified=True, training=False, GPU_used=False,
                  protected_numeric_data_used=False, independent_final_evaluation=False)
    r.save('verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import json
    main()
