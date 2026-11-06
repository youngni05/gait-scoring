"""Check A/root agreement, artifact refusal cases and the saved-head demo."""
import copy
import csv
from pathlib import Path
import evaluate as e
import numpy as np
import demo

ROOT_CHECK = e.HARNESS / 'control/manual-sessions/20261030-parallel-research/integration-check.json'
A = e.HARNESS / 'studies/20261030-score-analysis'
d, baseline, hashes = e.data()
path = e.HERE.parent / 'runs/main/all_predictions.npz'
meta_path = e.HERE / 'cpu_predictions.metadata.json'
pred, meta = e.validate_artifact(path, meta_path, d, hashes)
ours = e.read_json(e.HERE / 'cpu_comparison/summary.json')
a_summary = e.read_json(A / 'summary.json')
root = e.read_json(ROOT_CHECK)
assert root['passed'] and root['matching_source_hashes']
for key in ['train_only.npz', 'baseline_heads.npz', 'baseline_predictions.npz']:
    assert hashes[key] == a_summary['source_hashes'][key]
comparisons = []
for window in ['front', 'back']:
    r = next(row for row in ours['metrics'] if row['model'] == 'ridge' and row['window'] == window)
    a = next(row for row in a_summary['metrics'] if row['model'] == 'motionbert_ridge' and row['window'] == window)
    for key in r:
        if key in a and key not in ['model', 'window']:
            assert abs(r[key] - a[key]) < 1e-12, key
    for label, model in [('baseline', 'ridge'), ('predictions', 'cpu_head_only')]:
        m = next(row for row in ours['metrics'] if row['model'] == model and row['window'] == window)
        for root_key, own_key in [('mae', 'person_macro_mae'), ('exact_bin_rate', 'exact_bin_rate'),
                                 ('within_one_bin_rate', 'within_one_bin_rate')]:
            assert abs(m[own_key] - root['metrics'][label][window][root_key]) < 1e-12
        assert abs(m['person_macro_front_back_gap'] - root['metrics'][label]['front_back_gap']) < 1e-12
    comparisons.append({'window': window, 'A_all_shared_metrics_match': True, 'root_CPU_baseline_comparison_matches': True})
with (A / 'score_bands.csv').open(encoding='utf-8-sig') as f:
    original_bands = list(csv.DictReader(f))
band_checks = 0
for row in ours['score_bands']:
    if row['model'] != 'ridge':
        continue
    a = next(r for r in original_bands if r['model'] == 'motionbert_ridge' and r['window'] == row['window']
             and int(r['actual_bin_low']) == row['actual_bin_low'])
    for key in row:
        if key in a and key not in ['model', 'window']:
            assert abs(float(a[key]) - row[key]) < 1e-12, key
    band_checks += 1
assert band_checks == 16

# Independently recompute CPU within-band bias/error/exact rates as weighted sums.
for row in ours['score_bands']:
    values = baseline if row['model'] == 'ridge' else pred
    wi = ['front', 'back'].index(row['window'])
    mask = (d['y'] >= row['actual_bin_low']) & (d['y'] < row['actual_bin_high_exclusive'])
    signed = values[:, wi] - d['y']
    hits = np.floor(values[:, wi] / 10) == np.floor(d['y'] / 10)
    person_values = []
    for pid in np.unique(d['person_id']):
        where = mask & (d['person_id'] == pid)[:, None]
        if where.any():
            count = np.count_nonzero(where)
            person_values.append([np.sum(abs(signed)[where]) / count,
                                  np.sum(signed[where]) / count, np.sum(hits[where]) / count])
    answer = np.array(person_values).mean(0)
    for key, val in zip(['person_within_band_mae', 'person_within_band_bias', 'person_within_band_exact_rate'], answer):
        assert abs(row[key] - val) < 1e-12

fixtures = e.HERE / 'rejection_fixtures'
fixtures.mkdir(exist_ok=True)
with np.load(path, allow_pickle=False) as z:
    good_arrays = {key: z[key].copy() for key in z.files}
refusals = []
def reject(name, change_arrays=None, change_meta=None):
    arrays = {key: val.copy() for key, val in good_arrays.items()}
    if name != 'changed_embedded_baseline':
        arrays.pop('baseline', None)  # Reach the intended identity/shape guard, not the optional baseline guard.
    declaration = copy.deepcopy(meta)
    if change_arrays is not None:
        change_arrays(arrays)
    fixture = fixtures / (name + '.npz')
    np.savez_compressed(fixture, **arrays)
    declaration['prediction_sha256'] = e.sha(fixture)
    if change_meta is not None:
        change_meta(declaration)
    fixture_meta = fixtures / (name + '.json')
    e.write_json(fixture_meta, declaration)
    try:
        e.validate_artifact(fixture, fixture_meta, d, hashes)
    except ValueError as err:
        refusals.append({'case': name, 'rejected': True, 'reason': str(err)})
    else:
        raise AssertionError('invalid prediction accepted: ' + name)

reject('wrong_dataset', change_meta=lambda x: x.update(dataset_sha256='0' * 64))
reject('wrong_baseline_source', change_meta=lambda x: x.update(baseline_sha256='0' * 64))
reject('wrong_artifact_hash', change_meta=lambda x: x.update(prediction_sha256='0' * 64))
reject('swapped_window_declaration', change_meta=lambda x: x.update(windows=['back', 'front']))
reject('swapped_side_declaration', change_meta=lambda x: x.update(sides=['right_GDI', 'left_GDI']))
reject('missing_fold_provenance', change_meta=lambda x: x.update(fold_models=x['fold_models'][:-1]))
reject('leaked_training_person', change_meta=lambda x: x['fold_models'][0]['training_people'].__setitem__(0, x['fold_models'][0]['evaluation_people'][0]))
reject('missing_record', change_arrays=lambda x: [x.__setitem__(k, v[:-1]) for k, v in list(x.items())])
reject('duplicated_record', change_arrays=lambda x: x['video_id'].__setitem__(-1, x['video_id'][0]))
reject('wrong_row_order', change_arrays=lambda x: [x.__setitem__(k, v[::-1]) for k, v in list(x.items())])
reject('wrong_fold_array', change_arrays=lambda x: x['fold'].__setitem__(0, (int(x['fold'][0]) + 1) % 5))
reject('missing_identity_array', change_arrays=lambda x: x.pop('person_id'))
reject('nan_score', change_arrays=lambda x: x['predictions'].__setitem__((0, 0, 0), np.nan))
reject('wrong_shape', change_arrays=lambda x: x.update(predictions=x['predictions'][:, :, :1]))
reject('changed_embedded_baseline', change_arrays=lambda x: x.update(baseline=x['baseline'] + 1))
assert np.array_equal(np.floor(np.array([-1., 0., 9.999, 10., 100., 110.]) / 10), [-1, 0, 0, 1, 10, 11])

manifest = e.read_json(e.HERE / 'demo/input_manifest.json')
answer = demo.predict(e.HERE / 'demo/features.npz', manifest['checkpoint_path'], e.HERE / 'demo/output.json')
sample_index = manifest['record_index_for_internal_replay']
for wi in range(2):
    for si in range(2):
        v = answer['windows'][wi]['results'][si]
        assert abs(v['predicted_gdi'] - pred[sample_index, wi, si]) < 1e-10
        assert v['bin_lower_inclusive'] == int(np.floor(v['predicted_gdi'] / 10) * 10)
assert all(key not in answer for key in ['person_id', 'video_id', 'record_index'])
assert not answer['encoder_used'] and not answer['training_performed'] and not answer['rgb_or_app_validation']
before_files = e.read_json(e.HERE / 'preserved_primary_files.json')
for relative, fingerprint in before_files.items():
    assert e.sha(e.HERE.parent / relative) == fingerprint, relative
result = {'passed': True, 'A_and_root_comparison': comparisons, 'A_band_rows_checked': band_checks,
          'all_CPU_band_metrics_independently_checked': True, 'refusal_tests': refusals,
          'negative_and_above100_half_open_bin_cases_checked': True,
          'anonymous_demo_replays_saved_oof_prediction': True, 'primary_results_unchanged': True,
          'source_hashes': hashes, 'additional_training_updates': 0, 'encoder_reinference': False,
          'gpu_used': False, 'metadata_declarations_do_not_prove_training_independence_by_themselves': True}
e.write_json(e.HERE / 'verification.json', result)
print(__import__('json').dumps({'passed': True, 'A_band_rows_checked': band_checks,
                               'refusal_tests': len(refusals), 'demo_checked': True}, ensure_ascii=False))
