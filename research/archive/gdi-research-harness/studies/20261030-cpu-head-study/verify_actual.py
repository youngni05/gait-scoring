"""Independent NumPy replay and aggregation for completed CPU head-only results."""
from pathlib import Path
import json
import time
import cpu_head as m
import numpy as np
import torch

d, heads, baseline, sources = m.source_data()
out = m.HERE / 'runs/main'
summary = m.load_json(out / 'summary.json')
ready = m.load_json(m.HERE / 'protocol.ready.json')
assert ready['status'] == 'ready' and ready['data_kind'] == 'cached_development_head_only'
assert ready['steps'] == 200 and ready['batch_size'] == 32 and ready['head_lr'] == 1e-4
assert ready['weight_decay'] == 0 and ready['seeds'] == [20261028]
with np.load(out / 'all_predictions.npz', allow_pickle=False) as z:
    joined = {k: z[k].copy() for k in z.files}
assert set(joined) == {'predictions', 'baseline', 'person_id', 'video_id', 'fold'}
assert joined['predictions'].shape == (203, 2, 2)
assert np.array_equal(joined['baseline'], baseline)
for key in ('person_id', 'video_id', 'fold'):
    assert np.array_equal(joined[key], d[key])
assert summary['source_fingerprints'] == sources
coverage = []
fold_rows = []
for fold in range(5):
    job = out / f'head_only_seed20261028_fold{fold}'
    record = m.load_json(job / 'complete.json')
    ck = torch.load(job / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert record['identity'] == ck['identity'] == m.load_json(job / 'job.json')
    assert ck['identity']['config'] == m.digest(m.HERE / 'protocol.ready.json')
    assert ck['identity']['implementation'] == m.digest(m.HERE / 'cpu_head.py')
    assert ck['identity']['sources'] == sources
    assert ck['done'] == record['steps'] == 200
    assert m.digest(job / 'predictions.npz') == record['predictions_sha256']
    assert set(ck['model']) == {'mu', 'scale', 'output.weight', 'output.bias'}
    state = {key: value.numpy() for key, value in ck['model'].items()}
    prefix = f'fold{fold}__'
    assert np.array_equal(state['mu'], heads[prefix + 'scaler_mean'])
    assert np.array_equal(state['scale'], heads[prefix + 'scaler_scale'])
    tr = np.flatnonzero(d['fold'] != fold)
    te = np.flatnonzero(d['fold'] == fold)
    assert not set(d['person_id'][tr]) & set(d['person_id'][te])
    with np.load(job / 'predictions.npz', allow_pickle=False) as z:
        assert np.array_equal(z['indices'], te)
        pred = z['predictions'].copy()
    assert np.array_equal(joined['predictions'][te], pred)
    coverage.extend(te.tolist())
    replay_max = 0.0
    for w in range(2):
        standardized = (d['features'][te, w].astype(np.float64) - state['mu']) / state['scale']
        independent = standardized @ state['output.weight'].T + state['output.bias']
        replay_max = max(replay_max, float(np.abs(independent - pred[:, w]).max()))
    assert replay_max < 1e-10
    standardized = (d['features'][tr, 0].astype(np.float64) - state['mu']) / state['scale']
    initial = standardized @ heads[prefix + 'coef'].T + heads[prefix + 'intercept']
    trained = standardized @ state['output.weight'].T + state['output.bias']
    changes = float(np.abs(state['output.weight'] - heads[prefix + 'coef']).max())
    assert changes > 0
    fold_rows.append({'fold': fold, 'train_people': len(set(d['person_id'][tr])),
        'evaluation_people': len(set(d['person_id'][te])), 'train_records': len(tr), 'evaluation_records': len(te),
        'optimizer_updates': 200, 'independent_prediction_max_error': replay_max,
        'weight_max_change': changes, 'bias_max_change': float(np.abs(state['output.bias'] - heads[prefix + 'intercept']).max()),
        'training_record_equal_MSE_before': float(np.square(initial - d['y'][tr]).mean()),
        'training_record_equal_MSE_after': float(np.square(trained - d['y'][tr]).mean()),
        'actual_training_seconds_including_checkpoint_and_prediction': record['elapsed_seconds']})
assert sorted(coverage) == list(range(203))

def independent_metrics(pred):
    person_rows = []
    for person in np.unique(d['person_id']):
        ix = np.flatnonzero(d['person_id'] == person)
        front = np.abs(pred[ix, 0] - d['y'][ix]).mean()
        back = np.abs(pred[ix, 1] - d['y'][ix]).mean()
        gap = np.abs(pred[ix, 0] - pred[ix, 1]).mean()
        person_rows.append([front, back, gap])
    person_rows = np.asarray(person_rows)
    return person_rows.mean(axis=0), person_rows

base_metrics, base_people = independent_metrics(baseline)
head_metrics, head_people = independent_metrics(joined['predictions'])
keys = ['front_mae', 'back_mae', 'front_back_gap']
for i, key in enumerate(keys):
    assert abs(base_metrics[i] - summary['baseline'][key]) < 1e-12
    assert abs(head_metrics[i] - summary['head_only'][key]) < 1e-12
delta = base_people - head_people
assert np.max(np.abs(delta.mean(0) - summary['person_paired_reduction_front_back_gap'])) < 1e-12
rng = np.random.default_rng(20261029)
draws = rng.integers(0, 175, (10000, 175))
intervals = np.percentile(delta[draws].mean(1), [2.5, 97.5], axis=0).T
assert np.max(np.abs(intervals - summary['exploratory_person_bootstrap95_front_back_gap'])) < 1e-12
assert summary['actual_gdi_training_updates'] == 1000 and not summary['synthetic_only']
assert not summary['gpu_used'] and not summary['encoder_loaded']

result = {'status': 'passed', 'actual_gdi_optimizer_updates': 1000, 'head_trainable_parameters': 1026,
    'people': 175, 'records': 203, 'folds': fold_rows,
    'all_records_predicted_once': True, 'saved_model_replay_independent_numpy': True,
    'saved_scalers_unchanged': True, 'prediction_row_identity_preserved': True,
    'left_right_and_front_back_columns_preserved': True, 'source_files_unchanged': True,
    'source_fingerprints': sources, 'baseline': dict(zip(keys, base_metrics.tolist())),
    'head_only': dict(zip(keys, head_metrics.tolist())),
    'reduction_positive_is_improvement': dict(zip(keys, delta.mean(0).tolist())),
    'exploratory_person_bootstrap95': dict(zip(keys, intervals.tolist())),
    'people_improved_worsened_tied': {key: {'improved': int((delta[:, i] > 1e-12).sum()),
        'worsened': int((delta[:, i] < -1e-12).sum()), 'tied': int((np.abs(delta[:, i]) <= 1e-12).sum())} for i, key in enumerate(keys)},
    'total_five_fold_training_seconds_including_checkpoints_and_predictions': sum(x['actual_training_seconds_including_checkpoint_and_prediction'] for x in fold_rows),
    'prediction_contract': {'file': 'runs/main/all_predictions.npz', 'shape': [203, 2, 2],
        'keys': sorted(joined), 'axis0': 'original train_only.npz record order',
        'axis1': ['front', 'back'], 'axis2': ['left_GDI', 'right_GDI']},
    'encoder_loaded': False, 'gpu_used': False, 'independent_final_evaluation': False}
m.save_json(out / 'verification.json', result)
print(json.dumps(result, ensure_ascii=False))
