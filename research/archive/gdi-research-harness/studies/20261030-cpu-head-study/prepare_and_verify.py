"""Read-only source checks and synthetic-only training checks; no GDI update."""
from pathlib import Path
import json
import time
import cpu_head as m
import numpy as np
import torch

OUT = m.HERE / 'technical_v2'
OUT.mkdir(exist_ok=True)
d, heads, baseline, fingerprints = m.source_data()
rows = []
replay = np.full_like(baseline, np.nan)
for fold in range(5):
    tr = np.flatnonzero(d['fold'] != fold)
    te = np.flatnonzero(d['fold'] == fold)
    prefix = f'fold{fold}__'
    mu, scale = heads[prefix + 'scaler_mean'], heads[prefix + 'scaler_scale']
    xtrain = d['features'][tr, 0].astype(np.float64)
    calculated_mu = xtrain.mean(axis=0)
    calculated_scale = xtrain.std(axis=0)
    calculated_scale[calculated_scale == 0] = 1
    mu_error = float(np.abs(mu - calculated_mu).max())
    scale_error = float(np.abs(scale - calculated_scale).max())
    if mu_error > 1e-8 or scale_error > 1e-8:
        raise ValueError(f'train-only scaler differs for fold {fold}: {mu_error}/{scale_error}')
    model = m.Head(heads, fold)
    for w in range(2):
        with torch.inference_mode():
            pred = model(torch.from_numpy(d['features'][te, w])).numpy()
        independent = ((d['features'][te, w].astype(np.float64) - mu) / scale) @ heads[prefix + 'coef'].T + heads[prefix + 'intercept']
        err = float(np.abs(pred - baseline[te, w]).max())
        numpy_err = float(np.abs(pred - independent).max())
        if err > 1e-4 or numpy_err > 1e-10:
            raise ValueError('cached baseline mismatch')
        replay[te, w] = pred
        rows.append({'fold': fold, 'window': ['front', 'back'][w], 'records': len(te),
                     'people': len(set(d['person_id'][te])), 'cached_score_max_error': err,
                     'independent_numpy_max_error': numpy_err, 'train_scaler_mean_max_error': mu_error,
                     'train_scaler_scale_max_error': scale_error})
if not np.isfinite(replay).all():
    raise ValueError('baseline coverage incomplete')
base_metrics, _ = m.metrics(d, replay)
m.save_json(OUT / 'baseline_replay.json', {'rows': rows, 'baseline_metrics': base_metrics,
    'people': 175, 'records': 203, 'left_right_columns_preserved': True,
    'source_fingerprints': fingerprints, 'encoder_loaded': False, 'actual_gdi_optimizer_steps': 0,
    'torch': str(torch.__version__), 'numpy': str(np.__version__), 'cpu_threads': torch.get_num_threads()})

proposal = {'status': 'proposed_not_for_real_training', 'data_kind': 'cached_development_head_only',
    'variants': ['head_only'], 'folds': list(range(5)),
    'seeds': [20261028], 'steps': 200, 'batch_size': 32, 'head_lr': 0.0001,
    'encoder_lr_for_future_gpu_comparison': 0.00001, 'weight_decay': 0.0,
    'checkpoint_every': 10, 'loss': 'record_equal_MSE_left_right', 'train_window': 'front',
    'evaluate_windows': ['front', 'back'], 'optimizer': 'AdamW', 'dropout': 'disabled_all_arms',
    'cpu_threads': 2, 'requires_coordinator_followup_before_real_training': True,
    'selection_basis': 'One fixed conservative setting, not selected from evaluation outcomes. Reuse front-only record-equal MSE, deterministic minibatches and Ridge initialization. Zero weight decay avoids introducing a new coefficient-shrinkage factor. 200 updates bound a first experiment; repeat counts are not tuned on held-out fold scores.',
    'future_gpu_contract': 'Use the same head learning rate, batch size, update count, seed, fold, loss and sample order in head_only/tail/last_stage; encoder learning rate is a separate proposal, not applied here.'}
m.save_json(m.HERE / 'protocol.proposed.json', proposal)
try:
    m.config_ready(m.HERE / 'protocol.proposed.json')
except ValueError:
    pass
else:
    raise ValueError('proposed protocol incorrectly accepted for training')

torch.manual_seed(20261028)
rng = np.random.default_rng(20261028)
synthetic_heads = {}
for fold in range(5):
    prefix = f'fold{fold}__'
    synthetic_heads[prefix + 'scaler_mean'] = np.zeros(512)
    synthetic_heads[prefix + 'scaler_scale'] = np.ones(512)
    synthetic_heads[prefix + 'coef'] = rng.normal(0, 0.01, (2, 512))
    synthetic_heads[prefix + 'intercept'] = np.array([70.0, 71.0])
x = torch.from_numpy(rng.normal(size=(32, 512)).astype(np.float32))
y = torch.from_numpy(rng.normal(70, 10, size=(32, 2)))
model = m.Head(synthetic_heads, 0)
opt = m.optimizer(model, proposal)
initial = {k: v.clone() for k, v in model.state_dict().items()}
identity = {'kind': 'synthetic_only', 'seed': 20261028}
m.update(model, opt, x, y)
m.checkpoint(OUT / 'synthetic_resume.pt', model, opt, 1, identity)
loss2 = m.update(model, opt, x, y)
expected = {k: v.clone() for k, v in model.state_dict().items()}
resumed = m.Head(synthetic_heads, 0)
opt2 = m.optimizer(resumed, proposal)
assert m.restore(OUT / 'synthetic_resume.pt', resumed, opt2, identity) == 1
resumed_loss = m.update(resumed, opt2, x, y)
resume_error = max(float(torch.abs(expected[k] - v).max()) for k, v in resumed.state_dict().items())
assert resume_error == 0 and loss2 == resumed_loss
assert torch.equal(initial['mu'], expected['mu']) and torch.equal(initial['scale'], expected['scale'])
assert not torch.equal(initial['output.weight'], expected['output.weight'])
try:
    m.restore(OUT / 'synthetic_resume.pt', resumed, opt2, {'wrong': True})
except ValueError:
    pass
else:
    raise ValueError('wrong resume identity accepted')

for _ in range(10):
    m.update(model, opt, x, y)
timings = []
for repeat in range(3):
    torch.manual_seed(20261028)
    bench = m.Head(synthetic_heads, 0)
    benchopt = m.optimizer(bench, proposal)
    start = time.perf_counter()
    for step in range(proposal['steps']):
        m.update(bench, benchopt, x, y)
        if (step + 1) % proposal['checkpoint_every'] == 0:
            m.checkpoint(OUT / f'synthetic_benchmark_{repeat}.pt', bench, benchopt, step + 1, identity)
    timings.append(time.perf_counter() - start)

# Exercise full fold coverage, prediction export, reuse of completed jobs and aggregation on fake values.
people = np.concatenate([np.arange(175), np.arange(28)])
folds = people // 35
fake = {'features': rng.normal(size=(203, 2, 512)).astype(np.float32),
        'y': rng.normal(70, 10, size=(203, 2)), 'person_id': people,
        'video_id': np.arange(203), 'fold': folds}
fake_baseline = np.empty((203, 2, 2))
for fold in range(5):
    ix = np.flatnonzero(folds == fold)
    sm = m.Head(synthetic_heads, fold)
    with torch.inference_mode():
        for w in range(2):
            fake_baseline[ix, w] = sm(torch.from_numpy(fake['features'][ix, w])).numpy()
fake_config = {**proposal, 'status': 'ready', 'data_kind': 'synthetic_only', 'steps': 2, 'checkpoint_every': 1}
m.save_json(OUT / 'synthetic.ready.json', fake_config)
try:
    m.config_ready(OUT / 'synthetic.ready.json')
except ValueError:
    pass
else:
    raise ValueError('synthetic protocol incorrectly accepted for real training')
original_source = m.source_data
m.source_data = lambda: (fake, synthetic_heads, fake_baseline, {'kind': 'synthetic_only'})
try:
    m.train_all(OUT / 'synthetic.ready.json', OUT / 'synthetic_pipeline', synthetic_only=True)
    before_reuse = [m.digest(p) for p in sorted((OUT / 'synthetic_pipeline').rglob('checkpoint.pt'))]
    m.train_all(OUT / 'synthetic.ready.json', OUT / 'synthetic_pipeline', synthetic_only=True)
    after_reuse = [m.digest(p) for p in sorted((OUT / 'synthetic_pipeline').rglob('checkpoint.pt'))]
    assert before_reuse == after_reuse
finally:
    m.source_data = original_source

# Independently verify checkpoint loading and metrics exported by the fake pipeline.
with np.load(OUT / 'synthetic_pipeline/all_predictions.npz', allow_pickle=False) as z:
    joined = z['predictions'].copy()
for fold in range(5):
    job = OUT / f'synthetic_pipeline/head_only_seed20261028_fold{fold}'
    ck = torch.load(job / 'checkpoint.pt', map_location='cpu', weights_only=True)
    ix = np.flatnonzero(folds == fold)
    weights = ck['model']['output.weight'].numpy()
    bias = ck['model']['output.bias'].numpy()
    for w in range(2):
        pred = fake['features'][ix, w].astype(np.float64) @ weights.T + bias
        assert float(np.abs(pred - joined[ix, w]).max()) < 1e-10
manual_errors = np.abs(joined - fake['y'][:, None, :]).mean(2)
manual_front = np.mean([manual_errors[people == person, 0].mean() for person in np.unique(people)])
summary = m.load_json(OUT / 'synthetic_pipeline/summary.json')
assert abs(manual_front - summary['head_only']['front_mae']) < 1e-12
assert summary['synthetic_only'] and summary['actual_gdi_training_updates'] == 0

checks = {'synthetic_only': True, 'actual_gdi_optimizer_steps': 0, 'encoder_loaded': False,
          'proposal_rejected_for_training': True, 'synthetic_protocol_rejected_for_real_training': True,
          'resume_exact': True, 'resume_max_error': resume_error,
          'wrong_resume_identity_rejected': True, 'fixed_scaler_unchanged': True,
          'head_weights_updated': True, 'full_five_fold_pipeline_checked': True,
          'completed_jobs_not_retrained': True, 'independent_numpy_saved_model_replay_checked': True,
          'independent_person_macro_aggregation_checked': True,
          'batch32_200steps_with_checkpoints_seconds_three_repeats': timings,
          'estimated_five_fold_training_compute_seconds': float(np.median(timings) * 5),
          'estimate_excludes_startup_source_hashing_reporting_and_real_data_overheads': True,
          'torch': str(torch.__version__), 'numpy': str(np.__version__), 'threads': torch.get_num_threads()}
m.save_json(OUT / 'synthetic_checks.json', checks)
print(json.dumps({'baseline_max_error': max(x['cached_score_max_error'] for x in rows),
                  'baseline': base_metrics, 'technical_checks': checks}, ensure_ascii=False))
