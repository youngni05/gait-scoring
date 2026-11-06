"""Independently verify downloaded first-run predictions; no training or protected data."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle-py38'
REMOTE = HERE / 'received/dis01-first-run/extracted'
RUNS = REMOTE / 'runs/dis01_main'
OUT = HERE / 'analysis/dis01-first-run'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def csv_save(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

manifest = load(BUNDLE / 'manifest.json')
assert sha(REMOTE / 'manifest.json') == sha(BUNDLE / 'manifest.json')
for name, digest in manifest.items():
    assert sha(BUNDLE / name) == digest, name
for name in ['run.py', 'batch.py']:
    assert sha(REMOTE / name) == manifest[name], name
cfg = load(REMOTE / 'protocol.PRIVATE_HOST')
assert cfg == load(HERE / 'protocol.PRIVATE_HOST')
config_hash = sha(REMOTE / 'protocol.PRIVATE_HOST')
manifest_hash = sha(REMOTE / 'manifest.json')
summary = load(RUNS / 'summary.json')
assert summary['config_sha256'] == config_hash
assert summary['people'] == 175 and summary['records'] == 203
assert summary['seeds'] == cfg['seeds'] == [20261028]
with np.load(BUNDLE / 'train_only.npz', allow_pickle=False) as z:
    d = {k: z[k].copy() for k in ['y', 'person_id', 'video_id', 'fold']}
with np.load(BUNDLE / 'baseline_predictions.npz', allow_pickle=False) as z:
    predictions = {'baseline': z['predictions'].copy()}
ids, folds, truth = d['person_id'], d['fold'], d['y']
people, counts = np.unique(ids, return_counts=True)
assert len(people) == 175 and len(set(d['video_id'])) == 203
assert truth.shape == (203, 2) and np.isfinite(truth).all()
counts_by_person = dict(zip(people, counts))
weights = np.array([1.0 / (175 * counts_by_person[p] * 2) for p in ids])[:, None]
def weighted(a):
    assert a.shape == (203, 2)
    return float(np.sum(a * weights))

def per_person(a):
    return np.array([np.mean(a[ids == p]) for p in people])

job_evidence = []
for variant in cfg['variants']:
    combined = np.full((203, 2, 2), np.nan)
    coverage = np.zeros(203, dtype=int)
    for fold in cfg['folds']:
        tr = np.flatnonzero(folds != fold)
        te = np.flatnonzero(folds == fold)
        assert len(set(ids[tr])) == 140 and len(set(ids[te])) == 35
        assert not set(ids[tr]) & set(ids[te])
        job = RUNS / f'{variant}_seed20261028_fold{fold}'
        done = load(job / 'complete.json')
        ident = load(job / 'job.json')
        expected = {'config': config_hash, 'manifest': manifest_hash, 'variant': variant,
                    'fold': fold, 'seed': 20261028, 'torch': '2.4.0+cu121', 'device_type': 'cuda'}
        assert ident == done['identity'] == expected
        assert done['steps'] == cfg['steps'] == 200
        assert done['frozen_unchanged'] is True
        assert np.isfinite(done['last_loss_this_process'])
        assert sha(job / 'predictions.npz') == done['predictions_sha256']
        with np.load(job / 'predictions.npz', allow_pickle=False) as z:
            assert np.array_equal(z['indices'], te)
            p = z['predictions']
            assert p.shape == (len(te), 2, 2) and np.isfinite(p).all()
            combined[te] = p
            coverage[te] += 1
        job_evidence.append({'variant': variant, 'fold': fold, 'train_people': 140,
            'evaluation_people': 35, 'train_records': len(tr), 'evaluation_records': len(te),
            'prediction_sha256': done['predictions_sha256'],
            'complete_sha256': sha(job / 'complete.json'), 'steps': done['steps'],
            'frozen_unchanged_is_producer_check': True, 'checkpoint_downloaded': False})
    assert np.all(coverage == 1) and np.isfinite(combined).all()
    predictions[variant] = combined
    np.savez_compressed(OUT / f'{variant}_oof.npz', predictions=combined,
        baseline=predictions['baseline'], person_id=ids, video_id=d['video_id'], fold=folds)
assert len(job_evidence) == 15

metric_rows, band_rows, fold_rows = [], [], []
person_metrics = {}
summary_max_error = 0.0
for name, p in predictions.items():
    assert p.shape == (203, 2, 2) and np.isfinite(p).all()
    arrays = {'front_mae': abs(p[:, 0] - truth), 'back_mae': abs(p[:, 1] - truth),
              'front_back_gap': abs(p[:, 0] - p[:, 1])}
    person_metrics[name] = {k: per_person(v) for k, v in arrays.items()}
    for key, a in arrays.items():
        actual = weighted(a)
        expected = summary['metrics'][name]['means'][key]
        summary_max_error = max(summary_max_error, abs(actual - expected))
        assert abs(actual - expected) < 1e-10, (name, key)
        assert abs(actual - person_metrics[name][key].mean()) < 1e-10
    sd = float(per_person(p.mean(axis=1)).std(ddof=1))
    remote_sd = (summary['metrics'][name]['person_prediction_sd'] if name == 'baseline'
                 else summary['metrics'][name]['person_prediction_sd_per_seed'][0])
    assert abs(sd - remote_sd) < 1e-10
    for wi, window in enumerate(['front', 'back']):
        signed = p[:, wi] - truth
        bins = abs(np.floor(p[:, wi] / 10) - np.floor(truth / 10))
        metric_rows.append({'model': name, 'window': window,
            'person_mae': weighted(abs(signed)), 'person_bias': weighted(signed),
            'exact_bin_rate': weighted(bins == 0), 'within_one_bin_rate': weighted(bins <= 1),
            'two_or_more_bin_rate': weighted(bins >= 2),
            'prediction_min': float(p[:, wi].min()), 'prediction_max': float(p[:, wi].max()),
            'person_prediction_sd': float(per_person(p[:, wi]).std(ddof=1)),
            'front_back_gap': weighted(arrays['front_back_gap']),
            'front_back_same_bin_rate': weighted(np.floor(p[:, 0] / 10) == np.floor(p[:, 1] / 10))})
        for band in np.unique(np.floor(truth / 10).astype(int)):
            mask = np.floor(truth / 10) == band
            vals = []
            for pid in people:
                use = mask & (ids == pid)[:, None]
                if use.any():
                    vals.append([abs(signed[use]).mean(), signed[use].mean(), (bins[use] == 0).mean()])
            avg = np.mean(vals, axis=0)
            band_rows.append({'model': name, 'window': window, 'actual_bin_low': int(band * 10),
                'people_in_band': len(vals), 'record_side_count': int(mask.sum()),
                'person_within_band_mae': float(avg[0]), 'person_within_band_bias': float(avg[1]),
                'person_within_band_exact_bin_rate': float(avg[2])})
    for fold in range(5):
        fold_people = np.isin(people, ids[folds == fold])
        fold_rows.append({'model': name, 'fold': fold,
            **{k: float(v[fold_people].mean()) for k, v in person_metrics[name].items()}})

draw = np.random.default_rng(20261029).integers(0, 175, (10000, 175))
comparisons = {}
pairs = [('baseline','head_only'), ('head_only','tail'), ('head_only','last_stage'),
         ('tail','last_stage'), ('baseline','tail'), ('baseline','last_stage')]
bootstrap_max_error = 0.0
for ref, name in pairs:
    comparison = {}
    for metric in person_metrics[ref]:
        diff = person_metrics[ref][metric] - person_metrics[name][metric]
        ci = np.percentile(diff[draw].mean(axis=1), [2.5, 97.5])
        row = {'reduction': float(diff.mean()), 'person_bootstrap95': ci.tolist(),
               'people_improved': int((diff > 0).sum()), 'people_worsened': int((diff < 0).sum()),
               'people_equal': int((diff == 0).sum())}
        key = f'{ref}_minus_{name}'
        if key in summary['comparisons']:
            expected = summary['comparisons'][key][metric]
            err = max(abs(row['reduction']-expected['reduction']),
                      float(abs(ci-np.array(expected['person_bootstrap95'])).max()))
            bootstrap_max_error = max(bootstrap_max_error,err)
            assert err < 1e-10
        comparison[metric] = row
    comparisons[f'{ref}_minus_{name}'] = comparison

cpu_file = HERE.parent / '20261030-cpu-head-study/runs/main/all_predictions.npz'
with np.load(cpu_file, allow_pickle=False) as z:
    for key in ['person_id', 'video_id', 'fold']:
        assert np.array_equal(z[key], d[key])
    cpu_gpu_error = float(abs(z['predictions'] - predictions['head_only']).max())
assert cpu_gpu_error < 1e-8
assert load(REMOTE/'checks/dis01_replay/replay.json')['real_data_optimizer_steps'] == 0
for batch in [1,32]:
    check = load(REMOTE/f'checks/dis01_profile_b{batch}/checks.json')
    assert check['synthetic_only'] and check['real_data_optimizer_steps'] == 0
    assert all(r['resume_weight_max_error'] == 0 for r in check['results'])

csv_save(OUT/'metrics.csv',metric_rows)
csv_save(OUT/'score_bands.csv',band_rows)
csv_save(OUT/'fold_metrics.csv',fold_rows)
save(OUT/'summary.json', {'metrics':metric_rows,'comparisons':comparisons,
    'bin_rule':'floor(GDI/10); no clipping; descriptive bins, not clinical grades',
    'aggregation':'Equal people; average sides and records within each person first',
    'independent_final_evaluation':False,'one_seed_only':True})
verification = {'passed':True,'completed_jobs':15,'real_data_optimizer_steps':3000,
    'archive_sha256':sha(HERE/'received/dis01-first-run/results-review.tar.gz'),
    'dataset_sha256':sha(BUNDLE/'train_only.npz'),'config_sha256':config_hash,
    'manifest_sha256':manifest_hash,'model_code_matches_prepared_copy':True,
    'person_disjoint_per_fold':True,'all_records_predicted_once_per_variant':True,
    'prediction_hashes_verified':True,'summary_max_absolute_error':summary_max_error,
    'bootstrap_max_absolute_error':bootstrap_max_error,'cpu_gpu_head_max_prediction_error':cpu_gpu_error,
    'remote_log_start':'2026-10-30T14:21:54+09:00','remote_log_end':'2026-10-30T14:38:49+09:00',
    'training_and_aggregation_wall_seconds':1015,
    'checkpoint_weights_recomputed_locally':False,
    'limits':['Checkpoints were not downloaded; no independent model re-inference or weight-scope audit.',
              'Completion and frozen-weight checks are producer records; predictions and aggregation independently checked.',
              'Exploratory development cohort, one seed, fixed short schedule; no clinical or final-test claim.'],
    'jobs':job_evidence}
save(OUT/'verification.json',verification)
save(OUT/'prediction_provenance.json', {'schema':'gdi_gpu_verified_predictions_with_completion_evidence_v1',
    'dataset_sha256':verification['dataset_sha256'],'axes':['record','window','side'],
    'windows':['front','back'],'sides':['left_GDI','right_GDI'],
    'prediction_files':{v:sha(OUT/f'{v}_oof.npz') for v in cfg['variants']},
    'checkpoint_hashes_available':False,'jobs':job_evidence})
print(json.dumps({'passed':True,'metrics':metric_rows,'summary_error':summary_max_error,
    'bootstrap_error':bootstrap_max_error,'cpu_gpu_head_error':cpu_gpu_error,
    'baseline_minus_last_stage':comparisons['baseline_minus_last_stage']},ensure_ascii=False,indent=2))
