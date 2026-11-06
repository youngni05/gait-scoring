"""Fixed-width GDI evaluation for matching person-OOF prediction artifacts.

Does not fit models, change bins, read protected cohorts, or run a pose encoder.
"""
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
HARNESS = HERE.parents[2]
BUNDLE = HARNESS / 'control/manual-sessions/20261029-gpu-finetune-preparation/bundle'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    path = Path(path).resolve()
    path.relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path).resolve()
    path.relative_to(HERE)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def data():
    names = ['train_only.npz', 'baseline_predictions.npz', 'baseline_heads.npz']
    source_hashes = {name: sha(BUNDLE / name) for name in names}
    manifest = read_json(BUNDLE / 'manifest.json')
    if any(source_hashes[name] != manifest[name] for name in names):
        raise ValueError('source bundle fingerprint mismatch')
    with np.load(BUNDLE / 'train_only.npz', allow_pickle=False) as z:
        d = {key: z[key].copy() for key in ['y', 'person_id', 'video_id', 'fold']}
    with np.load(BUNDLE / 'baseline_predictions.npz', allow_pickle=False) as z:
        baseline = z['predictions'].copy()
    if d['y'].shape != (203, 2) or baseline.shape != (203, 2, 2):
        raise ValueError('canonical shape mismatch')
    if len(set(d['person_id'])) != 175 or len(set(d['video_id'])) != 203:
        raise ValueError('canonical identity count mismatch')
    for fold in range(5):
        tr = set(d['person_id'][d['fold'] != fold])
        te = set(d['person_id'][d['fold'] == fold])
        if tr & te or len(tr) != 140 or len(te) != 35:
            raise ValueError('canonical person split mismatch')
    if not np.isfinite(d['y']).all() or not np.isfinite(baseline).all():
        raise ValueError('nonfinite canonical values')
    return d, baseline, source_hashes


def make_metadata(predictions_path, d, source_hashes, model_files, name):
    """Called only for completed, independently verified CPU models in this study."""
    return {'schema': 'gdi_person_oof_predictions_v1', 'name': name,
            'prediction_sha256': sha(predictions_path), 'dataset_sha256': source_hashes['train_only.npz'],
            'baseline_sha256': source_hashes['baseline_predictions.npz'],
            'shape': [203, 2, 2], 'axes': ['record', 'window', 'side'],
            'windows': ['front', 'back'], 'sides': ['left_GDI', 'right_GDI'],
            'prediction_origin': 'person_oof', 'training_scope': 'development_175_only',
            'fold_models': [{'fold': fold, 'model_sha256': sha(model_files[fold]),
                             'training_people': np.unique(d['person_id'][d['fold'] != fold]).tolist(),
                             'evaluation_people': np.unique(d['person_id'][d['fold'] == fold]).tolist()}
                            for fold in range(5)]}


def validate_artifact(path, metadata_path, d, source_hashes):
    meta = read_json(metadata_path)
    expected = {'schema': 'gdi_person_oof_predictions_v1', 'shape': [203, 2, 2],
                'axes': ['record', 'window', 'side'], 'windows': ['front', 'back'],
                'sides': ['left_GDI', 'right_GDI'], 'prediction_origin': 'person_oof',
                'training_scope': 'development_175_only',
                'dataset_sha256': source_hashes['train_only.npz'],
                'baseline_sha256': source_hashes['baseline_predictions.npz'],
                'prediction_sha256': sha(path)}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('prediction metadata mismatch: ' + key)
    with np.load(path, allow_pickle=False) as z:
        required = {'predictions', 'person_id', 'video_id', 'fold'}
        if not required.issubset(z.files):
            raise ValueError('missing prediction identity arrays')
        a = {key: z[key].copy() for key in required}
        if 'baseline' in z.files:
            with np.load(BUNDLE / 'baseline_predictions.npz', allow_pickle=False) as b:
                if not np.array_equal(z['baseline'], b['predictions']):
                    raise ValueError('embedded baseline changed')
    p = a['predictions']
    if p.shape != (203, 2, 2) or p.dtype.kind != 'f' or not np.isfinite(p).all():
        raise ValueError('prediction shape/dtype/finite mismatch')
    if len(a['video_id']) != 203 or len(set(a['video_id'])) != 203:
        raise ValueError('missing or duplicated records')
    for key in ['person_id', 'video_id', 'fold']:
        if a[key].shape != (203,) or not np.array_equal(a[key], d[key]):
            raise ValueError('record order or dataset identity mismatch: ' + key)
    models = meta.get('fold_models', [])
    if len(models) != 5 or [item.get('fold') for item in models] != list(range(5)):
        raise ValueError('missing/duplicate fold provenance')
    for item in models:
        fold = item['fold']
        training = item.get('training_people', [])
        evaluation = item.get('evaluation_people', [])
        if len(training) != 140 or len(evaluation) != 35 or len(set(training)) != 140 or len(set(evaluation)) != 35:
            raise ValueError('invalid provenance count')
        if set(training) != set(d['person_id'][d['fold'] != fold]) or set(evaluation) != set(d['person_id'][d['fold'] == fold]):
            raise ValueError('fold training/evaluation membership mismatch')
        model_hash = item.get('model_sha256', '')
        if len(model_hash) != 64 or any(c not in '0123456789abcdef' for c in model_hash):
            raise ValueError('missing model fingerprint')
    return p, meta


def analyze(d, predictions, name):
    """Match A's fixed bins and per-person aggregation, including within-band means."""
    y, ids = d['y'], d['person_id']
    people = np.unique(ids)
    truth_bins = np.floor(y / 10).astype(int)
    def person(a):
        return np.array([a[ids == pid].mean(axis=0) for pid in people])
    def macro(a):
        return float(person(a).mean())
    metrics, bands = [], []
    for wi, window in enumerate(['front', 'back']):
        p = predictions[:, wi]
        error, signed = abs(p - y), p - y
        difference = abs(np.floor(p / 10).astype(int) - truth_bins)
        metrics.append({'model': name, 'window': window, 'people': 175, 'records': 203,
            'person_macro_mae': macro(error), 'person_macro_bias': macro(signed),
            'exact_bin_rate': macro(difference == 0), 'one_bin_error_rate': macro(difference == 1),
            'two_or_more_bin_error_rate': macro(difference >= 2), 'within_one_bin_rate': macro(difference <= 1),
            'person_mean_bilateral_prediction_sd': float(person(p).mean(1).std(ddof=1)),
            'prediction_min': float(p.min()), 'prediction_max': float(p.max()),
            'person_macro_front_back_gap': macro(abs(predictions[:, 0] - predictions[:, 1])),
            'front_back_same_bin_rate': macro(np.floor(predictions[:, 0] / 10) == np.floor(predictions[:, 1] / 10))})
        for band in np.unique(truth_bins):
            mask = truth_bins == band
            within = []
            for pid in people:
                use = mask & (ids == pid)[:, None]
                if use.any():
                    within.append([error[use].mean(), signed[use].mean(), (difference[use] == 0).mean(),
                                   (difference[use] >= 2).mean()])
            means = np.mean(within, axis=0)
            bands.append({'model': name, 'window': window, 'actual_bin_low': int(band * 10),
                'actual_bin_high_exclusive': int((band + 1) * 10), 'people_in_band': len(within),
                'record_side_count': int(mask.sum()), 'person_within_band_mae': float(means[0]),
                'person_within_band_bias': float(means[1]), 'person_within_band_exact_rate': float(means[2]),
                'person_within_band_two_or_more_rate': float(means[3])})
    return metrics, bands


def figures(out, metrics, bands, names):
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        sys.path.insert(0, str(HARNESS / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 160})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for wi, window in enumerate(['front', 'back']):
        for name in names:
            rows = [r for r in bands if r['model'] == name and r['window'] == window]
            x = [r['actual_bin_low'] + 5 for r in rows]
            axes[wi, 0].plot(x, [r['person_within_band_mae'] for r in rows], marker='o', label=name)
            axes[wi, 1].plot(x, [r['person_within_band_bias'] for r in rows], marker='o', label=name)
        axes[wi, 0].set(title=f'{window}: 실제 점수대별 MAE', ylabel='GDI점')
        axes[wi, 1].set(title=f'{window}: 예측 - 실제 편향', ylabel='GDI점')
        axes[wi, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
        for ax in axes[wi]:
            ax.grid(alpha=.2)
            ax.legend()
            ax.set_xlabel('실제 고정10점 구간의 중심')
    fig.suptitle('사람 안 구간 평균 → 구간 내 사람 평균 / 양끝 표본은 적음', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / '01_band_error_bias.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    width = .8 / len(names)
    for wi, window in enumerate(['front', 'back']):
        for ni, name in enumerate(names):
            row = next(r for r in metrics if r['window'] == window and r['model'] == name)
            x = np.arange(3) - .4 + width * (ni + .5)
            values = [row[key] * 100 for key in ['exact_bin_rate', 'one_bin_error_rate', 'two_or_more_bin_error_rate']]
            bars = axes[wi].bar(x, values, width, label=name)
            axes[wi].bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        axes[wi].set(xticks=np.arange(3), xticklabels=['같은 구간', '한 구간 차이', '두 구간 이상'],
                     title=window, ylabel='사람 동일 가중 비율(%)', ylim=(0, 60))
        axes[wi].legend()
        axes[wi].grid(axis='y', alpha=.2)
    fig.suptitle('고정 [10k,10(k+1)) 구간 / 임상 등급 아님', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / '02_bin_errors.png')
    plt.close(fig)


def evaluate(path, metadata_path, out, name):
    out = Path(out).resolve()
    out.relative_to(HERE)
    out.mkdir(parents=True, exist_ok=True)
    d, baseline, source_hashes = data()
    p, meta = validate_artifact(path, metadata_path, d, source_hashes)
    base_m, base_b = analyze(d, baseline, 'ridge')
    model_m, model_b = analyze(d, p, name)
    metrics, bands = base_m + model_m, base_b + model_b
    write_csv(out / 'metrics.csv', metrics)
    write_csv(out / 'score_bands.csv', bands)
    write_json(out / 'summary.json', {'source_hashes': source_hashes, 'prediction_sha256': sha(path),
        'metadata_sha256': sha(metadata_path), 'bin_rule': 'floor(GDI/10); [10k,10(k+1)); no clipping; no clinical labels',
        'aggregation': 'mean over sides/records within each person, then mean over 175 people',
        'within_band_aggregation': 'mean eligible sides/records within person, then mean of people present in band',
        'metrics': metrics, 'score_bands': bands, 'independent_final_evaluation': False,
        'metadata_is_producer_declaration_not_independent_proof_of_training': True})
    figures(out, metrics, bands, ['ridge', name])
    print(json.dumps({'output': str(out), 'metrics': metrics}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    evaluate(args.predictions, args.metadata, args.output, args.name)
