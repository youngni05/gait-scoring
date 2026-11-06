"""Train-person versus held-person diagnostics using completed models only."""
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
import argparse
import csv
import json
import numpy as np
import torch


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


def cpu_cubes(d, heads):
    ridge = np.empty((5, 203, 2, 2))
    head = np.empty_like(ridge)
    hashes = []
    for fold in range(5):
        prefix = f'fold{fold}__'
        state_path = m.HERE / f'runs/main/head_only_seed20261028_fold{fold}/checkpoint.pt'
        ck = torch.load(state_path, map_location='cpu', weights_only=True)
        state = ck['model']
        if ck['done'] != 200 or ck['identity']['fold'] != fold:
            raise ValueError('completed CPU model identity mismatch')
        if set(state) != {'mu', 'scale', 'output.weight', 'output.bias'}:
            raise ValueError('unexpected model scope')
        if not np.array_equal(state['mu'].numpy(), heads[prefix + 'scaler_mean']) or not np.array_equal(state['scale'].numpy(), heads[prefix + 'scaler_scale']):
            raise ValueError('fixed scaler mismatch')
        z = (d['features'].astype(np.float64) - heads[prefix + 'scaler_mean']) / heads[prefix + 'scaler_scale']
        ridge[fold] = z @ heads[prefix + 'coef'].T + heads[prefix + 'intercept']
        head[fold] = z @ state['output.weight'].numpy().T + state['output.bias'].numpy()
        hashes.append(m.digest(state_path))
    return ridge, head, hashes


def metadata(d, path, sources, model_hashes, name):
    return {'schema': 'gdi_fold_model_predictions_v1', 'name': name,
        'dataset_sha256': sources['train_only.npz'], 'baseline_sha256': sources['baseline_predictions.npz'],
        'prediction_sha256': m.digest(path), 'shape': [5, 203, 2, 2],
        'axes': ['fold_model', 'record', 'window', 'side'], 'fold_model_order': list(range(5)),
        'windows': ['front', 'back'], 'sides': ['left_GDI', 'right_GDI'],
        'training_scope': 'development_175_only', 'training_window': 'front',
        'fold_models': [{'fold': k, 'model_sha256': model_hashes[k],
            'training_people': np.unique(d['person_id'][d['fold'] != k]).tolist(),
            'evaluation_people': np.unique(d['person_id'][d['fold'] == k]).tolist()} for k in range(5)]}


def load_cube(path, metadata_path, d, sources):
    meta = m.load_json(metadata_path)
    expected = {'schema': 'gdi_fold_model_predictions_v1', 'dataset_sha256': sources['train_only.npz'],
        'baseline_sha256': sources['baseline_predictions.npz'], 'prediction_sha256': m.digest(path),
        'shape': [5, 203, 2, 2], 'axes': ['fold_model', 'record', 'window', 'side'],
        'fold_model_order': list(range(5)), 'windows': ['front', 'back'], 'sides': ['left_GDI', 'right_GDI'],
        'training_scope': 'development_175_only', 'training_window': 'front'}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('metadata mismatch: ' + key)
    with np.load(path, allow_pickle=False) as z:
        for key in ['person_id', 'video_id', 'fold']:
            if key not in z.files or not np.array_equal(z[key], d[key]):
                raise ValueError('record identity/order mismatch: ' + key)
        cube = z['predictions'].copy()
    if cube.shape != (5, 203, 2, 2) or cube.dtype.kind != 'f' or not np.isfinite(cube).all():
        raise ValueError('full fold-model prediction shape/finite mismatch')
    models = meta.get('fold_models', [])
    if len(models) != 5 or [x.get('fold') for x in models] != list(range(5)):
        raise ValueError('fold model missing/duplicate')
    for item in models:
        k = item['fold']
        train, test = item.get('training_people', []), item.get('evaluation_people', [])
        if len(train) != 140 or len(set(train)) != 140 or len(test) != 35 or len(set(test)) != 35:
            raise ValueError('person provenance count mismatch')
        if set(train) != set(d['person_id'][d['fold'] != k]) or set(test) != set(d['person_id'][d['fold'] == k]):
            raise ValueError('person provenance membership mismatch')
        h = item.get('model_sha256', '')
        if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('invalid model fingerprint')
    return cube, meta


def person_entries(d, cube, fold, role, wi, band=None):
    use_records = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
    rows = []
    y = d['y']
    p = cube[fold, :, wi]
    gap = abs(np.floor(p / 10) - np.floor(y / 10))
    for pid in np.unique(d['person_id'][use_records]):
        mask = (use_records & (d['person_id'] == pid))[:, None] & np.ones((203, 2), dtype=bool)
        if band is not None:
            mask &= np.floor(y / 10) == band
        if not mask.any():
            continue
        rows.append({'person': pid.item() if hasattr(pid, 'item') else pid,
            'mae': float(abs(p - y)[mask].mean()), 'bias': float((p - y)[mask].mean()),
            'mse': float(np.square(p - y)[mask].mean()), 'exact_bin_rate': float((gap == 0)[mask].mean()),
            'within_one_bin_rate': float((gap <= 1)[mask].mean()),
            'two_or_more_bin_rate': float((gap >= 2)[mask].mean()),
            'prediction_person_mean': float(p[mask].mean()), 'truth_person_mean': float(y[mask].mean())})
    return rows


def aggregate(rows):
    # Each unique person contributes equally; four training-model views are first averaged within person.
    groups = {pid: [r for r in rows if r['person'] == pid] for pid in {r['person'] for r in rows}}
    keys = ['mae', 'bias', 'mse', 'exact_bin_rate', 'within_one_bin_rate', 'two_or_more_bin_rate']
    result = {key: float(np.mean([np.mean([r[key] for r in group]) for group in groups.values()])) for key in keys}
    result.update(people=len(groups), person_model_views=len(rows))
    weights = np.array([1 / (len(groups) * len(groups[r['person']])) for r in rows])
    pp = np.array([r['prediction_person_mean'] for r in rows])
    yy = np.array([r['truth_person_mean'] for r in rows])
    pmean, ymean = float(weights @ pp), float(weights @ yy)
    psd = float(np.sqrt(weights @ np.square(pp - pmean)))
    ysd = float(np.sqrt(weights @ np.square(yy - ymean)))
    result.update(prediction_person_mean=pmean, truth_person_mean=ymean,
                  prediction_person_sd_ddof0=psd, truth_person_sd_ddof0=ysd,
                  prediction_truth_sd_ratio=psd / ysd if ysd > 0 else None)
    return result


def analyze_cube(d, cube, name):
    all_rows, band_rows, fold_rows = [], [], []
    for role in ['train_people', 'held_people']:
        for wi, window in enumerate(['front', 'back']):
            entries = []
            raw = []
            for fold in range(5):
                local = person_entries(d, cube, fold, role, wi)
                entries.extend(local)
                chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
                raw.extend(cube[fold, chosen, wi].ravel().tolist())
                fold_rows.append({'model': name, 'fold': fold, 'role': role, 'window': window,
                    'records': int(chosen.sum()), **aggregate(local)})
            metrics = aggregate(entries)
            expected_views = 700 if role == 'train_people' else 175
            if metrics['people'] != 175 or metrics['person_model_views'] != expected_views:
                raise ValueError('person/model coverage changed')
            all_rows.append({'model': name, 'role': role, 'window': window,
                'unique_records': 203, 'record_model_views': 812 if role == 'train_people' else 203,
                'raw_prediction_min': min(raw), 'raw_prediction_max': max(raw), **metrics})
            for band in np.unique(np.floor(d['y'] / 10).astype(int)):
                collected = []
                for fold in range(5):
                    collected.extend(person_entries(d, cube, fold, role, wi, band=band))
                band_rows.append({'model': name, 'role': role, 'window': window,
                    'actual_bin_low': int(band * 10), 'actual_bin_high_exclusive': int((band + 1) * 10),
                    'unique_record_side_count': int((np.floor(d['y'] / 10) == band).sum()), **aggregate(collected)})
    return all_rows, band_rows, fold_rows


def plots(d, cubes, metrics, bands, out):
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        sys.path.insert(0, str(m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 160})
    fig, axes = plt.subplots(2, len(cubes), figsize=(6.2 * len(cubes), 8.4), squeeze=False)
    colors = {'train_people': '#2574a9', 'held_people': '#dc7b22'}
    for mi, name in enumerate(cubes):
        for wi, window in enumerate(['front', 'back']):
            ax = axes[wi, mi]
            for role in ['train_people', 'held_people']:
                rows = [r for r in bands if r['model'] == name and r['role'] == role and r['window'] == window]
                ax.plot([r['actual_bin_low'] + 5 for r in rows], [r['bias'] for r in rows],
                        marker='o', color=colors[role], label='학습 사람' if role == 'train_people' else '평가 사람')
            ax.axhline(0, linestyle='--', color='gray', linewidth=1)
            ax.set(title=f'{name} / {window}', xlabel='실제 고정10점 구간의 중심', ylabel='편향: 예측 - 실제(GDI점)')
            ax.legend()
            ax.grid(alpha=.2)
    fig.suptitle('학습·평가 사람 모두 양끝 편향이 남는가? / 사람 동일 비중', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '01_train_held_band_bias.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), sharex=True, sharey=True)
    cube = cubes['ridge']
    limits = [35, 110]
    for ri, role in enumerate(['train_people', 'held_people']):
        for wi, window in enumerate(['front', 'back']):
            ax = axes[ri, wi]
            for fold in range(5):
                chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
                for pid in np.unique(d['person_id'][chosen]):
                    ix = chosen & (d['person_id'] == pid)
                    ax.scatter(d['y'][ix].mean(0), cube[fold, ix, wi].mean(0), s=8,
                               alpha=.14 if role == 'train_people' else .45, color=colors[role])
            ax.plot(limits, limits, '--', color='gray', linewidth=1)
            ax.set(xlim=limits, ylim=limits, title=('학습 사람(각4개 모델)' if ri == 0 else '평가 사람(각1개 모델)') + ' / ' + window,
                   xlabel='실제 GDI(사람 안 기록 평균)', ylabel='예측 GDI(사람 안 기록 평균)')
            ax.grid(alpha=.2)
    fig.suptitle('기존 Ridge의 학습·평가 분포 / 점 개수는 독립 사람 수가 아님', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / '02_ridge_actual_predicted.png')
    plt.close(fig)


def main(external=None, metadata_path=None, name=None, output=None):
    out = Path(output).resolve() if output else HERE
    out.relative_to(HERE)
    out.mkdir(parents=True, exist_ok=True)
    d, heads, baseline, sources = m.source_data()
    ridge, head, head_hashes = cpu_cubes(d, heads)
    cubes = {'ridge': ridge, 'cpu_head_only': head}
    if external:
        cubes = {'ridge': ridge, name: load_cube(external, metadata_path, d, sources)[0]}
    else:
        for model, cube in cubes.items():
            path = out / (model + '_fold_predictions.npz')
            np.savez_compressed(path, predictions=cube, person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
            hashes = [sources['baseline_heads.npz']] * 5 if model == 'ridge' else head_hashes
            meta_path = out / (model + '.metadata.json')
            write_json(meta_path, metadata(d, path, sources, hashes, model))
            loaded, _ = load_cube(path, meta_path, d, sources)
            if not np.array_equal(loaded, cube):
                raise ValueError('export/reload mismatch')
    metrics, bands, folds = [], [], []
    for model, cube in cubes.items():
        values = analyze_cube(d, cube, model)
        metrics.extend(values[0]); bands.extend(values[1]); folds.extend(values[2])
    write_csv(out / 'metrics.csv', metrics)
    write_csv(out / 'score_bands.csv', bands)
    write_csv(out / 'fold_metrics.csv', folds)
    write_json(out / 'summary.json', {'source_hashes': sources, 'metrics': metrics, 'score_bands': bands,
        'fold_metrics': folds, 'bin_rule': 'floor(GDI/10); [10k,10(k+1)); no clipping',
        'training_window': 'front', 'roles': {'train_people': 'person included in this fold model training; diagnostic only',
        'held_people': 'person excluded from this fold model training'},
        'aggregation': 'within person average of records/sides, then average of available model views within person, then equal people average',
        'spread': 'weighted population SD of each person/model bilateral-record mean; person weight 1/175, split across 4 training or 1 held model views',
        'model_views_are_not_independent_people': True, 'additional_training_updates': 0, 'encoder_reinference': False,
        'independent_final_evaluation': False})
    plots(d, cubes, metrics, bands, out)
    print(json.dumps({'output': str(out), 'metrics': metrics}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions'); parser.add_argument('--metadata'); parser.add_argument('--name'); parser.add_argument('--output')
    args = parser.parse_args()
    if any([args.predictions, args.metadata, args.name]) and not all([args.predictions, args.metadata, args.name]):
        parser.error('external predictions require metadata and model name')
    if args.name in ['ridge', 'cpu_head_only']:
        parser.error('use a distinct name for external results')
    main(args.predictions, args.metadata, args.name, args.output)
