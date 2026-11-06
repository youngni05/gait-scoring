"""Predeclared Ridge alpha comparison using frozen cached features only."""
import os
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'train_eval_diagnostic'))
import diagnose as diag
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
import json
import csv
import time
import numpy as np
from threadpoolctl import threadpool_limits


def write_json(path, obj):
    path = Path(path).resolve()
    path.relative_to(HERE)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(metrics, bands, protocol):
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:
        sys.path.insert(0, str(diag.m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.ticker import NullFormatter
    font = Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 160})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for wi, window in enumerate(['front', 'back']):
        for role, label, color in [('train_people', '학습 사람', '#2574a9'), ('held_people', '평가 사람', '#dc7b22')]:
            rows = sorted([r for r in metrics if r['role'] == role and r['window'] == window], key=lambda x: x['alpha'])
            axes[wi].plot([r['alpha'] for r in rows], [r['mae'] for r in rows], 'o-', label=label, color=color)
        axes[wi].set(xscale='log', title=window, xlabel='Ridge alpha (로그축)', ylabel='사람별 MAE(GDI점)')
        axes[wi].set_xticks(protocol['alphas'], labels=[f'{a:g}' for a in protocol['alphas']], rotation=35)
        axes[wi].xaxis.set_minor_formatter(NullFormatter())
        axes[wi].axvline(1, color='gray', linestyle='--', linewidth=1)
        axes[wi].grid(alpha=.2); axes[wi].legend()
    fig.suptitle('규제를 약하게 할 때 학습·평가 오차가 함께 줄어드는가?', fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / '01_alpha_mae.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for wi, window in enumerate(['front', 'back']):
        for role, label, color in [('train_people', '학습 사람', '#2574a9'), ('held_people', '평가 사람', '#dc7b22')]:
            rows = sorted([r for r in metrics if r['role'] == role and r['window'] == window], key=lambda x: x['alpha'])
            axes[wi].plot([r['alpha'] for r in rows], [r['prediction_truth_sd_ratio'] for r in rows], 'o-', label=label, color=color)
        axes[wi].set(xscale='log', title=window, xlabel='Ridge alpha (로그축)', ylabel='예측 퍼짐 / 실제 퍼짐')
        axes[wi].set_xticks(protocol['alphas'], labels=[f'{a:g}' for a in protocol['alphas']], rotation=35)
        axes[wi].xaxis.set_minor_formatter(NullFormatter())
        axes[wi].axhline(1, color='gray', linestyle='--', linewidth=1)
        axes[wi].axvline(1, color='gray', linestyle=':', linewidth=1)
        axes[wi].grid(alpha=.2); axes[wi].legend()
    fig.suptitle('점수 폭 확대와 정확도 개선은 별개 / 가중 표준편차 비율', fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / '02_alpha_spread.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ri, role in enumerate(['train_people', 'held_people']):
        for alpha in protocol['representative_plot_alphas']:
            rows = [r for r in bands if r['role'] == role and r['window'] == 'front' and r['alpha'] == alpha]
            axes[ri].plot([r['actual_bin_low'] + 5 for r in rows], [r['bias'] for r in rows], 'o-', label=f'alpha={alpha:g}')
        axes[ri].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[ri].set(title='학습 사람' if ri == 0 else '평가 사람', xlabel='실제 고정10점 구간의 중심', ylabel='편향: 예측 - 실제(GDI점)')
        axes[ri].grid(alpha=.2); axes[ri].legend()
    fig.suptitle('실행 전 정한 대표 alpha의 앞 구간 편향 / 양끝 표본 적음', fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / '03_alpha_band_bias.png'); plt.close(fig)


def main():
    protocol = diag.m.load_json(HERE / 'protocol.json')
    alphas = protocol['alphas']
    assert alphas == [1e-6, 1e-4, .01, .1, 1, 10, 100, 1000, 10000]
    d, old_heads, baseline, sources = diag.m.source_data()
    cube = np.empty((len(alphas), 5, 203, 2, 2))
    heads = {}
    fit_rows = []
    start = time.perf_counter()
    with threadpool_limits(limits=2):
        for fold in range(5):
            tr = d['fold'] != fold
            prefix = f'fold{fold}__'
            mu, scale = old_heads[prefix + 'scaler_mean'], old_heads[prefix + 'scaler_scale']
            z = (d['features'].astype(np.float64) - mu) / scale
            x = z[tr, 0]
            xmean, ymean = x.mean(0), d['y'][tr].mean(0)
            xc, yc = x - xmean, d['y'][tr] - ymean
            u, singular, vt = np.linalg.svd(xc, full_matrices=False)
            for ai, alpha in enumerate(alphas):
                coef = ((vt.T * (singular / (singular**2 + alpha))) @ (u.T @ yc)).T
                intercept = ymean - xmean @ coef.T
                cube[ai, fold] = z @ coef.T + intercept
                for name, values in [('coef', coef), ('intercept', intercept), ('scaler_mean', mu), ('scaler_scale', scale)]:
                    heads[f'alpha{ai}_fold{fold}__{name}'] = values
                residual = (xc.T @ xc + alpha * np.eye(512)) @ coef.T - xc.T @ yc
                relative = float(np.max(abs(residual)) / max(1.0, np.max(abs(xc.T @ yc))))
                fit_rows.append({'alpha': alpha, 'fold': fold, 'train_records': int(tr.sum()),
                                 'train_people': 140, 'held_people': 35,
                                 'normal_equation_relative_residual': relative,
                                 'coefficient_frobenius_norm': float(np.linalg.norm(coef)),
                                 'training_record_equal_MSE': float(np.square(cube[ai, fold, tr, 0] - d['y'][tr]).mean())})
        if not np.isfinite(cube).all() or not all(np.isfinite(v).all() for v in heads.values()):
            raise ValueError('nonfinite fitted coefficients/predictions')
        np.savez_compressed(HERE / 'all_fold_predictions.npz', alphas=np.array(alphas), predictions=cube,
                            person_id=d['person_id'], video_id=d['video_id'], fold=d['fold'])
        np.savez_compressed(HERE / 'fitted_heads.npz', **heads)
        fit_seconds = time.perf_counter() - start
        metrics, bands, folds = [], [], []
        for ai, alpha in enumerate(alphas):
            values = diag.analyze_cube(d, cube[ai], f'alpha_{alpha:g}')
            for target, rows in zip([metrics, bands, folds], values):
                target.extend([{'alpha': alpha, **r} for r in rows])
            print(f'Analyzed alpha={alpha:g}', flush=True)
    write_csv(HERE / 'metrics.csv', metrics)
    write_csv(HERE / 'score_bands.csv', bands)
    write_csv(HERE / 'fold_metrics.csv', folds)
    write_csv(HERE / 'fit_audit.csv', fit_rows)
    write_json(HERE / 'summary.json', {'protocol': protocol, 'protocol_sha256': diag.m.digest(HERE / 'protocol.json'),
        'source_hashes': sources, 'alphas': alphas, 'models_fitted': len(alphas) * 5,
        'fit_and_prediction_export_seconds': fit_seconds, 'metrics': metrics, 'score_bands': bands,
        'fold_metrics': folds, 'fit_audit': fit_rows, 'independent_final_evaluation': False,
        'hyperparameter_selection_on_outer_results_is_exploratory': True,
        'training_people_metrics_are_diagnostic': True, 'encoder_reinference': False,
        'GPU_used': False, 'aggregation': 'same person weighting and population spread as train_eval_diagnostic'})
    plot(metrics, bands, protocol)
    print(json.dumps({'models_fitted': len(alphas) * 5, 'fit_seconds': fit_seconds,
        'front_held_MAE': {str(a): next(r['mae'] for r in metrics if r['alpha'] == a and r['role'] == 'held_people' and r['window'] == 'front') for a in alphas}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
