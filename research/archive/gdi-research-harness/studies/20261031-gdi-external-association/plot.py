"""Static research figures; identical color scale across separate cohort summaries."""
from common import *
import os
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {
    'mb_ridge': 'MotionBERT + Ridge', 'mb_pls32': 'MotionBERT + PLS',
    'rf_coordinate': 'Coordinate RF', 'rf_motionbert': 'MotionBERT RF',
    'rf_concatenated': 'Coordinate + MotionBERT RF',
    'single_cnn_original': 'Original single-window CNN', 'raw_cnn': 'Matched-split raw CNN',
    'temporal_cnn': 'MotionBERT frame CNN', 'multiwindow_cnn': 'Multi-window CNN',
    'multiwindow_high90': 'Multi-window CNN: high90',
    'mb_leg_time_ridge': 'MotionBERT + leg + time Ridge',
    'rf_coordinate_pls_equal': 'Coordinate RF + PLS',
    'rf_concatenated_pls_equal': 'Combined RF + PLS', 'raw_cnn_pls_equal': 'Raw CNN + PLS',
    'raw_cnn_rf_coordinate_equal': 'Raw CNN + coordinate RF', 'temporal_cnn_pls_equal': 'Frame CNN + PLS'
}


def main():
    summary = read(HERE / 'summary.json')
    names = read(HERE / 'protocol.json')['models']
    keys = [(cohort, target) for cohort in ['common6', 'h36m12', 'cnn10'] for target in ['TUG', 'POMA_Gait']]
    data = np.full((len(names), len(keys)), np.nan)
    lookup = {(r['model'], r['cohort'], r['target']): r for r in summary['rows']}
    for i, name in enumerate(names):
        for j, (cohort, target) in enumerate(keys):
            if (name, cohort, target) in lookup:
                value = lookup[(name, cohort, target)]['spearman_rho']
                if value is not None:
                    data[i, j] = value * (-1 if target == 'TUG' else 1)
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    cmap = plt.colormaps['RdBu'].copy()
    cmap.set_bad('#eeeeee')
    im = ax.imshow(data, vmin=-1, vmax=1, cmap=cmap, aspect='auto')
    ax.set_xticks(range(6), ['TUG\nsame 6', 'POMA\nsame 6', 'TUG\nexisting 12', 'POMA\nexisting 12', 'TUG\nCNN 10', 'POMA\nCNN 10'])
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    for i in range(len(names)):
        for j in range(6):
            ax.text(j, i, f'{data[i, j]:+.2f}' if np.isfinite(data[i, j]) else '—', ha='center', va='center',
                    color='white' if np.isfinite(data[i, j]) and abs(data[i, j]) > .6 else 'black', fontsize=9)
    ax.axvline(1.5, color='black', lw=1.3)
    ax.axvline(3.5, color='black', lw=1.3)
    ax.set_title('Frozen GDI predictions vs actual clinical measures\nPositive color = expected direction (−TUG rho / +POMA rho)', fontsize=12, pad=15)
    fig.colorbar(im, ax=ax, label='Direction-aligned Spearman rho (raw values in CSV)', shrink=.75)
    fig.savefig(HERE / 'association-overview.png', dpi=180)
    plt.close(fig)
    with (HERE / 'person-predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    selected = [('mb_pls32', 'h36m12'), ('rf_coordinate', 'h36m12'), ('multiwindow_cnn', 'cnn10')]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for column, (name, cohort) in enumerate(selected):
        records = [r for r in rows if r['model'] == name and r['cohort'] == cohort]
        for j, (target, field, xlabel) in enumerate([('TUG', 'actual_TUG_seconds', 'Actual TUG (seconds)'),
                                                    ('POMA_Gait', 'actual_POMA_Gait', 'Actual POMA-Gait')]):
            ax = axes[j, column]
            x = np.array([float(r[field]) for r in records])
            y = np.array([float(r['predicted_GDI_mean']) for r in records])
            slow = np.array([float(r['actual_TUG_seconds']) == 25 for r in records])
            ax.scatter(x[~slow], y[~slow], color='#2873a5', alpha=.8)
            ax.scatter(x[slow], y[slow], color='#c64b3c', marker='s', label='TUG 25-second person')
            row = lookup[(name, cohort, target)]
            rho = row['spearman_rho']
            ax.set_title(f'{LABELS[name]}\n{cohort}, rho={rho:+.2f}' if rho is not None else f'{LABELS[name]}\nUndefined rho')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Frozen predicted GDI (left/right mean)')
            ax.grid(alpha=.2)
            if j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Separate supplementary cohorts: these panels are not a matched-input model ranking', fontsize=11)
    fig.savefig(HERE / 'supplementary-scatter.png', dpi=180)
    plt.close(fig)
    print('Two research figures saved', flush=True)


if __name__ == '__main__':
    main()
