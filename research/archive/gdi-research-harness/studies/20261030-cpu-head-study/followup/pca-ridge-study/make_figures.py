"""Figures from completed PCA/Ridge outputs; no fitting."""
import run as a
import numpy as np
import importlib.util
import sys
if importlib.util.find_spec('matplotlib') is None:
    sys.path.insert(0, str(a.diag.m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager
font = __import__('pathlib').Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 160})
d, _, baseline, _ = a.diag.m.source_data()
with np.load(a.HERE / 'oof_predictions.npz', allow_pickle=False) as z:
    pred = z['predictions'].copy()
fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.8), sharex=True, sharey=True)
for mi, (name, values) in enumerate([('기존 Ridge', baseline), ('PCA + Ridge', pred)]):
    for wi, window in enumerate(['앞 구간', '뒤 구간']):
        ax = axes[mi, wi]
        for si, side in enumerate(['왼쪽', '오른쪽']):
            truth = np.array([d['y'][d['person_id'] == pid, si].mean() for pid in np.unique(d['person_id'])])
            predicted = np.array([values[d['person_id'] == pid, wi, si].mean() for pid in np.unique(d['person_id'])])
            ax.scatter(truth, predicted, s=12, alpha=.45, label=side)
        ax.plot([35, 110], [35, 110], '--', color='gray', linewidth=1)
        ax.set(xlim=(35, 110), ylim=(35, 110), title=f'{name} / {window}',
               xlabel='실제 GDI(사람 안 기록 평균)', ylabel='예측 GDI(사람 안 기록 평균)')
        ax.grid(alpha=.2); ax.legend()
fig.suptitle('학습에서 제외한 사람의 점수 분포 / PCA 크기는 내부 학습 사람만으로 선택', fontsize=12)
fig.tight_layout(); fig.savefig(a.HERE / '03_held_actual_predicted.png'); plt.close(fig)
print('Held-person comparison figure rendered.')
