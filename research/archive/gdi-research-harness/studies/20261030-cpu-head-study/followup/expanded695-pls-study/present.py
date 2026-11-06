"""Present saved results; no fitting, selection, or protected data access."""
import os
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
sys.path.insert(0, str(m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

NAMES = {'old140_ridge': '기존 규모 Ridge', 'ridge': '확대 Ridge',
         'pca_ridge': '확대 PCA + Ridge', 'pls': '확대 PLS'}
COLORS = ['#8a8f98', '#2876ae', '#44a489', '#d47830']


def main():
    s = m.load_json(HERE / 'summary.json')
    v = m.load_json(HERE / 'verification.json')
    assert v['verified']
    with np.load(HERE / 'development695.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    with np.load(HERE / 'oof_predictions.npz', allow_pickle=False) as z:
        predictions, methods = z['predictions'].copy(), z['methods'].tolist()
    people = np.unique(d['person_id'])
    truth = np.array([d['y'][d['person_id'] == pid].mean() for pid in people])
    means = np.array([[p[d['person_id'] == pid].mean() for pid in people] for p in predictions])
    font = Path('C:/Windows/Fonts/malgun.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 160})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.scatter(truth, means[i], s=11, alpha=0.4, color=COLORS[i], edgecolors='none')
        ax.plot([30, 120], [30, 120], '--', color='#555555', lw=1)
        ax.set(xlim=(30, 120), ylim=(30, 120), xlabel='실제 GDI · 사람별 좌우·기록 평균',
               ylabel='예측 GDI · 사람별 좌우·기록 평균', title=NAMES[methods[i]])
        ax.grid(alpha=0.15)
    fig.suptitle('같은 695명 평가: 예측이 중앙에 모이는 현상은 남아 있다', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(HERE / '01_actual_predicted.png'); plt.close(fig)
    band_values = sorted({r['actual_bin_low'] for r in s['score_bands']})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    x = np.arange(len(band_values))
    for i, method in enumerate(methods):
        rows = [next(r for r in s['score_bands'] if r['method'] == method and r['actual_bin_low'] == k) for k in band_values]
        axes[0].plot(x, [r['bias'] for r in rows], 'o-', color=COLORS[i], label=NAMES[method], alpha=0.9)
    axes[0].axhline(0, color='#555555', ls='--', lw=1)
    axes[0].set(ylabel='평균 편향 · 예측 - 실제 GDI', title='낮은 점수는 높게, 높은 점수는 낮게 예측하는 편향')
    axes[0].legend(ncol=2); axes[0].grid(alpha=0.15)
    counts = [next(r['people'] for r in s['score_bands'] if r['method'] == 'ridge' and r['actual_bin_low'] == k) for k in band_values]
    axes[1].bar(x, counts, color='#bbc8d7')
    for ix, count in zip(x, counts):
        axes[1].text(ix, count + 6, str(count), ha='center', fontsize=9)
    axes[1].set(xticks=x, xticklabels=[f'{k}~{k+9}' for k in band_values], ylabel='해당 사람 수',
                xlabel='실제 GDI 점수대 · 한 사람이 여러 구간에 포함될 수 있음', ylim=(0, max(counts)*1.22))
    fig.tight_layout(); fig.savefig(HERE / '02_band_bias.png'); plt.close(fig)
    weights = np.empty(d['y'].shape)
    for pid in people:
        mask = d['person_id'] == pid
        weights[mask] = 1 / len(people) / (2 * mask.sum())
    bins = np.arange(30, 125, 5)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), gridspec_kw={'height_ratios': [1.5, 1]})
    axes[0].stairs(np.histogram(d['y'], bins=bins, weights=weights)[0] * 100, bins, label='실제 GDI', color='#222222', lw=2)
    for i, method in enumerate(methods):
        axes[0].stairs(np.histogram(predictions[i], bins=bins, weights=weights)[0] * 100, bins,
                       label=NAMES[method], color=COLORS[i], lw=1.6)
    axes[0].set(xlabel='GDI · 5점 단위', ylabel='사람 가중 비율 (%)', title='예측 분포: PLS가 조금 넓어졌으나 실제보다 좁다')
    axes[0].legend(ncol=3, fontsize=9); axes[0].grid(alpha=0.15)
    labels = ['실제 GDI'] + [NAMES[k] for k in methods]
    stds = [truth.std()] + [p.std() for p in means]
    axes[1].barh(labels[::-1], stds[::-1], color=(['#222222'] + COLORS)[::-1])
    for iy, std in enumerate(stds[::-1]):
        axes[1].text(std + 0.1, iy, f'{std:.2f}', va='center')
    axes[1].set(xlabel='사람별 평균 GDI의 표준편차 · 클수록 점수가 넓게 퍼짐', xlim=(0, 12.5))
    fig.tight_layout(); fig.savefig(HERE / '03_score_distribution.png'); plt.close(fig)
    rows = {r['method']: r for r in s['metrics'] if r['group'] == 'all695'}
    main_table = '\n'.join(f"| {NAMES[k]} | {'140' if k == 'old140_ridge' else '556'}명 | {rows[k]['person_MAE']:.3f} | {rows[k]['bias']:+.3f} | {rows[k]['prediction_person_sd']:.2f} | {100*rows[k]['exact_bin_rate']:.2f}% |" for k in methods)
    group_table = '\n'.join('| ' + ('기존 175명' if group == 'original175' else '추가 520명') + ' | ' + ' | '.join(f"{next(r['person_MAE'] for r in s['metrics'] if r['group'] == group and r['method'] == k):.3f}" for k in methods) + ' |' for group in ['original175', 'additional520'])
    band_table = '\n'.join('| ' + str(band) + '–' + str(band+9) + ' | ' + str(next(r['people'] for r in s['score_bands'] if r['method'] == 'ridge' and r['actual_bin_low'] == band)) + ' | ' + ' | '.join(f"{next(r['bias'] for r in s['score_bands'] if r['method'] == k and r['actual_bin_low'] == band):+.2f}" for k in methods) + ' |' for band in band_values)
    report = f'''# 695명 자료 확대와 Ridge·PCA·PLS 비교

2026-10-30 · CPU 실험 및 독립 재계산 완료 · 수행: AI가 로컬 CPU에서 실행

자료를 늘리는 것이 도움이 되는지, GDI와 관련된 방향으로 특징을 압축하는 PLS가 기존 예측부를 개선할 수 있는지 확인했다. **저장된 결과에 따르면 자료 확대는 전체 오차를 줄였고, PLS는 그 위에서 약 0.10점의 작은 추가 개선을 보였다. 그러나 양끝 점수 압축을 해결하거나 점수화 성능이 확실히 개선됐다고 보기는 어렵다.**

## 같은 평가 사람으로 맞춘 결과

모든 방법을 **동일한 695명·1064기록**에서 비교했다. 각 사람은 자신을 학습에 포함하지 않은 모델로 한 번씩 평가했다. 오차는 좌우·기록의 절대오차를 사람 안에서 먼저 평균한 뒤, 사람마다 같은 비중으로 합친 MAE다. 작을수록 좋다.

| 방법 | 매회 학습 사람 | 평가 MAE | 평균 편향 | 예측 점수 퍼짐¹ | 10점 구간 정확 일치 |
|---|---:|---:|---:|---:|---:|
{main_table}

¹ 사람별 좌우·기록 평균의 표준편차다. 실제 GDI의 퍼짐은 **11.13점**이다. 편향은 예측−실제이며, 양수는 높게 예측했다는 뜻이다. 구간은 `floor(GDI/10)`으로 나누며 0–100으로 자르지 않았다.

- **자료 확대 효과:** 기존 규모 Ridge→확대 Ridge에서 MAE가 0.566점 줄었다. 사람 재표집의 탐색적 95% 구간은 0.369~0.765점이다. 695명 중 409명에서 오차가 줄고 286명에서 늘었다.
- **PCA 효과:** 확대 Ridge와 차이가 약 0.000005점으로 사실상 같았다.
- **PLS 효과:** 확대 Ridge 대비 0.099점 감소했다. 탐색적 95% 구간은 −0.023~+0.221점으로 0을 포함한다. 372명에서 줄고 323명에서 늘었다. 현재 결과만으로 확실한 우세를 주장하지 않는다.

이 구간은 같은 평가 사람을 10,000번 재표집한 계산이다. 모델을 매번 재학습한 것이 아니며, 반복된 모델 검토에 따른 불확실성을 모두 담지는 않는다. 독립 최종 평가 결과도 아니다.

### 기존 사람과 추가 사람을 구분하면

| 평가 집단 | 기존 규모 Ridge | 확대 Ridge | 확대 PCA+Ridge | 확대 PLS |
|---|---:|---:|---:|---:|
{group_table}

**확대의 이득은 주로 추가 520명에서 나타났다. 기존 175명에서는 개선이 확인되지 않았다.** 실제 GDI의 사람별 평균은 기존 175명 71.63점, 추가 520명 75.72점으로 달랐다. 확대 Ridge는 전체 평균의 낮은 예측 편향을 줄였다. 이 관측만으로 그 변화의 원인을 한 가지로 확정하지 않는다.

예전 175명만 평가한 8.276점과 이번 695명 8.527점을 바로 빼서 악화라고 판단하면 평가 집단의 변화가 섞인다. 위 비교는 동일한 695명을 평가한 기존 모델 9.093점을 함께 둬 이 문제를 줄였다. 다만 개발 자료 안의 한 번 정한 사람 분할에서 얻은 비교다.

## 양끝 점수와 점수 분포

| 실제 GDI | 해당 사람² | 기존 규모 Ridge 편향 | 확대 Ridge 편향 | 확대 PCA 편향 | 확대 PLS 편향 |
|---|---:|---:|---:|---:|---:|
{band_table}

² 해당 점수대의 좌우·기록이 하나라도 있는 사람 수다. 같은 사람이 여러 구간에 포함될 수 있으므로 합계는 695명이 아니다.

40–49점의 과대예측은 확대 Ridge 19.99점→PLS 17.42점으로 줄었지만 여전히 크다. 90–99점의 과소예측은 14.50점→13.99점으로 조금 줄었다. 가장 낮은 30점대와 110점대는 각각 2명뿐이며, 그 구간의 우열을 일반화하기 어렵다.

전체 실제 GDI는 **{s['truth_raw_range'][0]:.2f}~{s['truth_raw_range'][1]:.2f}점**이다. 확대 Ridge의 예측은 51.36~88.76점, PLS는 48.03~99.98점이다. PLS의 예측 퍼짐은 실제의 약 59.6%로, 확대 Ridge의 52.6%보다 넓지만 중앙 압축은 남아 있다. 범위가 넓어졌다는 사실만으로 정확도가 높아졌다고 판단하지 않는다.

10점 구간 정확 일치는 확대 Ridge 35.81%→PLS 35.51%로 오히려 조금 낮아졌다. 한 구간 이내 비율은 82.26%→83.65%로 높아졌고, 두 구간 이상 차이는 17.74%→16.35%로 줄었다. **큰 구간 오차를 일부 줄인 신호는 있으나, 10점 단위 점수화가 충분히 잘 된다는 근거는 아직 약하다.** 이 구간은 탐색용 구분이며 임상 등급·건강 점수로 검증한 기준이 아니다. GDI는 0–100점 능력 점수가 아니며 100점 초과 자료도 그대로 사용했다.

![실제와 예측 비교](01_actual_predicted.png)

![점수대별 편향과 사람 수](02_band_bias.png)

![점수 분포와 퍼짐 비교](03_score_distribution.png)

## 무엇을 어떻게 학습했는가

입력은 **고정 MotionBERT-Lite가 미리 뽑아 둔 512개 특징**, 정답은 기록마다 연결된 좌우 GDI다. 추가 520명·861기록은 2026-10-27에 이미 특징이 준비돼 있었다. 이번에는 본체를 다시 실행하지 않고 기존 175명·203기록의 앞 구간 특징과 합쳤다. 추가 자료의 출처·GDI 연결·개별 특징 861개의 증빙과 체크섬을 대조했다.

추가 자료에 뒤 구간 캐시가 없어 이번 비교는 **영상 앞 124프레임만** 사용한다. 여기서 ‘앞’은 시간상 초반 구간이라는 뜻이며 촬영 각도를 뜻하지 않는다. 뒤 구간의 결과를 복제하거나 추정해서 넣지 않았다. 촬영 방향별 성능과 실제 RGB영상→좌표 정확도를 검증한 실험도 아니다.

1. 기존 175명의 바깥 5묶음 배정은 그대로 보존했다. 추가 520명은 GDI를 보지 않고 사람 ID를 seed=20261030으로 섞어 각 묶음에 104명씩 더했다.
2. 매회 556명으로 학습하고 139명으로 평가했다. 같은 사람의 모든 기록은 같은 쪽에 뒀다. 695명은 개발 자료 총수이며, 한 번에 695명을 모두 학습한 뒤 그 사람들을 평가한 것이 아니다.
3. 학습 556명 안에서 다시 4묶음으로 나눴다. 내부 학습 417명/검증 139명으로 압축 크기를 선택했다. 표준화·압축·회귀는 내부 학습 417명에서만 적합했다. 바깥 평가 139명의 특징·정답은 크기 선택에 넣지 않았다.
4. 내부에서 고른 크기로 바깥 학습 556명에 다시 적합한 뒤, 남긴 139명을 평가했다. 모든 사람의 미학습 예측을 모아 최종 비교했다.

| 방법 | 처리 흐름 | 바꾼 부분 |
|---|---|---|
| Ridge | 학습 집단 표준화→512특징→Ridge(alpha1)→좌우 GDI | 학습 자료 확대 |
| PCA+Ridge | 학습 집단 표준화→PCA(k8/16/32/64 중 내부선택, whitening 없음)→Ridge(alpha1) | 특징 변화가 큰 방향으로 압축 |
| PLS | 학습 집단 표준화→PLSRegression(k2/4/8/16 중 내부선택)→좌우 GDI | 정답과의 관련성을 이용해 압축·회귀 |

PLS는 좌우 GDI 두 출력을 함께 학습하는 PLS2 회귀이며, 뒤에 Ridge를 추가하지 않았다. X는 학습 측 StandardScaler로 표준화하고, PLS 자체는 `scale=False, max_iter=500, tol=1e-6`으로 실행했다. 좌우 GDI의 단위가 같아 Y는 별도로 표준화하지 않았다. [scikit-learn PLSRegression 공식 설명](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)

다섯 바깥 묶음 모두 **PCA 64개·PLS 16개**를 골랐다. 둘 다 정해 둔 후보 중 최댓값이다. 특히 PLS 16개를 전체 가능한 크기의 최적값이라고 해석하지 않는다. 후보 범위를 넓힌다면 학습 사람 내부 선택을 유지해야 한다. 이번 결과를 본 뒤 후보나 점수 기준을 바꿔 같은 결과인 것처럼 보고하지 않는다.

PCA 64개는 각 바깥 학습 집단에서 표준화된 특징 변화의 99.999999% 이상을 담았다. 이는 GDI 정보를 그 비율만큼 보존했다는 뜻이 아니다. 이번에는 전체 512개를 쓰는 Ridge와 예측이 거의 같았다. PLS도 고정된 특징을 조합하는 선형 예측부이므로, 이 결과만으로 MotionBERT의 특징 부족 여부나 촬영 한계의 원인을 확정할 수 없다.

모델 적합에서는 기록마다 같은 비중을 줬다. 여러 기록이 있는 사람은 학습에 더 많이 기여할 수 있다. 선택·평가 지표는 사람마다 같은 비중이며, 이는 이전 Ridge와 같은 기준을 유지한 것이다.

## 현재 판단과 GPU 결과 연결

현재는 **자료 확대의 이득을 우선 확인했고, PLS는 작은 개선 가능성이 보이는 후보**로 남긴다. 기본 Ridge를 교체하거나 임상 점수 체계를 확정하지 않았다. 전체 평균 오차 감소와 양끝 구분 능력은 따로 봐야 한다.

기존 GPU 실험은 175명 조건이므로, 이번 695명 CPU 수치와 바로 비교해 본체 파인튜닝의 효과라고 해석할 수 없다. 먼저 GPU 결과는 기존 175명 기준과 비교하고, 확대 실험으로 연결할 때는 이 695명 분할·앞 124프레임·사람별 가중 지표를 맞춰야 한다. 이 폴더에는 같은 사람 순서의 자료·분할·예측을 저장했다. 실제 GPU 결과를 받아 비교한 작업은 아직 하지 않았다.

GPU 본체를 GDI 정답으로 학습했다면, 압축 크기 선택 때 그 본체가 내부 검증 139명의 정답을 이미 봤는지도 확인해야 한다. 전체 파이프라인의 엄격한 내부 검증을 주장하려면 본체 학습도 내부 학습 사람으로 한정해야 한다. 바깥 평가 사람의 제외와 내부 선택의 독립성은 별도로 기록한다.

기존 37/38명과 최종 평가 예약 223명은 ID로 제외 여부만 대조했고, 그 집단의 특징·GDI 수치는 이번에 읽지 않았다. 예약 집단의 역할은 바꾸지 않았다. 이번 695명은 방법을 검토하는 개발 자료이며 독립적인 최종 평가나 임상·고령자·촬영 영상 실증으로 부르지 않는다.

## 검증과 재사용 위치

- 독립 코드에서 내부 160개·바깥 15개 모델을 scikit-learn으로 다시 적합했다. 기존 참조 Ridge는 재학습 없이 재현했다. 저장된 예측과의 최대 차이는 약 {max(v['max_inner_prediction_difference'],v['max_outer_prediction_difference']):.2e}점이다.
- PLS는 저장된 잠재성분으로 직접 계산한 예측도 대조했다. 내부·바깥 사람 중복 0명, 기존 175명의 분할 보존, 크기 선택 재현을 확인했다.
- 전체/집단별 및 점수대별 48행의 지표를 독립적인 사람 가중치로 다시 계산했다. 이전 결과물 100개와 원본 출처 20개의 체크섬이 유지됐다.
- 후보별 내부 점수 40행, CSV 내보내기, 사람 재표집 비교 4개를 재계산했고 보호 집단 ID와의 중복 0명을 확인했다.
- CPU 2스레드와 기존 설치 환경을 사용했다. GPU·서버 접속·MotionBERT 본체 추론·본체 학습·새 패키지 설치는 수행하지 않았다.

핵심 계산은 [run.py](run.py)의 `choose()`다. 이 함수는 바깥 학습 사람만 입력받아 내부 예측으로 크기를 고른다. 평가 사람의 데이터는 이 함수의 인자에 없다. 수렴 경고·입력 지문 불일치·사람 중복이 발생하면 결과를 성공 처리하지 않고 중단하도록 했다.

실험 설정은 [protocol.json](protocol.json), 요약은 [summary.json](summary.json), 검증은 [verification.json](verification.json)에 있다. 비교표는 [metrics.csv](metrics.csv), 점수대는 [score_bands.csv](score_bands.csv), 크기 선택은 [selected_dimensions.csv](selected_dimensions.csv)와 [inner_candidate_scores.csv](inner_candidate_scores.csv)에 저장했다. 원본 경로·지문은 [data_provenance.json](data_provenance.json)을 따른다.

`development695.npz`의 사람·기록 순서를 유지하며, `oof_predictions.npz`의 축은 **방법 4개×기록 1064개×좌우 2개**다. [oof.metadata.json](oof.metadata.json)에 새 앞 구간 전용 계약을 기록했다. 기존 175명 앞/뒤 계약과 모양이 다르므로 같은 형식으로 잘못 연결하면 안 된다. 모든 신규 산출물은 현재 CPU 담당 폴더 안에 저장했으며 공통·GPU·A 담당 파일은 수정하지 않았다.

검증 재실행 명령(PowerShell):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B gdi-research-harness/studies/20261030-cpu-head-study/followup/expanded695-pls-study/verify.py
```

`run.py`는 완료된 실험을 덮어쓰지 않도록 막는다. 추가 후보·분할을 비교할 때는 별도 결과 경로와 설정을 사용한다.
'''
    (HERE / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report': str(HERE / 'report.md'), 'figures': 3}, ensure_ascii=False))


if __name__ == '__main__':
    main()
