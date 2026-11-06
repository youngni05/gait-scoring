"""Figures and Korean research record from saved, independently verified results."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'expanded695-pls-study'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
sys.path.insert(0, str(m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager


def main():
    s, v = m.load_json(HERE / 'summary.json'), m.load_json(HERE / 'verification.json')
    assert v['verified']
    with np.load(PARENT / 'development695.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    with np.load(HERE / 'oof_predictions.npz', allow_pickle=False) as z:
        predictions, methods = z['predictions'].copy(), z['methods'].tolist()
    candidates = list(csv.DictReader((HERE / 'inner_candidate_scores.csv').open(encoding='utf-8-sig')))
    counts = [2, 4, 8, 16, 32, 64]
    font = Path('C:/Windows/Fonts/malgun.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 10, 'axes.unicode_minus': False, 'savefig.dpi': 160})
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for fold in range(5):
        values = [float(next(r['inner_person_MAE'] for r in candidates if int(r['outer_fold']) == fold and int(r['components']) == k)) for k in counts]
        ax.plot(range(6), values, 'o-', label=f'바깥 분할 {fold}', alpha=0.85)
    ax.axvline(4, color='#555555', ls='--', lw=1)
    ax.set(xticks=range(6), xticklabels=counts, xlabel='PLS 성분 수 · 학습 사람 안에서만 비교',
           ylabel='내부 검증 MAE (점) · 작을수록 좋음', title='다섯 분할 모두 32개 선택, 64개에서는 내부 오차 증가')
    ax.legend(ncol=3, fontsize=9); ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(HERE / '01_inner_selection.png'); plt.close(fig)
    people = np.unique(d['person_id'])
    truth = np.array([d['y'][d['person_id'] == pid].mean() for pid in people])
    means = np.array([[p[d['person_id'] == pid].mean() for pid in people] for p in predictions])
    labels = {1: '확대 Ridge', 3: '이전 PLS · 16개', 4: '확장 PLS · 32개'}
    colors = {1: '#2876ae', 3: '#969ba4', 4: '#d47830'}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for i in [3, 4]:
        axes[0, 0].scatter(truth, means[i], s=10, alpha=0.32, color=colors[i], edgecolors='none', label=labels[i])
    axes[0, 0].plot([30, 120], [30, 120], '--', color='#333333', lw=1)
    axes[0, 0].set(xlim=(30, 120), ylim=(30, 120), xlabel='실제 GDI · 사람별 평균',
                   ylabel='예측 GDI · 사람별 평균', title='예측 분포는 넓어졌지만 중앙 압축은 남는다')
    axes[0, 0].legend(fontsize=9)
    bands = sorted({r['actual_bin_low'] for r in s['score_bands']})
    for i in [1, 3, 4]:
        values = [next(r['bias'] for r in s['score_bands'] if r['method'] == methods[i] and r['actual_bin_low'] == k) for k in bands]
        axes[0, 1].plot(range(len(bands)), values, 'o-', color=colors[i], label=labels[i])
    axes[0, 1].axhline(0, color='#333333', ls='--', lw=1)
    axes[0, 1].set(xticks=range(len(bands)), xticklabels=[f'{k}대' for k in bands], xlabel='실제 GDI 점수대',
                   ylabel='편향 · 예측 - 실제 (점)', title='양끝 편향: 110점대는 오히려 악화 · 2명')
    axes[0, 1].legend(fontsize=9)
    weights = np.empty(d['y'].shape)
    for pid in people:
        mask = d['person_id'] == pid
        weights[mask] = 1 / len(people) / (2 * mask.sum())
    bins = np.arange(30, 125, 5)
    axes[1, 0].stairs(np.histogram(d['y'], bins=bins, weights=weights)[0]*100, bins, label='실제 GDI', color='#222222', lw=2)
    for i in [1, 3, 4]:
        axes[1, 0].stairs(np.histogram(predictions[i], bins=bins, weights=weights)[0]*100, bins, label=labels[i], color=colors[i])
    axes[1, 0].set(xlabel='GDI · 5점 단위', ylabel='사람 가중 비율 (%)', title='같은 695명의 실제·예측 분포')
    axes[1, 0].legend(fontsize=9)
    spread = [truth.std(), means[1].std(), means[3].std(), means[4].std()]
    axes[1, 1].barh(['실제 GDI', labels[1], labels[3], labels[4]][::-1], spread[::-1],
                    color=['#222222', colors[1], colors[3], colors[4]][::-1])
    for iy, value in enumerate(spread[::-1]):
        axes[1, 1].text(value+0.12, iy, f'{value:.2f}', va='center')
    axes[1, 1].set(xlim=(0, 12.5), xlabel='사람별 평균 GDI의 표준편차', title='실제 퍼짐의 59.6% → 67.9%')
    for ax in axes.flat:
        ax.grid(alpha=0.13)
    fig.suptitle('PLS 후보 확대: 선택된 32개 모델과 이전 결과 비교', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(HERE / '02_error_bias_distribution.png'); plt.close(fig)
    names = {'old140_ridge': '기존 규모 Ridge', 'ridge': '확대 Ridge', 'pca_ridge': '확대 PCA+Ridge',
             'pls16': '이전 PLS · 후보 2~16개', 'pls64': '후속 PLS · 후보 2~64개'}
    metric_table = '\n'.join(f"| {names[r['method']]} | {r['person_MAE']:.3f} | {r['bias']:+.3f} | {r['prediction_person_sd']:.2f} | {100*r['exact_bin_rate']:.2f}% | {100*r['two_or_more_bin_rate']:.2f}% |" for r in s['metrics'] if r['group'] == 'all695')
    dimension_table = '\n'.join(f"| {fold} | " + ' | '.join(f"{float(next(r['inner_person_MAE'] for r in candidates if int(r['outer_fold']) == fold and int(r['components']) == k)):.3f}" for k in [16,32,64]) + ' | **32** |' for fold in range(5))
    group_table = '\n'.join('| ' + label + ' | ' + ' | '.join(f"{next(r['person_MAE'] for r in s['metrics'] if r['method'] == method and r['group'] == group):.3f}" for method in ['ridge','pls16','pls64']) + ' |' for group, label in [('original175','기존 175명'),('additional520','추가 520명')])
    band_table = '\n'.join(f"| {k}–{k+9} | {next(r['people'] for r in s['score_bands'] if r['method']=='pls64' and r['actual_bin_low']==k)} | " + ' | '.join(f"{next(r['bias'] for r in s['score_bands'] if r['method']==method and r['actual_bin_low']==k):+.2f}" for method in ['ridge','pls16','pls64']) + ' |' for k in bands)
    report = f'''# PLS 후보 확대: 32·64개까지 비교한 결과

2026-10-30 · 수행: AI가 로컬 CPU에서 학습·검증 · 기존 695명 실험의 후속 탐색

이전 PLS가 모든 분할에서 후보의 최댓값인 16개를 골랐으므로, 더 많은 성분이 도움이 되는지 확인했다. **이번에는 다섯 분할 모두 32개를 골랐고, 같은 평가 사람의 MAE가 8.428→7.975점으로 줄었다.** 64개는 학습 사람 내부의 검증 오차가 더 컸다. 현재 CPU 비교에서는 PLS 32개가 이전 PLS와 Ridge보다 가능성이 높아 보인다. 양끝 점수의 중앙 압축은 여전히 남아 있다.

## 같은 조건에서 무엇만 바꿨는가

기존 695명·1,064기록, 고정 MotionBERT-Lite의 512특징, 영상 앞 124프레임을 그대로 사용했다. ‘앞’은 시간상 초반이며 카메라 정면을 뜻하지 않는다. 이번 변경은 **PLS 후보 [2,4,8,16]→[2,4,8,16,32,64]**다.

- 바깥 학습 556명/평가 139명과 내부 학습 417명/검증 139명의 배정을 기존 저장 파일에서 그대로 읽었다. 같은 사람의 모든 기록은 같은 쪽에 둔다.
- 표준화·PLS는 내부 학습 417명에서만 적합했다. 학습 집단 안에서 모든 사람의 내부 미학습 예측을 모아 사람별 MAE로 크기를 골랐다. 바깥 평가 사람은 선택 함수의 입력에 없다.
- 선택 후 표준화·PLS를 바깥 학습 556명에 다시 적합하고 남긴 139명을 평가했다. 내부·바깥 적합 모두 `scale=False, max_iter=500, tol=1e-6`이며 X는 학습 측 StandardScaler, Y는 원래 좌우 GDI 단위를 유지했다. 학습은 기록마다 같은 비중, 선택·평가는 사람마다 같은 비중이다.
- Ridge·PCA·이전 PLS의 예측은 기존 저장 결과를 그대로 사용했다. 새로 적합한 기존 후보 2·4·8·16개의 내부 예측도 이전과 일치했다. 기존 결과를 덮어쓰지 않았다.

## 압축 크기는 학습 사람 안에서 결정했다

| 바깥 분할 | 내부 MAE · 16개 | 내부 MAE · 32개 | 내부 MAE · 64개 | 선택 |
|---|---:|---:|---:|---:|
{dimension_table}

각 행은 해당 바깥 학습 556명 안의 검증 결과다. 후보 2·4·8개의 수치는 [전체 내부 점수표](inner_candidate_scores.csv)에 함께 저장했다. 모든 분할에서 32개보다 64개의 오차가 커졌으므로, 현재 후보 범위에서는 크기를 늘릴수록 계속 좋아지는 양상이 아니다. 32개는 이번 후보·자료·분할에서 선택된 크기이며 전체 가능한 최적값을 증명한 것은 아니다.

![학습 사람 내부에서 압축 크기 선택](01_inner_selection.png)

## 평가 오차와 점수화 비교

같은 695명의 미학습 예측으로 비교했다. MAE는 좌우·기록의 절대오차를 사람 안에서 평균한 뒤 사람마다 같은 비중으로 합친 값이며 작을수록 좋다. 편향은 예측−실제다.

| 방법 | 평가 MAE↓ | 평균 편향 | 예측 퍼짐¹ | 10점 구간 일치↑ | 두 구간 이상 오차↓ |
|---|---:|---:|---:|---:|---:|
{metric_table}

¹ 사람별 좌우·기록 평균의 표준편차다. 실제 GDI의 퍼짐은 11.13점이다. 기존 규모 Ridge는 매회 기존 140명으로 학습한 참조이고, 나머지는 바깥 학습 556명 조건이다.

- **이전 PLS 대비:** MAE 0.453점(약 5.4%) 감소. 사람 재표집의 탐색적 95% 구간은 **0.253~0.653점**이다. 695명 중 400명에서 오차가 줄고 295명에서 늘었다.
- **확대 Ridge 대비:** MAE 0.552점(약 6.5%) 감소. 탐색적 95% 구간은 **0.333~0.771점**이다. 393명에서 줄고 302명에서 늘었다. PCA 대비 감소도 거의 같다.
- **10점 구간:** 이전 PLS의 정확 일치 35.51%→38.71%, 한 구간 이내 83.65%→84.47%, 두 구간 이상 차이 16.35%→15.53%다. 오차와 구간 지표가 함께 개선됐지만, 정확 일치가 아직 약 39%이므로 충분한 점수화 성능으로 확정하지 않는다.

재표집은 같은 사람을 10,000번 뽑아 비교한 것이며 매번 재학습하지 않았다. 이번 후보 확대는 앞선 결과를 보고 수행한 후속 탐색이다. 따라서 이 구간을 반복 탐색의 영향까지 보정한 검정이나 독립 최종 평가로 해석하지 않는다.

| 평가 집단 | 확대 Ridge | 이전 PLS · 16개 | 후속 PLS · 32개 |
|---|---:|---:|---:|
{group_table}

이번에는 기존 175명과 추가 520명 양쪽에서 이전 PLS보다 오차가 줄었다. 모든 사람·점수대가 좋아진 것은 아니며, 실제 70–79점 구간의 MAE는 이전 PLS 4.884→후속 PLS 5.142점으로 조금 커졌다.

## 양끝 편향과 분포

| 실제 GDI | 해당 사람² | 확대 Ridge 편향 | 이전 PLS 편향 | 후속 PLS 편향 |
|---|---:|---:|---:|---:|
{band_table}

² 같은 사람이 여러 점수대의 좌우·기록을 가질 수 있으므로 사람 수 합계는 695명이 아니다.

40–49점 구간의 과대예측은 **17.42→15.17점**, 90–99점의 과소예측은 **13.99→12.94점**으로 줄었다. 다만 40점대는 20명이며, 110점대 2명에서는 과소예측이 29.16→32.34점으로 커졌다. 양끝 전체가 일관되게 해결됐다고 볼 수 없다.

예측 퍼짐은 **6.64→7.56점**, 실제 퍼짐 대비 **59.6%→67.9%**로 늘었다. 중앙 압축이 일부 완화됐지만 실제 분포보다 여전히 좁다. 실제 좌우 GDI 범위는 39.04~111.79점이다. 이전 PLS의 예측 범위는 48.03~99.98점, 후속 PLS는 36.46~91.51점이다. 최솟값·최댓값만으로 분포 개선을 판단하지 않고 사람별 퍼짐·오차·점수대별 편향을 함께 본다.

구간은 이전처럼 `floor(GDI/10)`으로 나눴고 0–100으로 자르지 않았다. 예측 범위가 넓거나 일부 구간 지표가 좋아졌다는 이유로 임상 점수·진단 기준을 확정하지 않는다.

![오차·양끝 편향·점수 분포 비교](02_error_bias_distribution.png)

## 현재 판단과 GPU 결과 연결

저장된 결과상 **PLS의 후보를 32개까지 넓힌 것이 이번 CPU 예측부에서 확인된 개선 방향**이다. PCA의 단순 압축보다 GDI와의 관련성을 이용하는 PLS 쪽에 가능성이 있어 보인다. 다만 왜 개선됐는지, 어떤 특징이 작용했는지와 MotionBERT의 측정 한계는 이 비교만으로 확정하지 않는다. 기본 모델 교체와 점수화 기준 채택은 아직 결정하지 않았다.

GPU 결과와 연결할 때는 같은 695명 분할·앞 124프레임·사람별 지표를 맞춰 이 결과를 CPU 비교 기준으로 쓸 수 있다. 현재 175명 조건의 GPU 수치와 직접 빼서 파인튜닝 효과로 부르지 않는다. GPU 본체가 GDI 정답을 학습했다면 내부 검증 사람의 정답을 이미 봤는지도 별도로 확인해야 한다. 이번 실험은 처음부터 고정된 사전학습 특징만 사용했다. 실제 GPU 결과를 이용한 비교는 수행하지 않았다.

## 검증과 저장 위치

AI가 기존 환경의 로컬 CPU 2스레드로 내부 PLS 120개·바깥 PLS 5개를 적합했다. 별도 코드에서 StandardScaler→PLSRegression 파이프라인 125개를 다시 적합했고, 저장 계수와 잠재성분으로 계산한 예측도 대조했다. 최대 차이는 약 {max(v['max_inner_prediction_difference'],v['max_outer_prediction_difference']):.2e}점이었다. 수렴 경고 없이 실행됐으며 내부 성분의 최대 반복 횟수는 {v['max_inner_component_iterations']}회로 고정 한도 500회 이내였다.

분할·기존 후보 재현·참조 예측 불변·지표 60행·후보 점수 30행·재표집 비교 3개를 확인했다. 기존 결과 파일 43개와 원본 출처 20개의 지문이 유지됐다. 보호 집단 37/38/223명은 ID 중복 0명만 대조했으며 그 특징·GDI 수치는 읽지 않았다. GPU·본체 재추론·본체 학습·서버 접속·새 패키지 설치는 하지 않았다.

- 실행·검증: [run.py](run.py), [verify.py](verify.py), [verification.json](verification.json)
- 조건·출처: [protocol.json](protocol.json), [provenance.json](provenance.json)
- 수치: [summary.json](summary.json), [metrics.csv](metrics.csv), [score_bands.csv](score_bands.csv), [fold_metrics.csv](fold_metrics.csv)
- 선택: [selected_dimensions.csv](selected_dimensions.csv), [inner_candidate_scores.csv](inner_candidate_scores.csv), [split_audit.csv](split_audit.csv)

`run.py`의 `choose()`는 바깥 학습 자료와 기존 내부 배정만 받아 크기를 선택한다. 경고·지문 불일치·사람 중복이 있으면 중단하며, 완료 결과를 덮어쓰지 않는다. 기존 [개발자료](../expanded695-pls-study/development695.npz)를 재사용하고, 새 예측의 축은 **방법 5개×기록 1,064개×좌우 2개**다. 메타데이터의 `pls64`는 **후보 상한이 64라는 뜻이며 실제 선택은 모든 분할에서 32개**다. `pls16`은 이전 후보 범위의 결과다. [예측 계약](oof.metadata.json)에 같은 자료 지문과 사람 순서를 연결했다. 기존 175명 앞/뒤 GPU 계약과는 구분한다.

검증 명령(PowerShell, D_P 폴더):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B gdi-research-harness/studies/20261030-cpu-head-study/followup/expanded695-pls64-study/verify.py
```
'''
    (HERE / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps({'report_written': True, 'figures': 2}))


if __name__ == '__main__':
    main()
