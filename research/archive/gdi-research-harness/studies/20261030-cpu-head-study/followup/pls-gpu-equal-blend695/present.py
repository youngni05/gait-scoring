"""Readable Korean research record and figure, generated only after verification."""
import os
import sys
import run as r

np, m, HERE = r.np, r.m, r.HERE
os.environ['MPLCONFIGDIR'] = str(HERE / 'mplconfig')
sys.path.insert(0, str(m.ROOT / 'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

LABELS = {'ridge': '기존 Ridge', 'pls32': '고정 MotionBERT + PLS32',
          'gpu_last_stage': 'MotionBERT 마지막 블록 부분학습', 'equal_blend': '두 예측의 50:50 결합'}


def main():
    s, v = m.load_json(HERE / 'summary.json'), m.load_json(HERE / 'verification.json')
    assert v['verified']
    z = r.npz(HERE / 'oof_predictions.npz')
    people = np.unique(z['person_id'])
    means = np.array([[p[z['person_id'] == pid].mean() for pid in people] for p in z['predictions']])
    truth = np.array([z['y'][z['person_id'] == pid].mean() for pid in people])
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),
                         'font.size': 10, 'axes.unicode_minus': False, 'savefig.dpi': 160})
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    ax[0, 0].scatter(means[1] - truth, means[2] - truth, s=11, alpha=.4, edgecolors='none')
    ax[0, 0].axhline(0, color='gray', lw=1)
    ax[0, 0].axvline(0, color='gray', lw=1)
    ax[0, 0].set(xlabel='PLS: 사람 평균 예측 - 실제', ylabel='부분학습: 사람 평균 예측 - 실제',
                 title=f"오차 방향이 비슷하다 · 상관 {s['complementarity']['person_mean_signed_error_correlation']:.3f}")
    colors = ['#888888', '#d47830', '#2876ae', '#388552']
    short = ['Ridge', 'PLS32', '부분학습', '50:50 결합']
    for i in [1, 2, 3]:
        ax[0, 1].scatter(truth, means[i], s=9, alpha=.24, color=colors[i], label=short[i], edgecolors='none')
    ax[0, 1].plot([30, 120], [30, 120], '--', color='black', lw=1)
    ax[0, 1].set(xlabel='실제 GDI · 사람 평균', ylabel='예측 GDI · 사람 평균', title='결합 후에도 양끝 점수 압축은 남는다')
    ax[0, 1].legend()
    bins = sorted({x['actual_bin_low'] for x in s['score_bands']})
    for i in [1, 2, 3]:
        rows = [next(x for x in s['score_bands'] if x['method'] == r.METHODS[i] and x['actual_bin_low'] == b) for b in bins]
        ax[1, 0].plot(bins, [x['bias'] for x in rows], 'o-', color=colors[i], label=short[i])
        ax[1, 1].hist(means[i], bins=np.arange(30, 125, 5), histtype='step', lw=1.7, color=colors[i], label=short[i])
    counts = [next(x for x in s['score_bands'] if x['method'] == 'pls32' and x['actual_bin_low'] == b)['people'] for b in bins]
    ax[1, 0].axhline(0, color='gray', lw=1)
    ax[1, 0].set(xticks=bins, xticklabels=[f'{b}대\nn={n}' for b, n in zip(bins, counts)],
                 xlabel='실제 GDI 구간 · 같은 사람이 여러 구간에 포함될 수 있음', ylabel='편향 (예측 - 실제)',
                 title='40점대·90점대 편향은 PLS 단독보다 커진다')
    ax[1, 0].legend()
    ax[1, 1].hist(truth, bins=np.arange(30, 125, 5), histtype='step', color='black', lw=1.7, label='실제')
    ax[1, 1].set(xlabel='GDI · 사람 평균', ylabel='사람 수', title='예측 분포: 결합은 PLS보다 좁다')
    ax[1, 1].legend()
    for a in ax.flat:
        a.grid(alpha=.15)
    fig.suptitle('695명 · 같은 사람 분할 · 앞 124프레임 · 학습에서 제외된 예측만 비교', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(HERE / '01_complementarity_and_distribution.png')
    plt.close(fig)
    rows = {x['method']: x for x in s['metrics'] if x['group'] == 'all695'}
    lines = ['# PLS와 GPU 부분학습의 오차 비교 · 50:50 결합', '',
        '**두 모델의 예측을 반씩 섞었지만, 평균 절대오차와 정확한 10점 구간 일치율은 PLS 단독보다 좋아지지 않았다. 두 구간 이상 빗나가는 비율은 줄어 일부 보완 효과는 관찰됐다. 현재는 PLS를 교체할 근거가 부족하다고 본다.**', '',
        '## 무엇을 확인했는가', '',
        '같은 사람에게 두 모델이 서로 다른 실수를 하는지, 예측 점수를 평균 내면 그 실수를 줄일 수 있는지 확인하고 싶었다. 사용자 요청에 따라 AI가 저장된 예측을 계산·검산했다. 새 학습, GPU 접속, 결합 비율 탐색은 하지 않았다.', '',
        '- PLS: 고정 MotionBERT 특징에서 학습 사람 내부로 성분 수를 고른 모델. 모든 분할에서32개가 선택됐다.',
        '- GPU 부분학습: MotionBERT 마지막 블록과 예측부를 추가 학습한 last_stage의 기존 결과.',
        '- 결합: 각 영상·좌우 GDI 예측을 `(PLS + 부분학습) / 2`로 계산. 평균 오차 숫자를 섞는 것이 아니다.',
        '- 자료: 개발695명·1,064기록, 기존5분할, 학습556명/평가139명, 앞124프레임. 각 사람은 자신을 학습하지 않은 모델의 예측으로 비교했다.', '',
        '## 전체 결과', '',
        '| 방법 | MAE↓ | RMSE↓ | 정확한10점 구간↑ | 두 구간 이상 오류↓ | 사람 평균 예측의 표준편차 |',
        '|---|---:|---:|---:|---:|---:|']
    for method, x in rows.items():
        lines.append(f"| {LABELS[method]} | {x['person_MAE']:.4f} | {x['person_RMSE']:.4f} | {100*x['exact_bin_rate']:.2f}% | {100*x['two_or_more_bin_rate']:.2f}% | {x['prediction_person_sd']:.2f} |")
    delta = next(c for c in s['comparisons'] if c['baseline'] == 'pls32' and c['candidate'] == 'equal_blend')
    ci = delta['exploratory_bootstrap95']
    lines += ['', f"실제 사람 평균 점수의 표준편차는 {rows['pls32']['truth_person_sd']:.2f}점이다. 결합 후 예측 분포가 PLS보다 좁아졌으며, 분포가 넓다는 것만으로 정확도를 판단하지는 않는다.", '',
        f"PLS 대비 결합의 MAE 감소량은 {delta['MAE_reduction']:.4f}점(음수이므로 악화), 사람 재표집95% 범위는 {ci[0]:.4f}~{ci[1]:.4f}점이다. 범위가0을 포함하므로 소폭 차이를 확실한 우열로 해석하지 않는다. {delta['people_improved']}명은 개선, {delta['people_worsened']}명은 악화했다. 5분할 중2개에서 개선했다.", '',
        '두 구간 이상 오류는15.53%→14.35%로 감소했지만 정확 구간 일치는38.71%→37.55%로 낮아졌다. 이는 서로 다른 성능 지표의 변화이며, 구간 지표 개선의 통계적 확실성이나 임상적 적합성까지 검증한 것은 아니다. 10점 구간은 탐색용 표시 방식이다.', '',
        '## 서로 다른 실수를 얼마나 했는가', '',
        f"- PLS와 부분학습의 사람별 MAE 상관은 {s['complementarity']['person_MAE_correlation']:.3f}, 사람 평균 점수의 부호 있는 오차 상관은 {s['complementarity']['person_mean_signed_error_correlation']:.3f}이다. 한쪽이 어려워하는 사람을 다른 쪽도 어려워하는 경향이 있다.",
        f"- 같은 영상·좌우에서 오차 방향이 반대인 비율은 사람별로 먼저 평균하면 {100*s['complementarity']['person_weighted_opposite_error_fraction']:.2f}%다. 반대 방향의 오차가 상쇄되는 기회는 일부에 그쳤다.",
        f"- 부분학습 모델 단독이 PLS보다 정확한 사람은311명이지만, 결합이 두 단독 모델 모두보다 정확한 사람은 {s['complementarity']['blend_better_than_both_people']}명이다. 실제 정답을 보고 사람마다 좋은 모델을 고르는 방식은 사용하지 않았다.", '',
        '이 상관은 두 모델의 구조적 원인을 입증하는 결과가 아니다. 같은 입력 정보나 부족한 자료가 공통 한계일 가능성은 있지만 이번 계산만으로 원인을 확정할 수 없다.', '',
        '## 양끝 점수와 자료 집단', '',
        '| 실제 점수대 | 사람 수 | PLS 편향 | 부분학습 편향 | 결합 편향 |', '|---|---:|---:|---:|---:|']
    for b in [40, 90, 110]:
        band = [next(x for x in s['score_bands'] if x['method'] == method and x['actual_bin_low'] == b) for method in r.METHODS[1:]]
        lines.append(f"| {b}–{b+9} | {band[0]['people']} | {band[0]['bias']:+.2f} | {band[1]['bias']:+.2f} | {band[2]['bias']:+.2f} |")
    lines += ['', '편향은 예측−실제다. 40점대 과대예측과90점대 과소예측은 PLS보다 커졌다. 110점대는 줄었지만2명뿐이어서 넓게 일반화하지 않는다. 구간별 인원은 그 점수대의 좌우·기록을 가진 사람 수이므로 같은 사람이 여러 구간에 들어갈 수 있다.', '',
        '| 집단 | PLS MAE | 부분학습 MAE | 결합 MAE |', '|---|---:|---:|---:|']
    for group, label in [('original175', '기존175명'), ('additional520', '추가520명')]:
        vals = [next(x['person_MAE'] for x in s['metrics'] if x['method'] == method and x['group'] == group) for method in r.METHODS[1:]]
        lines.append(f'| {label} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
    lines += ['', '기존175명에서는 결합이 나아졌지만 추가520명에서는 악화했다. 집단마다 효과가 달랐다는 관찰이며, 이 결과를 보고 집단별 결합 비율을 정하지 않았다.', '',
        '![오차의 관계와 점수 분포](01_complementarity_and_distribution.png)', '',
        '## 지금의 판단과 다음 비교에 남길 것', '',
        '이번50:50 결합을 주력 방법으로 바꾸지는 않는다. PLS를 CPU 비교 기준으로 남기고, GPU에서 준비하는 CNN이 도착하면 같은 사람의 오차·양끝 편향·구간 오류를 같은 방식으로 비교할 수 있다. 이번 결과는 결합 자체가 불가능하다는 결론도 아니다. 다른 비율을 시험하거나 학습하지 않았으며, 비율을 학습하려면 바깥 학습 사람 내부의 미학습 예측부터 준비해야 한다.', '',
        'PLS 후보 확대와 GPU last_stage 선택에는 이전 개발 결과가 사용됐다. 이번 결합은 그 이후의 탐색적 비교다. 재표집 구간은 저장된 예측에 조건부이며 모델 선택·학습의 불확실성을 모두 반영하지 않는다. 최종예약223명과 기존37/38명의 수치자료는 사용하지 않았고 독립 최종 평가도 아니다.', '',
        '## 계산·검증과 재개 위치', '',
        '`run.py`는 기존 예측의 사람·영상·좌우·분할·정답을 맞춘 뒤 한 줄로50:50 결합을 만들고 기존 사람 단위 집계를 재사용한다. 사람 안에서 기록·좌우 오차를 먼저 평균하고 사람을 동일 가중치로 합치므로 영상 수가 많은 사람이 전체 오차를 지배하지 않는다. Python·NumPy를 사용했으며 기존 모델을 다시 학습하지 않았다.', '',
        f"`verify.py`에서 기존 PLS 저장모델5개로 평가 예측을 다시 계산하고 GPU 원본 예측5파일을 대조했다. 지표{v['independent_metric_rows']}행을 별도 스칼라 계산으로 검산하고 사람별695행과 재표집4비교를 확인했다. 최대 지표 차이는{v['max_metric_difference']:.3g}이다. 기존 입력·참조파일과 이번 프로토콜을 포함한{v['source_files_preserved']}개의 지문은 계산 전후 일치했다. GPU 체크포인트는 회수되어 있지 않아 GPU 모델 자체를 다시 추론한 검증은 아니다.", '',
        '- `protocol.json`: 평가 전에 고정한 결합 비율·집계·검증 범위.',
        '- `summary.json`, `metrics.csv`, `score_bands.csv`, `fold_metrics.csv`: 전체·집단·점수대·분할 결과.',
        '- `person_comparison.csv`, `record_side_predictions.csv`, `oof_predictions.npz`: 사람별 오차와 실제 결합 예측.',
        '- `source_hashes.json`, `split_audit.csv`, `verification.json`: 원본 보존·분할 대응·독립 계산 근거.',
        '- 실행 순서: 기존 Python3.12 환경에서 `python -B run.py` → `python -B verify.py` → `python -B present.py`. 완료된 `run.py` 결과는 덮어쓰지 않는다.', '']
    (HERE / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved report.md and 01_complementarity_and_distribution.png')


if __name__ == '__main__':
    main()
