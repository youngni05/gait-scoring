"""Present observational input diagnosis without treating associations as causes."""
import csv
import os
import sys
import run as r
np,m,HERE=r.np,r.m,r.HERE
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
sys.path.insert(0,str(m.ROOT/'control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager

LABELS={'upper_missing_fraction':'머리·팔 누락 비율','upper_confidence_mean':'머리·팔 신뢰도',
        'leg_missing_fraction':'다리 누락 비율','leg_confidence_mean':'다리 신뢰도',
        'observed_boundary_fraction':'관측점 정규화 경계 비율','missing_boundary_fraction_all_points':'누락점 정규화 경계 비율',
        'leg_step_p95':'다리 좌표 변화량','leg_acceleration_p95':'다리 좌표 변화의 변화량','leg_length_cv_mean':'영상상 다리 길이 변동',
        'target_distance_from_outer_training_mean':'정답과 학습 평균의 거리','actual_left_right_difference':'실제 좌우 GDI 차이'}


def main():
    s=m.load_json(HERE/'summary.json');v=m.load_json(HERE/'verification.json')
    assert v['verified'] and v['side_diagnostic_metrics_checked']==18
    rows=list(csv.DictReader((HERE/'person_input_errors.csv').open(encoding='utf-8-sig')))
    sides=m.load_json(HERE/'side_diagnostic.json')['rows']
    assoc={a['feature']:a for a in s['associations'] if a['outcome']=='common_error'}
    def values(key):return np.array([float(x[key]) for x in rows])
    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name(),
                         'font.size':10,'axes.unicode_minus':False,'savefig.dpi':160})
    fig,ax=plt.subplots(2,2,figsize=(13,9))
    keys=['upper_missing_fraction','upper_confidence_mean','leg_confidence_mean','leg_step_p95','leg_length_cv_mean']
    yy=np.arange(len(keys))
    ax[0,0].barh(yy-.18,[assoc[k]['spearman'] for k in keys],height=.35,label='전체 순위 상관')
    ax[0,0].barh(yy+.18,[assoc[k]['within_score_cohort_rank_correlation'] for k in keys],height=.35,label='점수대·집단 안의 순위 관계')
    ax[0,0].set(yticks=yy,yticklabels=[LABELS[k] for k in keys],xlim=(-.6,.6),xlabel='상관 · 원인이나 개선 효과가 아님',title='입력 특성과 공통 오차의 관계는 작다')
    ax[0,0].axvline(0,color='gray',lw=1);ax[0,0].legend(fontsize=8)
    ax[0,1].scatter(values('target_distance_from_outer_training_mean'),values('common_error'),s=10,alpha=.4)
    ax[0,1].set(xlabel='실제 GDI와 학습 평균의 거리',ylabel='공통 오차 · 두 MAE 중 작은 값',title='평균에서 먼 정답과 오차가 함께 증가')
    ax[1,0].scatter(values('actual_left_right_difference'),values('common_error'),s=10,alpha=.4,color='#bd7538')
    ax[1,0].set(xlabel='실제 좌우 GDI 차이 · 사람 평균',ylabel='공통 오차',title='좌우 차이가 큰 사람도 확인할 대상')
    vals=[sides[0]['actual_abs_LR_difference']]+[x['predicted_abs_LR_difference'] for x in sides]
    bars=ax[1,1].bar(['실제','PLS','부분학습','50:50 결합'],vals,color=['#555555','#d47830','#2876ae','#388552'])
    for bar,value in zip(bars,vals):ax[1,1].text(bar.get_x()+bar.get_width()/2,value+.12,f'{value:.2f}',ha='center')
    ax[1,1].set(ylim=(0,8.5),ylabel='좌우 GDI 차이의 절댓값 · 사람 동일 가중 평균',title='두 모델이 예측한 좌우 차이는 작다')
    for a in ax.flat:a.grid(alpha=.15)
    fig.suptitle('695명 입력 진단 · 정답을 이용한 사후 분석이며 모델 개선 실험은 아님',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(HERE/'01_input_error_and_side_difference.png');plt.close(fig)
    lines=['# GPU 대기 중 입력 품질·공통 오차 진단','',
        '**이번에 확인한 입력 특성만으로 두 모델의 공통 오차를 설명할 강한 관계는 찾지 못했다. 반면 실제 좌우 GDI 차이는 평균7.21점인데 예측 차이는 PLS1.28점·부분학습0.82점으로 작았다. CNN 결과에서도 좌우 차이를 얼마나 표현하는지 함께 확인할 근거가 생겼다.**','',
        '## 이번 작업의 범위','',
        'GPU를 기다리는 동안 PLS와 부분학습 모델이 공통으로 크게 틀린 사람들의 입력을 살펴보고 싶었다. AI가 기존695명·1,064기록의124프레임·17관절 입력과 학습에서 제외된 예측을 연결해 분석했다. GPU 실행·모델 학습·입력 수정·표본 제외는 하지 않았다. 이는 성능 개선을 측정한 학습 실험이 아니라 다음 입력·출력 비교를 위한 사후 진단이다.','',
        '두 모델 모두 오차가 커야 공통 문제로 다루기 위해 사람별 두 MAE 중 작은 값을 공통 오차로 정했다. 사람마다 기록·좌우를 먼저 평균하고 사람을 같은 비중으로 계산했다. 공통 오차가 큰70명은 검토 목록으로만 저장했으며, 학습 제외 기준이나 임상적인 위험 기준이 아니다.','',
        '## 저장된 입력에서 직접 확인한 내용','',
        f"- 다리6관절의 신뢰도0인 점은0개다. 원래부터 다리가 연속 관측되는 구간을 선별했기 때문이다. 누락이 많은 실제 촬영 상황에까지 일반화할 수 없다.",
        f"- 전체 신뢰도0인 점은{s['inventory']['zero_confidence_points']:,}개다. 모두 저장된 정규화 좌표의 경계에 놓여 있다. 이 좌표를 그대로 CNN에 주는 것이 유리한지는 이번 분석으로 확인하지 않았다.",
        f"- 신뢰도가 양수인 점 중 정규화 경계 근처에 있는 점은{s['inventory']['observed_boundary_points']:,}개다. 생성 코드는 관측점의 최소·최대로 좌표를 정규화하므로 경계점 자체가 생기는 것은 자연스럽다. 이 숫자를 영상 잘림 횟수로 해석하면 안 된다.",
        '- 원 영상·정규화 전 좌표는 이번에 보지 않았다. 카메라 잘림, 관절의 실제 위치 정확도, 좌우 혼동을 확정하지 않는다.','',
        '관측 신뢰도는 좌표 오차의 확률로 검증한 값이 아니다. 머리·팔 묶음에는 변환 과정에서 눈 두 점을 평균한 머리점이 포함된다. 다리 좌표 변화량·길이 변동에는 실제 움직임과 촬영 투영도 영향을 주므로 큰 값 자체를 검출 실패로 부르지 않는다.','',
        '## 입력 특성과 오차의 관계','',
        '| 확인 항목 | 전체 순위 상관 | 점수대·집단 안의 관계 |','|---|---:|---:|']
    for key in ['upper_missing_fraction','upper_confidence_mean','leg_confidence_mean','observed_boundary_fraction','leg_step_p95','leg_acceleration_p95','leg_length_cv_mean']:
        a=assoc[key];lines.append(f"| {LABELS[key]} | {a['spearman']:+.3f} | {a['within_score_cohort_rank_correlation']:+.3f} |")
    lines+=['','양수는 해당 값이 클수록 오차도 큰 경향, 음수는 반대 경향이다. 절댓값이0에 가까울수록 이 방식으로 본 관계는 작다. 신뢰도가 낮거나 좌표 변화가 큰 것이 주된 오차 원인이라는 뚜렷한 신호는 없었다. 입력 품질 문제가 없다고 입증한 것은 아니다.','',
        '‘점수대·집단 안의 관계’는 사람 평균 GDI의10점대와 기존175명/추가520명 집단을 함께 나누고, 각 그룹 안에서 순위의 평균을 뺀 뒤 계산했다. 5명 미만 그룹에 속한4명은 이 계산에서만 제외하여691명이 포함됐다. 전체695명 결과에서 제외한 것은 아니다. 거친 점수대 보정이며 모든 차이를 통제하거나 원인을 밝힌 분석은 아니다.','',
        '정규화 경계 비율은 관측점 수를 분모로 쓰므로 누락률과도 연결된다. 누락점 경계 비율 또한 이 자료에서는 누락률을 다른 방식으로 센 것에 가깝다. 여러 항목의 상관을 독립적인 여러 증거로 세지 않는다.','',
        '## 정답 점수와 출력에서 보인 현상','',
        '| 정답을 이용한 진단 항목 | 전체 순위 상관 | 점수대·집단 안의 관계 |','|---|---:|---:|']
    for key in ['target_distance_from_outer_training_mean','actual_left_right_difference']:
        a=assoc[key];lines.append(f"| {LABELS[key]} | {a['spearman']:+.3f} | {a['within_score_cohort_rank_correlation']:+.3f} |")
    lines+=['','학습 평균은 해당 바깥 학습556명에서 좌우별로 계산했다. 이 표는 정답이 평균에서 멀거나 좌우 차이가 클 때 어떤 오차가 남는지 보는 진단이다. **정답이 오차 계산에도 함께 들어가므로, 위 입력 특성보다 인과적으로 더 중요한 원인이라고 순위를 매길 수 없다.** 실제 예측 때 사용할 입력 특징도 아니다.','',
        '좌우 차이와 오차의 관계를 본 뒤 출력의 좌우 간격을 추가 계산했다. 이 추가 분석은 `side_protocol.json`에 별도로 정의하고 수행했다.','',
        '| 구분 | 좌우 점수 차이의 절댓값 평균 | 좌우 차이 자체의 MAE |','|---|---:|---:|',f"| 실제 정답 | {sides[0]['actual_abs_LR_difference']:.2f} | — |"]
    names=['고정 MotionBERT + PLS','MotionBERT 부분학습','50:50 결합']
    for label,row in zip(names,sides):lines.append(f"| {label} | {row['predicted_abs_LR_difference']:.2f} | {row['LR_difference_MAE']:.2f} |")
    lines+=['','좌우 점수 차이를 크게 출력하기만 하면 정확해지는 것은 아니다. 차이의 방향과 크기가 실제 정답과 맞아야 한다. 입력의 좌우 대응, 한쪽 다리의 가림, 특징의 좌우 정보, 학습 목표 중 무엇이 영향을 주었는지는 아직 구분하지 못했다.','',
        '두 다리의 오차를 계산하면 영상별 평균 절대오차는 `평균 점수 오차`와 `좌우 차이 오차의 절반` 중 큰 값과 정확히 같다. 이 관계를 원 예측과 독립 계산으로 확인했다. 좌우 차이 오차가 더 큰 기록의 비율은 사람별로 먼저 평균하면 PLS29.66%, 부분학습28.33%였다. 따라서 좌우 차이만 고치면 전체 오차가 해결된다는 뜻도 아니다.','',
        '![입력 관계와 좌우 차이](01_input_error_and_side_difference.png)','',
        '## CNN 실험에 어떻게 연결할 것인가','',
        '이미 준비한 CNN 기준 실험은 그대로 진행한다. 이번 상관 결과만으로 입력을 지우거나 사람을 제외하지 않는다. CNN 예측이 도착하면 기존 전체MAE·양끝 편향과 함께 실제/예측 좌우 차이의 크기와 차이 예측 오차도 비교할 수 있다.','',
        '누락점 좌표가 경계로 전달되는 처리와 좌우 정보의 구분은 후속 비교 후보로 남긴다. 마스킹·한쪽 다리별 입력·손실 변경은 아직 채택하거나 실행하지 않았다. 이 진단에서 개선을 보장하는 방법을 찾은 것으로 표현하지 않는다.','',
        '## 검산과 재개 위치','',
        f"기록별 입력 요약{v['record_descriptors_checked']:,}개, 사람별695행, 상관{v['association_rows_checked']}행, 검토 목록70명과 좌우 진단18개 지표를 별도 계산으로 대조했다. 입력·참조·프로토콜{v['source_files_unchanged']}개 지문을 보존했고, 기존 MAE와 사람·영상·좌우·정답·분할 대응을 확인했다.",
        '첫 검산에서 경계점 집계의float32/float64 비교 차이로2개 점의 불일치를 발견했다. 선언한 경계값을float64로 계산하도록 통일해2,175→2,173개로 수정하고 다시 검산했다. 원 입력·개인별 입력 특성·상관·예측은 바뀌지 않았다. 수정 전 집계와 이유를 보존했다.','',
        '새 GPU 작업·보호37/38/223명 수치자료·독립 최종평가를 사용하지 않았다. 표본 선별과 이미 반복해서 본 개발자료에 조건부인 탐색 결과다.','',
        '- `protocol.json`, `side_protocol.json`: 분석 정의와 해석 범위.',
        '- `record_input_features.csv`, `person_input_errors.csv`: 입력 요약과 기존 오차 연결.',
        '- `associations.csv`, `feature_summary.csv`, `joint_inventory.csv`, `review_top70.csv`: 관계·분포·관절별 집계·검토 목록.',
        '- `summary.json`, `side_diagnostic.json`, `verification.json`: 결과와 독립 검산.',
        '- `verification_correction.json`: 검산 중 발견한 정밀도 차이 수정 근거.',
        '- 재현: 기존 Python3.12 환경에서 `python -B run.py` → `python -B side_diagnostic.py` → `python -B verify.py` → `python -B present.py`. 완료된 실행 결과는 덮어쓰지 않는다.','']
    (HERE/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote report.md and diagnostic figure')


if __name__=='__main__':main()
