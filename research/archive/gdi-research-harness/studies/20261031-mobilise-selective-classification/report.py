"""Readable results from verified predictions; no model/threshold selection here."""
from study import *

def run():
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    m=pd.read_csv(HERE/'metrics-by-seed.csv');q=pd.read_csv(HERE/'probability-by-seed.csv')
    primary=m[m.band.eq('long')&m.k.eq(5)]
    names={'SVR_original':'기존 SVR 점수 기준','RFreg_original':'기존 RF 점수 기준','DeepSets600_original':'기존 Deep Sets 점수 기준',
           'SVRreg_cal_fpr10_r0':'SVR 보정·오경고10% 목표','RFreg_cal_fpr10_r0':'회귀 RF 보정·오경고10% 목표',
           'LR_cal_fpr10_r0':'로지스틱 보정·오경고10% 목표','RF_cal_fpr10_r0':'분류 RF 보정·오경고10% 목표',
           'SVRreg_cal_sens80_r0':'SVR 보정·검출80% 목표','RF_cal_sens80_r0':'분류 RF 보정·검출80% 목표',
           'RF_cal_sens80_r10':'분류 RF 검출80% 목표·보류10%','RF_cal_sens80_r20':'분류 RF 검출80% 목표·보류20%'}
    lines=['# Mobilise-D 저기능 분류·확률 보정·판단 보류 실험', '',
           '기존 보행 구간 네 특징으로 SPPB 3점 이하와 TUG 20초 초과를 예측했다. 아래 수치는 임상 진단 기준이 아니라 기존 실험의 분석 구분선에 대한 결과다.', '',
           '사람·방문·구간·외부 5분할과 초기값 5개를 유지했다. 모든 방문은 사람 단위로 묶었다. 짧은 [4,10)초와 긴 [30,60)초 구간을 각각 1·3·5개 사용했으며, 기존 RF/SVR와 동일한 구간별 특징의 중앙값 네 개만 입력했다. 주 분석은 기존 관례에 따라 긴 구간 5개, 구간 추출 draw0으로 사전 지정했다.', '',
           '## 긴 구간 5개: 초기값 5개 평균', '',
           '놓침은 저기능 대상을 정상으로 답한 비율, 오경고는 나머지 대상을 저기능으로 답한 비율이다. 보류를 정상으로 세거나 평가에서 숨기지 않았다. 저기능의 찾음·놓침·보류 합은 100%다. 사람마다 같은 비중을 주고, 실제 저기능/나머지 방문 집단 안에서 다시 사람별 비중을 맞췄다.', '']
    for target,title in [('sppb','SPPB 3점 이하'),('tug','TUG 20초 초과')]:
        lines += ['### '+title,'','| 방법 | 찾음 | 놓침 | 오경고 | 전체 보류 | 저기능 보류 |','|---|---:|---:|---:|---:|---:|']
        for method,name in names.items():
            r=primary[(primary.target==target)&(primary.method==method)].mean(numeric_only=True)
            lines.append('| '+name+' | '+' | '.join(f'{100*r[c]:.1f}%' for c in ['detect_rate','miss_rate','fpr','defer_rate','positive_defer_rate'])+' |')
    lines+=['','## 6가지 구간 조건: 같은 오경고10% 내부 목표','',
            '목표는 내부 검증에서만 정한다. 아래 실제 외부 분할의 오경고가 10%로 고정되는 것은 아니다. 기준을 평가 정답에 맞춰 다시 조정하지 않았다.','',
            '| 정답 | 구간·개수 | SVR 찾음 / 오경고 | 분류 RF 찾음 / 오경고 | RF 찾음 범위(5개 초기값) |','|---|---|---:|---:|---:|']
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        part=m[(m.target==target)&(m.band==band)&(m.k==k)]
        a=part[part.method.eq('SVRreg_cal_fpr10_r0')];b=part[part.method.eq('RF_cal_fpr10_r0')]
        lines.append(f'| {target.upper()} | {"4–10초" if band=="short" else "30–60초"} × {k} | {a.detect_rate.mean()*100:.1f}% / {a.fpr.mean()*100:.1f}% | {b.detect_rate.mean()*100:.1f}% / {b.fpr.mean()*100:.1f}% | {b.detect_rate.min()*100:.1f}–{b.detect_rate.max()*100:.1f}% |')
    lines+=['','## 확률 보정','',
            'Brier 점수는 예측 확률과 실제 0/1 정답의 제곱 차이이며 낮을수록 좋다. 전체 Brier가 좋아도 드문 저기능을 잘 찾는다는 보장은 없어 검출·오경고와 함께 본다. ECE는 확률을 10개 구간으로 묶은 보정 오차로 보조 지표다.','',
            '| 정답 | 방법 | Brier | ROC AUC | PR AUC | ECE |','|---|---|---:|---:|---:|---:|']
    for target in ['sppb','tug']:
        for method in ['prior','LR_raw','LR_cal','RF_raw','RF_cal','SVRreg_cal','RFreg_cal']:
            r=q[(q.target==target)&q.band.eq('long')&q.k.eq(5)&q.method.eq(method)].mean(numeric_only=True)
            lines.append(f'| {target.upper()} | {method} | {r.brier:.4f} | {r.roc_auc:.3f} | {r.average_precision:.3f} | {r.ece10:.4f} |')
    lines+=['','## 실행·검증과 한계','',
            '- 분류 조건별 300개 작업: 최종 분류기 600개, 내부 분류기 1,800개, 확률 보정기 2,400회 적합. 회귀 대조는 60개 조건에서 기존 최종 모델을 재사용하고 내부 회귀 360개를 재적합했다.',
            '- 분류 RF는 무작위 초기값 5개다. 로지스틱 회귀는 결정적인 계산이고 기존 SVR/RF 회귀도 한 번의 저장 모델을 재사용하므로 5회 독립 반복이라고 해석하지 않는다. 기존 Deep Sets는 실제 초기값 5개의 저장 예측이다.',
            '- 분류기 설정은 결과를 보기 전에 고정했다. 학습 사람마다 같은 비중을 주었고 저기능에 추가 클래스 가중치는 주지 않았다. 확률 보정과 판정·보류 기준은 외부 학습 사람의 내부 교차검증 예측만 사용했다.',
            '- 80% 검출·10% 오경고·10/20% 보류는 내부 운용 목표다. 별도 임상 타당성이나 외부 집단 보장이 아니다. 엄격한 동률 처리를 사용해 실제 보류율이 목표와 다를 수 있다.',
            '- 보류는 경계 근처의 애매함을 나타낸다. 영상 품질 오류나 학습 분포 밖 사례를 식별하는 검증된 장치가 아니며 conformal 포함률 보장도 없다.',
            '- 기존 회귀도 같은 확률 보정·기준 조정을 받는다. 기존 점수 기준만 비교했을 때 생기는 경고 기준 변경 효과와 직접 분류 효과를 분리한다.',
            '- 과거 반복 탐색에 쓰인 개발용 외부 분할을 재사용했다. 독립 최종 평가나 단일 카메라 검증은 아니다. 짧은 기록·극단 정답의 원자료 품질도 새로 확정하지 않았다.',
            '- C에 코드·예측·검증·보고를 저장하고 큰 모델 파일은 D:/gait-research-data/D_P/mobilise-selective-classification-20261031/models에 저장했다. 추가 다운로드는 없었다.',
            '- 수렴 검사 재사용 오류와 Windows 프로세스 오류는 implementation-correction.json에 기록했다. 회귀 대조를 수정·재실행했으며 최종 분할·정답·성능 기준은 바꾸지 않았다.',
            f'- 검산: 분류 확률 최대 차이 {verification["classifier_max_probability_difference"]:.3g}, 회귀 보정 확률 {verification["regression_max_probability_difference"]:.3g}, 지표 {verification["metric_max_difference"]:.3g}. 모든 사람 분리·정답·학습 전용 정규화·내부 판정 기준 재계산 검사를 통과했다.', '',
            '상세 결과: metrics-by-seed.csv, probability-by-seed.csv, decision-summary.csv, calibration-bins.csv, paired-bootstrap.json, verification.json.']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('REPORT_WRITTEN',flush=True)

if __name__=='__main__':run()
