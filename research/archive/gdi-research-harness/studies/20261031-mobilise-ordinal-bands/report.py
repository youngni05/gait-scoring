"""Render verified tables. Conclusions are written after reviewing complete results."""
from study import *

LABELS={'Prior':'학습자료 구간 빈도','SVRreg_point':'기존 SVR 점수를 구간으로 변환','RFreg_point':'기존 단일 초기값 RF 회귀 점수','RFreg5_point':'RF 회귀 점수 · 초기값 5개',
        'SVRreg_raw':'SVR + 확률 변환','SVRreg_cal':'SVR + 확률 변환·보정','RFreg_raw':'기존 단일 초기값 RF + 확률 변환','RFreg_cal':'기존 단일 초기값 RF + 확률 변환·보정',
        'RFreg5_raw':'RF 회귀 + 확률 변환','RFreg5_cal':'RF 회귀 + 확률 변환·보정','RF_raw':'RF 분류','RF_cal':'RF 분류 + 보정','OrdinalRF_raw':'순서형 RF','OrdinalRF_cal':'순서형 RF + 보정'}
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def val(v,percent=False):return f'{v*100:.2f}%' if percent else f'{v:.4f}'
def interval(v,percent=False):return f'{val(v.mean(),percent)} [{val(v.min(),percent)}, {val(v.max(),percent)}]'

def run():
    check=json.loads((HERE/'verification.json').read_text());assert check['status']=='passed'
    m=pd.read_csv(HERE/'metrics-by-seed.csv');o=pd.read_csv(HERE/'low-function-by-seed.csv');c=pd.read_csv(HERE/'class-probabilities-by-seed.csv')
    p=m[m.band.eq('long')&m.k.eq(5)];q=o[o.band.eq('long')&o.k.eq(5)];cls=c[c.band.eq('long')&c.k.eq(5)]
    parts=['# Mobilise-D: 네 점수 구간의 확률 예측 비교',
           '**최종 판단: 네 구간의 확률에는 보행 정보가 담겨 있지만, 순서형 분류가 기존 방법을 전반적으로 개선하지는 못했다.** SPPB는 회귀 RF에 확률 변환을 붙인 방식이 낫고, TUG에서는 순서형 RF가 회귀 RF보다 확률 오차를 줄였다. 일반 RF 분류와 순서형 RF의 차이는 작다. 저기능 놓침을 크게 줄이려면 여전히 오경고를 많이 허용해야 한다.',
           '대표 조건에서 보정 후 Brier는 SPPB 회귀 RF 0.6022 → 순서형 RF 0.6089로 나빠졌고, TUG는 0.5905 → 0.5844로 좋아졌다. 두 방향 모두 초기값 5개 전체에서 유지됐다. 순서형의 TUG RPS도 0.1439 → 0.1407로 좋아졌지만, 일반 RF 분류의 0.1405보다 좋지는 않았다. 순서형−일반 RF의 Brier·RPS 차이는 사람 재표집 95% 구간이 0을 포함한다.',
           '**극단 오판 감소는 입증되지 않았다.** 기존 회귀 점수를 그대로 구간화한 방식과 비교하면 전체 양끝 오판이 SPPB 0.34% → 순서형 2.15%, TUG 1.77% → 4.84%로 늘었다. 회귀값이 중간 구간에 모여 극단 구간 선택이 적다는 성질과, 확률의 최댓값 구간을 고르는 규칙의 차이를 함께 반영한 결과다. 회귀 RF에 확률 변환을 붙인 비교군과는 TUG에서 양끝 오판이 5.28% → 4.84%로 줄지만 SPPB에서는 1.77% → 2.15%로 늘었다. 확률 오차 개선과 단일 구간 판정의 안전성을 동일시하면 안 된다.',
           '안쪽 오경고 10% 기준에서 순서형 RF는 SPPB 저기능의 53.27%, TUG 저기능의 50.71%를 여전히 놓쳤다. 실제 오경고는 각각 9.82%, 9.73%였다. 회귀 RF의 놓침 53.90%, 50.98%와 큰 차이가 아니며, 놓침 차이의 95% 구간은 모두 0을 포함한다. 안쪽 감지율 80% 기준으로 바꾸면 놓침은 18.84%, 18.12%가 되지만 오경고가 34.50%, 30.67%로 늘었다.',
           '구간별 성능도 균일하지 않다. SPPB 7–9점 구간에서 순서형 RF의 Brier는 0.2272로 구간 빈도 기준선 0.2253보다 나빴다. TUG는 네 구간 모두 기준선보다 작았지만, 10–15초 구간은 0.2058 → 0.2031로 개선 폭이 작았다. 전체 확률 오차 감소를 모든 구간의 좋은 구분 능력으로 해석하지 않는다. 긴 구간 여러 개의 이점이 더 뚜렷했고, 짧은 구간의 성능은 기준선에 가까웠다.',
           '**현재 사용할 출력 후보는 네 확률과 저기능 가능성을 함께 보여 주는 연구용 출력이다.** 최댓값 구간 하나를 확정 판정으로 내놓는 것은 보류하는 편이 타당하다. SPPB에서는 회귀 기반 확률을 기준으로 유지하고, TUG에서는 일반 RF와 순서형 RF를 함께 후보로 두되 독립 집단에서 비교해야 한다. 이번 결과로 영상에서의 정확도나 임상 사용 가능성을 주장할 수 없다.',
           '보행 특징으로 정확한 한 점수 대신 네 구간에 속할 확률을 예측한다. 모든 결과는 평가 대상 사람의 기록을 제외해 학습한 바깥 평가 예측이다. 대표 조건은 결과 확인 전에 고정한 **30–60초 구간 5개**다. 각 표는 초기값 5개의 평균이며, 범위 표시는 다섯 결과의 최솟값·최댓값이다. 기존 SVR과 기존 단일 초기값 RF를 반복 5회 학습한 것으로 세지 않는다.',
           '## 무엇을 비교했는가',
           '- SPPB: **0–3 / 4–6 / 7–9 / 10–12점**. 낮을수록 저기능이다.\n- TUG: **10초 이하 / 10초 초과–15초 이하 / 15초 초과–20초 이하 / 20초 초과**. 길수록 저기능이다.\n- 이는 비교용 연구 구간이다. 네 구간 자체를 검증된 임상 위험 등급으로 해석하지 않는다.',
           'SPPB 2,174명·7,356방문, TUG 1,939명·5,688방문을 유지했다. 같은 사람의 모든 방문을 같은 바깥 분할에 배치했다. 4–10초와 30–60초 보행 구간을 각각 1·3·5개 사용했다. 정확한 길이 범위는 [4,10), [30,60)초다. 기존에 선택한 draw0 구간과 안쪽 3분할·바깥 5분할을 그대로 사용했다. 여러 방문을 가진 사람의 가중치 합이 한 사람 분량이 되게 계산했다. 연속형 SPPB 회귀값은 반올림 없이 3.5·6.5·9.5를 경계로 구분했다.',
           '입력은 구간별 보행 빈도·주기 시간·속도·보폭의 중앙값 4개다. RF 분류는 네 정답을 직접 학습한다. 순서형 RF는 세 경계를 각각 넘는지를 함께 학습한다. 두 모델 모두 128개 트리·최소 잎 표본 20개로 맞췄다. 순서형 RF는 같은 트리의 잎을 공유해 경계 확률의 순서가 뒤집히지 않는다. 시간축 모델이나 원본 구간 입력을 추가한 실험은 아니다.',
           '회귀 RF는 기존 안쪽 검증에서 선택한 설정을 보존하고 초기값 5개로 다시 학습했다. 기존 단일 초기값 결과도 별도 보존했다. SVR은 무작위 초기값이 없는 기존 예측을 재사용했다. 회귀값을 확률과 비교할 수 있도록 학습 사람의 안쪽 평가 예측으로만 네 구간 확률 변환기를 학습했다. 모델 구조 비교에는 RF 분류와 순서형 RF가 더 직접적이며, 회귀 RF는 기존에 선택된 잎 크기가 다를 수 있다.',
           '확률 보정은 학습자료의 안쪽 평가 예측에서 온도 한 개를 선택했다. 평가 대상 정답으로 설정·보정·경고 기준을 고르지 않았다. 이 보정은 최댓값인 구간을 바꾸지 않으므로, 보정 전후 구간 오판율은 같다. 내부 결과의 재사용은 모델 선택용이며, 성능은 바깥 평가 예측으로만 집계했다.',
           '## 전체 확률과 구간 예측',
           '**Brier**는 네 확률의 제곱 오차 합(0–2), **RPS**는 순서가 있는 세 경계 확률의 평균 제곱 오차(0–1)다. 둘 다 작을수록 좋다. RPS는 멀리 떨어진 구간에 확률을 두는 오류를 더 불리하게 반영한다. 적중률만 높이고 드문 저기능을 무시하는지 별도로 확인한다.']
    probability_methods=['Prior','SVRreg_raw','SVRreg_cal','RFreg5_raw','RFreg5_cal','RF_raw','RF_cal','OrdinalRF_raw','OrdinalRF_cal']
    for target in ['sppb','tug']:
        rows=[]
        for name in probability_methods:
            g=p[p.target.eq(target)&p.method.eq(name)];rows.append([LABELS[name],val(g.brier.mean()),val(g.rps.mean()),val(g.log_loss.mean()),val(g.accuracy.mean(),True)])
        parts.extend(['### '+target.upper(),table(['방법','Brier ↓','RPS ↓','로그 손실 ↓','구간 적중률 ↑'],rows)])
    parts.extend(['## 극단 오판과 저기능 놓침',
                  '‘양끝 오판’은 전체 평가 기록 중 첫 구간과 마지막 구간을 서로 바꿔 예측한 비율이다. ‘저기능→최상 구간’과 ‘저기능 놓침’은 실제 저기능(SPPB 0–3점 / TUG >20초)에 조건을 건 비율이다. 저기능 놓침은 가장 높은 확률의 구간을 하나 고르는 규칙의 결과다. 저기능 경고 확률의 문턱을 따로 낮추는 결과는 다음 표에서 구분한다. 각 비율은 해당 평가 집단 안에서 사람별 동일 가중치다.'])
    error_methods=['SVRreg_point','RFreg5_point','SVRreg_cal','RFreg5_cal','RF_cal','OrdinalRF_cal']
    for target in ['sppb','tug']:
        rows=[]
        for name in error_methods:
            g=p[p.target.eq(target)&p.method.eq(name)];rows.append([LABELS[name],val(g.two_or_more_bands.mean(),True),val(g.opposite_extreme.mean(),True),val(g.low_to_best.mean(),True),val(g.low_miss.mean(),True),val(g.argmax_fpr.mean(),True)])
        parts.extend(['### '+target.upper(),table(['방법','2구간 이상 오판 ↓','양끝 오판 ↓','저기능→최상 구간 ↓','저기능 놓침 ↓','비저기능에 잘못 경고 ↓'],rows)])
    parts.extend(['## 같은 경고 조건에서 저기능을 얼마나 놓쳤는가',
                  'FPR10은 안쪽 검증에서 비저기능의 잘못 경고를 10% 이내로 제한하는 기준이다. Sens80은 안쪽 검증에서 저기능 80% 이상을 잡는 기준이다. 바깥 평가에서 그 비율이 보장되는 것은 아니다. 표의 잘못 경고는 **실제 비저기능 중 경고한 비율**이다. “경고받은 사람 중 오경고 비율”과 다르며 후자는 원자료 표의 false_discovery 열에 보존했다.'])
    for target in ['sppb','tug']:
        rows=[]
        for policy in ['fpr10','sens80']:
            for name in ['SVRreg_cal','RFreg5_cal','RF_cal','OrdinalRF_cal']:
                g=q[q.target.eq(target)&q.method.eq(name)&q.policy.eq(policy)]
                rows.append([policy,LABELS[name],interval(g['miss'],True),interval(g.fpr,True)])
        parts.extend(['### '+target.upper(),table(['내부 기준','방법','저기능 놓침: 평균 [범위] ↓','잘못 경고: 평균 [범위] ↓'],rows)])
    parts.extend(['## 각 구간 확률의 정확도',
                  '아래는 보정 후 확률의 구간별 Brier와 ECE10(확률을 10칸으로 나눠 예측 확률과 실제 빈도의 차이를 합한 값)이다. 둘 다 작을수록 좋다. 드문 구간은 낮은 확률만 출력해도 Brier가 작아질 수 있어 빈도만 출력하는 기준선과 비교해야 한다. ECE만으로 분류 능력이나 개인별 신뢰도를 입증하지 않는다.'])
    for target in ['sppb','tug']:
        rows=[]
        for j in range(4):
            for name in ['Prior','RFreg5_cal','RF_cal','OrdinalRF_cal']:
                g=cls[cls.target.eq(target)&cls['class'].eq(j)&cls.method.eq(name)]
                rows.append([j+1,LABELS[name],val(g.prevalence.mean(),True),val(g.brier.mean()),val(g.ece10.mean(),True)])
        parts.extend(['### '+target.upper(),table(['구간','방법','실제 비중','해당 구간 Brier ↓','ECE10 ↓'],rows)])
    parts.extend(['## 여섯 보행 조건과 초기값 안정성','표는 보정 후 결과다. “5개 중 개선”은 순서형 RF의 지표가 같은 초기값의 비교 모델보다 작았던 횟수다. 그 자체로 새로운 사람에서도 개선됨을 보장하지 않는다.'])
    rows=[]
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        g=m[m.target.eq(target)&m.band.eq(band)&m.k.eq(k)].pivot(index='seed',columns='method',values='rps')
        rows.append([target,band,k,val(g.RFreg5_cal.mean()),val(g.RF_cal.mean()),val(g.OrdinalRF_cal.mean()),f'{int((g.OrdinalRF_cal<g.RF_cal).sum())}/5',f'{int((g.OrdinalRF_cal<g.RFreg5_cal).sum())}/5'])
    parts.append(table(['목표','길이','구간 수','회귀 RF RPS','RF 분류 RPS','순서형 RPS','RF 대비 개선','회귀 RF 대비 개선'],rows))
    comparisons=json.loads((HERE/'paired-bootstrap.json').read_text())['comparisons'];rows=[]
    for r in comparisons:
        if r['variant']!='cal' or r['baseline'] not in ['RF','RFreg5'] or r['metric'] not in ['brier','rps','low_fpr10_miss','opposite_extreme']:continue
        factor=100 if r['metric'] in ['low_fpr10_miss','opposite_extreme'] else 1
        rows.append([r['target'],r['baseline'],r['metric'],f"{r['difference']*factor:+.4f}",f"[{r['ci95'][0]*factor:+.4f}, {r['ci95'][1]*factor:+.4f}]",f"{r['lower_error_seeds']}/5"])
    parts.extend(['### 대표 조건의 차이와 불확실성','차이는 순서형−비교 모델이다. 음수일수록 순서형이 좋다. 오류율 차이는 %p다. 사람마다 다섯 초기값의 차이를 평균한 뒤 사람 단위로 2,000회 재표집했다. 이는 고정한 모델·기존 분할에 조건을 둔 95% 구간이며 재학습 불확실성·다중 비교 보정·독립 외부 검증을 포함하지 않는다.',table(['목표','비교 모델','지표','차이','조건부 95% 구간','개선 초기값'],rows)])
    sizes={folder:sum(f.stat().st_size for f in (STORE/folder).rglob('*') if f.is_file()) for folder in ['models','folds','regression','regression-five']}
    parts.extend(['## 검산·재현·저장',
                  f"검산 상태: **{check['status']}**. {check['ledgers']}개 완료 기록, {check['final_model_or_mapper_replays']}개 최종 모델·확률 변환기와 추가 300개 회귀 모델을 다시 불러 예측했다. 사전에 고른 셀에서 내부·최종 학습 {check['selected_refits']}회를 재실행했다. 예측 재계산 최대 차이 {check['max_prediction_replay_error']:.3g}, 독립 지표 재계산 최대 차이 {check['max_independent_metric_error']:.3g}; 절대 허용오차 {check['absolute_tolerance']:.0e}를 유지했다.",
                  '코드·프로토콜·비교표·검산 요약은 C에, 모델·구간별 예측·내부 배열·개인별 손실·확률 보정 상세 표는 D에 저장했다.',
                  f'- 작은 산출물: `{HERE}`\n- 큰 산출물: `{STORE}`\n- D 주요 폴더 합계: {sum(sizes.values())/1024**3:.3f} GiB',
                  '실행 순서: study.py prepare → study.py main → study.py regression → regression_five.py → analyze.py → verify.py → report.py. 원본 준비 데이터와 기존 회귀 모델이 필요하다. 신규 실험 폴더의 protocol.json, regression-five-protocol.json에 입력 파일·코드 지문과 설정을 기록했다. 마지막 두 단계는 검산 후 보고서를 생성한다.',
                  '## 해석 한계',
                  '이 결과는 센서에서 계산한 보행 특징의 내부 사람 분리 평가다. 같은 개발용 분할을 여러 후속 실험에서 사용했으므로 새 사람 집단의 확증 시험이 아니다. 원본 영상·HMR2 오차·촬영 방향의 영향을 검증하지 않았다. 구간 확률 예측이 좋아져도 보행만으로 SPPB 전체나 TUG를 직접 측정했다는 뜻은 아니다. 이번 실험은 판단 보류를 새로 평가하지 않았으며, 확률 보정만으로 분포 밖 대상을 탐지하거나 안전한 자동 판정을 보장하지 않는다.'])
    (HERE/'README.md').write_text('\n\n'.join(parts)+'\n',encoding='utf8')
    print('REPORT_DONE',flush=True)

if __name__=='__main__':run()
