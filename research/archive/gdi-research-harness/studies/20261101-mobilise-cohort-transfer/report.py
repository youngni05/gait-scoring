from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==40
    assert sha(HERE/'verify.py')==v['verify_sha256'] and sha(HERE/'analyze.py')==v['analysis_sha256']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');cm=pd.read_csv(HERE/'cohort-metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 질환 집단을 통째로 제외한 보행 점수화 진단','',
      '**결론: 회전 정보의 추가 가치는 유지됐지만, 질환이 바뀌면 확률과 경고의 신뢰성이 떨어졌다.** 회전 포함 모델의 확률 오차는 기존 혼합질환 평가 대비 SPPB0.5770→0.6094,TUG0.5353→0.5634로 초기값5개 모두 증가했다. 반면 새 질환 제외 조건 안에서 회전 정보 추가는 확률 오차와 놓침을5개 모두 줄였다. 보행·회전 정보의 신호는 남지만 정확한 확률과 공통 경고 문턱의 일반화에는 한계가 있다.','',
      '전체 놓침이 SPPB43.63→40.26%,TUG35.31→34.51%로 줄어든 것만으로 개선이라고 볼 수 없다. 오경고가 각각9.95→11.25%,10.03→10.39%로 늘었고,PFF에서는 놓침이 줄어드는 동시에 오경고가23.64→31.25%,21.77→27.00%로 늘었다. 질환별로 경고가 치우쳐 전체 평균이 실제 문제를 가린다.','',
      '같은 질환·비슷한 속도 구분은 회전 포함 모델에서 SPPB65.33→63.35%,TUG70.17→72.58%였다. 다음 실험은 최종 RF를 고정하고 학습 내부에서도 질환 전체 제외로 보정용 예측을 만들어 이러한 확률·경고 편향이 줄어드는지 확인한다. 이는 결과를 보고 정한 새 탐색 가설이며 이미 효과가 확인된 방법은 아니다.','',
      '질문은 현재 보행·회전 특징의 예측 정보가 학습하지 않은 질환 집단에서도 유지되는가다. 기존 혼합 질환의 사람5분할 예측과,한 질환 전체를 제외하는4개 새 평가 조건을 같은 사람·방문에서 비교했다. SPPB2,174명/TUG1,939명과각각7,356/5,688방문,30–60초5구간,초기값5개를 유지했다.','',
      'MixedCore/MixedTurn은 기존 사람 분리 평가 예측을 재사용한다. LocoCore/LocoTurn은 COPD/MS/PD/PFF 중 평가 대상 질환 전체를 제외하고 나머지3개 질환으로만 RF·정규화·확률 보정·경고 문턱을 학습했다. Core는기존4개×5구간20열,Turn은회전 빈도를추가한25열이다. 질환명은예측입력이아니다. RF128나무·최소잎20·max_features1을고정하고추가설정탐색은하지않았다.','',
      '**평가 방식이 달라지는 스트레스 검사다.** 사람·방문은 같지만 학습 사람 수와 질환 구성이 함께 달라지므로 차이를 질환의 순수 인과효과로 해석하지 않는다. 새 외부 자료를 확보한 임상 검증도 아니다.','',
      '## 전체 사람 평균','',
      '| 목표 | 방법 | Brier ↓ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: |']
    for target in ['sppb','tug']:
        for method in METHODS:
            name=method+'_cal';q=m[m.target.eq(target)&m.method.eq(name)].mean(numeric_only=True)
            order=mt[mt.target.eq(target)&mt.method.eq(name)].pair_order_accuracy.mean()
            lines.append(f'| {target.upper()} | {method} | {q.brier:.4f} | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {order*100:.2f}% |')
    lines+=['','## 질환별 경고','',
      '저기능은SPPB≤3점/TUG>20초다. 놓침·오경고는 학습 내부 오경고10% 목표로 정한 별도 경고 규칙을 평가에 그대로 적용한 값이다. 저기능 사람이 적은 질환에서는 몇 명의 변화로 비율이 크게 바뀔 수 있다.','',
      '| 목표 | 질환 | 전체 사람 | 저기능 사람 | 방법 | 놓침 | 오경고 | Brier |','| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |']
    for target,cohort in itertools.product(['sppb','tug'],COHORTS):
        for method in ['MixedTurnRF_cal','LocoTurnRF_cal']:
            q=cm[cm.target.eq(target)&cm.cohort.eq(cohort)&cm.method.eq(method)].mean(numeric_only=True)
            lines.append(f'| {target.upper()} | {cohort} | {int(q.people)} | {int(q.low_people)} | {method} | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {q.brier:.4f} |')
    lines+=['','값은 각 사람에게 동일 비중을 준 초기값5개 평균이다. 같은 질환·속도 짝은 원래 사람5분할 번호도 같은 다른 사람을0.05m/s 이내로 연결했다. 혼합 모델의 서로 다른 보정 분할끼리 짝짓지 않고 모든 모델에서 동일한 짝을 사용했다. 비교 대상의 반복 사용을 허용하므로 이 비율은 모집단 AUROC가 아니다.','',
      '보정과 경고 문턱에 제외 질환의 정답을 사용하지 않았다. 내부3분할도 남은 질환의 사람만 포함한다. 과거의 혼합 질환 모델에서는 평가 사람은 제외했지만 같은 질환의 다른 사람은 학습에 포함됐다. 현재의 차이는 이러한 평가 질문의 차이를 보여준다.','',
      f'검산40작업, 모델·평가부분집합 재현{v["model_replays"]}회, 지정 재학습{v["refits"]}개, 독립 지표 최대 차이{v["independent_metric_error"]:.3g}.',
      f'작은 결과: {HERE}',f'모델·대량 예측: {STORE}','',
      '[질환별 초기값 결과](cohort-metrics-by-seed.csv) · [전체 초기값 결과](metrics-by-seed.csv) · [사람 재표집](paired-bootstrap.json) · [속도 짝](matched-by-seed.csv) · [검산](main-verification.json) · [조건과 사람 수](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('REPORT_READY',flush=True)
