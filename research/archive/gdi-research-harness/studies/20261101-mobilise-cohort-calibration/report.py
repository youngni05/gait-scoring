from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==40 and v['model_replays']==160 and v['refits']==8
    assert v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py')
    m=pd.read_csv(HERE/'metrics-by-seed.csv');cm=pd.read_csv(HERE/'cohort-metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 질환 변화 상황을 재현한 확률 보정과 경고 문턱','',
      '**결론: 보정 방식만 바꿔 놓침과 오경고를 함께 줄이지 못했다.** 질환 단위 보정은 SPPB 놓침40.26→35.52%,TUG34.51→32.51%로 줄었지만 오경고는11.25→11.94%,10.39→11.09%로 늘었다. 확률 오차도 SPPB에서는 조금 감소했으나 TUG에서는 증가했다. 사전에 정한 두 목표의 공동 성공 기준을 만족하지 못했다.','',
      '보수적 문턱은 오경고를8.98%/9.12%로 낮췄지만 놓침이44.13%/37.17%로 늘었다. 이는 경고를 적게 내보내는 대가이지 보행 정보가 더 좋아진 결과가 아니다. 다음 진단은 같은 사람의 서로 다른 방문에서도 보행이 점수 차이를 구분하는지 확인해 사람 간 고정 차이에 의존한 신호인지 살펴본다.','',
      '초기값5개 모두에서 새 질환 보정은 놓침 감소·오경고 증가를 보였다. 보수적 문턱은5개 모두에서 반대로 오경고 감소·놓침 증가를 보였다. Brier는SPPB5/5개선,TUG0/5개선이고 순서까지 반영한RPS는 두 목표 모두5/5악화했다. 초기값에 따른 우연한 방향 변화로 공동 개선이 가려진 결과는 아니었다.','',
      '최종 RF와 입력은 그대로 두고, 확률을 보정하고 경고 문턱을 정하는 내부 검증 방식을 비교했다. 기존에는 같은 질환의 다른 사람이 내부 학습에 포함됐지만 새 방식은 보정용 예측을 만들 때도 질환 하나를 통째로 제외한다. 평가 질환의 정답은 어느 설정에도 쓰지 않는다.','',
      'SPPB2,174명/TUG1,939명,같은 방문·30–60초5구간·회전25열·질환전체제외4분할·초기값5개를 유지했다. RawRF는 무보정,PersonCal은 기존 사람 내부검증 보정,CohortCal은 새 질환 내부검증 보정이다. CohortGuard는 CohortCal과 같은 확률에 학습 세 질환의 오경고10% 문턱 중 가장 높은 값을 적용한다. 새로운 질환의 오경고가10% 이하임을 보장하지 않는다.','',
      '| 목표 | 방법 | Brier ↓ | 놓침 ↓ | 오경고 ↓ | 경고율 | 같은 질환·속도 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for target,method in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(method)].mean(numeric_only=True);pair=mt[mt.target.eq(target)&mt.method.eq(method)].pair_order_accuracy.mean()
        lines.append(f'| {target.upper()} | {method} | {q.brier:.4f} | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {q.warning_rate*100:.2f}% | {pair*100:.2f}% |')
    lines+=['','확률 보정은 가장 가능성이 높은 점수 구간을 바꾸지 않는다. 따라서 구간 적중률·극단 구간 오판은 동일하며, 경고 문턱의 변화와 구분해야 한다. 전체 값은 사람에게 같은 비중을 준 초기값5개 평균이다. 저기능은SPPB≤3점/TUG>20초다.','',
      '## 질환별 경고','', '| 목표 | 질환 | 저기능 사람 | 방법 | 놓침 | 오경고 |','| --- | --- | ---: | --- | ---: | ---: |']
    for target,cohort,method in itertools.product(['sppb','tug'],COHORTS,['PersonCal','CohortCal','CohortGuard']):
        q=cm[cm.target.eq(target)&cm.cohort.eq(cohort)&cm.method.eq(method)].mean(numeric_only=True)
        lines.append(f'| {target.upper()} | {cohort} | {int(q.low_people)} | {method} | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% |')
    lines+=['','개발 평가 결과를 보고 다음 가설을 정한 탐색 연구다. 세 학습 질환만으로 보정하고, 내부 숲은 두 질환·최종 숲은 세 질환으로 학습해 분포 차이가 남는다. COPD/PD의 심한 사례는 소수다. 새 영상·임상 표본에서의 유효성은 확인하지 않았다.','',
      f'검산40작업,저장모델 재추론{v["model_replays"]}회,지정 재학습{v["refits"]}개,독립 지표 최대 차이{v["independent_metric_error"]:.3g}.',
      f'작은 산출물:{HERE}',f'모델·대량 예측:{STORE}','',
      '[조건](protocol.json) · [초기값별 지표](metrics-by-seed.csv) · [질환별 지표](cohort-metrics-by-seed.csv) · [사람 재표집](paired-bootstrap.json) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_READY',flush=True)
