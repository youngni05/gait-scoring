from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==50
    assert sha(HERE/'verify.py')==v['verify_sha256'] and sha(HERE/'analyze.py')==v['analysis_sha256']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 구간별 회전 입력에서 일반 RF와 순서형 RF 비교','',
      '이진 직접 분류와 보정만의 변경이 개선을 만들지 못해 네 점수 구간의 순서 정보를 학습하는 방식을 확인했다. 같은25열 입력(기존4개+회전 빈도,30–60초5구간),SPPB2,174명/TUG1,939명,사람5분할/안쪽3분할/초기값5개를 유지했다.','',
      '두 숲은128나무·최소잎20·max_features1·사람 동일 학습 비중으로 같다. 일반 RF는4개 구간을 분류한다. 순서형 RF는 세 경계를 넘는지를 함께 학습한 뒤 차이를 이용해4개 확률로 바꾼다. 같은 잎에서 세 출력을 계산하므로 누적 확률의 순서가 보장된다. 정규화·확률 보정·문턱은 학습 안에서만 계산한다. 신규 설정 탐색은 하지 않았다.','',
      '| 목표 | 방법 | Brier ↓ | RPS ↓ | 구간 적중 ↑ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for t in ['sppb','tug']:
        for method in METHODS:
            name=method+'_cal';q=m[m.target.eq(t)&m.method.eq(name)].mean(numeric_only=True)
            pair=mt[mt.target.eq(t)&mt.method.eq(name)&mt.matching.eq('speed_cohort')].pair_order_accuracy.mean()
            lines.append(f'| {t.upper()} | {method} | {q.brier:.4f} | {q.rps:.4f} | {q.accuracy*100:.2f}% | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {pair*100:.2f}% |')
    lines+=['','Brier와 RPS'+'는 확률 오차다. 낮을수록 좋으며 RPS는 순서가 멀리 틀린 오류를 반영한다. 놓침·오경고는 내부 오경고10% 목표로 정한 별도 경고 규칙의 실제 평가 값이다. 값은 초기값5개의 사람 동일 비중 평균이다.','',
      '같은 질환·속도 구분은 평가자료 안에서 같은 질환·평가 분할의 다른 사람을0.05m/s 이내로 연결하고 저기능 쪽 확률이 높은지 계산했다. 동률은0.5이며 모집단 AUROC나 구간 적중률이 아니다.','',
      '같은 개발 평가 사람을 여러 번 사용한 탐색 결과다. 순서형 RF는 이전 중앙값4개 입력에서도 시험했지만 이번에는 구간별25열 입력에 고정해 비교했다. 새 임상 영상·외부 표본의 유효성은 확인하지 않았다. paired-bootstrap.json의 조건부 사람 재표집 구간과 초기값별 방향을 함께 확인한다.','',
      f'검산50작업, 모델 재현{v["model_replays"]}개, 지정 재학습{v["refits"]}개, 독립 지표 최대 차이{v["independent_metric_error"]:.3g}.',
      f'작은 산출물: {HERE}',f'모델·대량 예측: {STORE}','',
      '[초기값별 결과](metrics-by-seed.csv) · [사람 재표집](paired-bootstrap.json) · [속도 짝](matched-by-seed.csv) · [검산](main-verification.json) · [조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('REPORT_READY',flush=True)
