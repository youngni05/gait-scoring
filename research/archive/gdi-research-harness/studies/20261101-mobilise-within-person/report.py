from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==40 and v['model_replays']==80
    assert v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py')
    m=pd.read_csv(HERE/'metrics-by-seed.csv');p=json.loads((HERE/'protocol.json').read_text())
    lines=['# 같은 사람의 방문 간 점수 차이를 보행으로 구분하는가','','사람 간 점수 예측에 사용한 RF를 고정하고, 같은 사람이 서로 다른 방문에서 받은 점수 구간의 차이도 구분하는지 확인했다. 새 학습이나 문턱 조정은 하지 않았다. 모든 방문의 해당 사람·질환은 원래 모델 학습에서 제외됐다.','',
      'SPPB는1,834명의11,055방문쌍,TUG는1,653명의6,721방문쌍을 평가한다. 방문쌍은 독립 사람이 아니다. 사람마다 같은 비중을 주고 초기값5개를 유지했다. 방문번호를 정렬해 두 기록을 구별했으며 시간 간격이나 임상적 호전·악화 속도를 가정하지 않는다.','',
      '| 목표 | 방법 | 구간 차이 방향 적중 ↑ | 구간 차이 MAE ↓ | 저기능 전환 양쪽 경고 일치 ↑ | 거짓 저기능 경고 전환 ↓ |','| --- | --- | ---: | ---: | ---: | ---: |']
    for target,name in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(name)].mean(numeric_only=True);fmt=lambda value:f'{value:.4f}' if pd.notna(value) else '해당 없음';pct=lambda value:f'{100*value:.2f}%' if pd.notna(value) else '해당 없음'
        lines.append(f'| {target.upper()} | {name} | {pct(q.direction)} | {fmt(q.get("class_delta_mae"))} | {pct(q.get("correct_low_switch"))} | {pct(q.get("false_low_switch"))} |')
    lines+=['','방향 적중은 실제 점수 구간이 다른 방문쌍에서 계산하며 예측 동률은0.5다. Speed는 속도 변화 방향만 보는 비교군이라 점수 차이 MAE나 보정 경고가 없다. NoChange는 변화가 없다고 두는 기준선이다. RF는4구간 확률의 기대 구간 차이를 쓰며 기존 경고 문턱을 그대로 적용했다.','',
      '저기능 전환 양쪽 경고 일치는 한 방문만 저기능일 때 두 방문의 경고가 모두 정답과 맞는 비율이다. 단일 방문 민감도와 다르다. 거짓 전환은 실제 저기능 여부가 같은데 경고가 바뀐 비율이다. SPPB≤3/TUG>20초 경계는 유지했다.','',
      '점수 구간 변화에는 검사 오차도 포함될 수 있다. 센서 입력에서의 탐색 진단이며 영상 측정 정확도·임상적 변화 검출·인과효과를 검증한 결과는 아니다.','',
      f'검산40작업·모델80개재추론·독립지표 최대 차이{v["independent_metric_error"]:.3g}.',f'대량 방문쌍:{STORE}','',
      '[조건과 사람 수](protocol.json) · [초기값별 지표](metrics-by-seed.csv) · [질환별 지표](cohort-metrics-by-seed.csv) · [사람 재표집](paired-bootstrap.json) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_READY',flush=True)
