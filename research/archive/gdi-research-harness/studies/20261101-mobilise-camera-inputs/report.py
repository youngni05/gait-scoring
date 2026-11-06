from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and (v['tasks'],v['model_replays'],v['refits'])==(50,800,16)
    assert v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py')
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 고정 휴대폰에 맞춘 속도 중심 입력 비교','',
      '**결론: 속도를 중앙값 하나로 줄이지 않고 구간별로 유지한 것이 주된 개선이었다.** SPPB 적중률46.58→49.05%(+2.47%p),TUG50.77→54.51%(+3.74%p)로5개 초기값 모두 개선됐다. 차이의 사람 재표집95%구간은 각각+1.11~+3.90%p,+2.47~+4.96%p였다. 평가 사람을 반복 활용한 조건부 탐색 결과이며 외부 검증은 아니다.','',
      '걸음 빈도를 추가한 적중률은49.04%/54.41%로 추가 개선이 없었고,개선 초기값은두목표각각2/5였다. 확률 오차는조금줄었으나 저기능경고오경고는늘었다. 같은질환·비슷한속도구분은SPPB55.37→61.65%,TUG66.03→64.41%로방향이달랐다. 기존4개RF의적중률49.43%/54.80%와도차이가작았다.','',
      '구간별 속도RF의 저기능 놓침54.95%/45.49%,오경고9.63%/9.85%였다. 걸음 빈도 추가는54.42%/43.45%와9.97%/10.20%였다. 다음 실험은 같은1/2개특징에서RF와DeepSets및내부적중률설정을비교한다.','',
      '전체 적중률 개선이 모든 구간의 개선은 아니다. 최대확률구간을 하나 고를 때 SPPB0–3점 재현율은속도중앙값15.12%에서속도5개10.84%로낮아졌다. 가장많은구간만예측하는기준선도SPPB47.39%/TUG49.27%이므로 전체49.05%/54.51%를강한분류성능으로해석하지않는다. 별도저기능경고규칙과최대확률구간선택은다르다.','',
      '사용자는 휴대폰 고정·바닥 거리 표시 없음·키1회 입력 가능 조건에서 가장 중요한 값 하나와 영상에서 얻을 만한 추가 값으로4구간 적중률을 높이고 싶다고 정했다. 이번 첫 실험은 속도를 기준축으로,구간 정보 보존과 걸음 빈도 추가 효과를 분리한다.','',
      '**센서 자료의 입력 가치 실험이다. 영상에서 키를 이용해 복원한 속도나 점수의 정확도를 검증한 결과가 아니다.**','',
      '기존 속도 기준선은5개 속도를 중앙값 하나로 줄였다. 새 SpeedBouts는속도5개,SpeedCadence는속도5개+걸음빈도5개,Core4는기존4개×5구간이다. RF128나무·최소잎20을 고정해 모델 구조 차이가 섞이지 않게 했다. 구간 지속 시간·회전 빈도는 제외했다.','',
      'SPPB2,174명/TUG1,939명,기존 사람5분할·내부3분할·초기값5개·30–60초5구간을 유지했다. 모든 방문을 같은 사람 분할에 둔다. 질환 전체 제외4분할 진단과는 평가 질문이 다르다.','',
      '| 목표 | 입력 | 4구간 적중 ↑ | 구간별 평균 적중 ↑ | 양끝 오판 ↓ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target,name in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(name+'_cal')].mean(numeric_only=True);order=mt[mt.target.eq(target)&mt.method.eq(name+'_cal')].pair_order_accuracy.mean()
        lines.append(f'| {target.upper()} | {name} | {q.accuracy*100:.2f}% | {q.balanced_recall*100:.2f}% | {q.extreme_error*100:.2f}% | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {order*100:.2f}% |')
    lines+=['','## 구간별 재현율','','| 목표 | 입력 | 구간1 | 구간2 | 구간3 | 구간4 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for target,name in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(name+'_cal')].mean(numeric_only=True);lines.append('| '+target.upper()+' | '+name+' | '+' | '.join(f'{q[f"recall_{j}"]*100:.2f}%' for j in range(4))+' |')
    lines+=['','주 평가는4구간 적중률이며 구간 경계를 바꾸지 않았다. 저기능 경고는 기존 내부 오경고10% 규칙으로 별도 평가했다. 전체 적중률이 늘어도 특정 구간을 더 놓치는지는 구간별 재현율에서 확인해야 한다.','',
      '보행 빈도와 보행 주기 시간은 관련이 크고,속도·보행주기·이동거리도 독립적인 물리량이 아니다. 평균 수치 간 알고리즘 차이가 모델에 도움이 될 수 있으므로,특징 수 증가를 독립 정보 증가로 단정하지 않는다.','',
      '영상 고유의 좌우 비대칭·걸음별 시간 변동·몸통 흔들림은 현재 보행 구간 평균표에 없으므로 이번 실험에 넣지 않았다. 키 입력은 거리 보정의 단서이며 촬영 각도·가림·추적 오차까지 해결하지 않는다.','',
      f'검산50작업·저장모델재추론800회·지정재학습16개·독립지표최대차이{v["independent_metric_error"]:.3g}.',f'대량자료:{STORE}','',
      '[조건](protocol.json) · [초기값별 지표](metrics-by-seed.csv) · [사람 재표집](paired-bootstrap.json) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_READY',flush=True)
