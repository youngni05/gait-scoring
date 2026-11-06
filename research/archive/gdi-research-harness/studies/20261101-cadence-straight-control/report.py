from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==14 and v['event_replays']==28 and v['period_replays']==224 and v['verify_sha256']==sha(HERE/'verify.py')
    metrics=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'geometry-paired.csv');views=pd.read_csv(HERE/'view-paired.csv');primary=pairs[pairs.method.eq('Reference3D')].iloc[0]
    fmt=lambda x:'계산 불가' if pd.isna(x) else f'{x:.2f}'
    success=primary.common_people>=10 and primary.straight_discrepancy<=5 and primary.difference_straight_minus_turn<=-5
    lines=['# 직선·회전 조건에서 걸음 빈도 대용값 비교','','보행 방향이 바뀌어서 두 계산법의 결과가 달라지는지 확인했다. 과거에 선정한 같은 14명의 직선·회전 2초 구간을 사용하고, 계산법과 품질 기준을 고정했다. 영상에서 발목의 반복 주기를 구하는 방법과 광학 마커에서 발뒤꿈치 사건을 찾는 방법을 비교한다. **두 방법 모두 대용값이며 전문가 발접촉 정답과 대조한 정확도가 아니다.**','',
      f'주 비교의 공동 가용은 {int(primary.common_people)}/14명이다. 같은 사람에서 기준 3D 주기와 사건 대용값의 평균 절대 차이는 직선 {fmt(primary.straight_discrepancy)}, 회전 {fmt(primary.turn_discrepancy)}걸음/분이다. 사전 일관성 탐색 목표는 '+('충족했다.' if success else '충족하지 못했다.'),'',
      '| 조건 | 방법 | 카메라 | 전체 | 사건 품질 | 주기 품질 | 사건 품질·유한 주기 | 양쪽 품질 | 원래 주기 차이 | 품질 통과 차이 |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for _,r in metrics.iterrows():lines.append(f'| {r.condition} | {r.method} | {r.camera} | {r.people} | {r.event_quality} | {r.period_quality} | {r.raw_available} | {r.both_quality} | {fmt(r.raw_discrepancy)} | {fmt(r.quality_discrepancy)} |')
    lines+=['','| 방법 | 카메라 | 두 조건 공동 사람 | 직선 차이 | 회전 차이 | 직선−회전 | 차이가 줄어든 사람 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for _,r in pairs.iterrows():lines.append(f'| {r.method} | {r.camera} | {r.common_people} | {fmt(r.straight_discrepancy)} | {fmt(r.turn_discrepancy)} | {fmt(r.difference_straight_minus_turn)} | {r.improved_people} |')
    lines+=['','| 조건 | RGB 방법 | 사건·유한 주기 공동 | 두 시점 차이 | 양쪽 품질 공동 | 품질 통과 시점 차이 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for _,r in views.iterrows():lines.append(f'| {r.condition} | {r.method} | {r.raw_common_people} | {fmt(r.raw_view_difference)} | {r.both_quality_people} | {fmt(r.quality_view_difference)} |')
    lines+=['','단위는 걸음/분이다. 2초의 1.5주기 품질 기준은 추정값 약 91.53걸음/분 미만을 구조적으로 거부한다. 선택 효과를 피하려고 이번 주 비교는 사건 품질을 통과하고 주기가 계산 가능한 모든 사람을 쓰며, 원래 주기 품질 통과 결과도 따로 표시했다. 품질 기준을 완화한 것은 아니다.','',
      '원래 14명은 이전 27명과 다른 표본이다. 이번에는 같은 14명 안에서 사람·길이를 맞췄으며, 2초가 4초보다 좋다는 비교가 아니다. 직선이라는 원래 선정 조건과 현재 물리 마커의 별도 기하 판정은 같지 않으므로 상세 표에 모두 남겼다.','',
      '사건 계산은 천골 대신 뒤 허리 중점과 매 프레임의 골반 방향을 사용한 미검증 적용이다. 광학 마커는 사건 계산에만 들어가며 RGB 입력에는 넣지 않았다. 사건 계산에만 양옆 최대 0.5초의 원래 보행 문맥을 사용했다.','',
      '14명·28사건 구간·224주기 추정의 원본/프레임/마커 연결과 SHA256, 독립 사건 검출 및 SVD 주기 계산, 지표 재계산을 통과했다. 사건 인덱스는 완전 일치, 수치 허용오차 1e-10, 주기 R² 허용오차 1e-9를 유지했다. 새 학습은 없고 기존 SPPB/TUG 모델·사람·구간·초기값 5개는 보존했다. 실제 휴대폰이나 임상 점수 정확도를 입증하지 않는다.',f'선택 배열: {STORE}','',
      '[프로토콜](protocol.json) · [전체 비교](metrics.csv) · [동일인 비교](geometry-paired.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
