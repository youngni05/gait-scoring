from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==14 and v['gate_replays']==56 and v['verify_sha256']==sha(HERE/'verify.py')
    m=pd.read_csv(HERE/'metrics.csv');p=pd.read_csv(HERE/'paired.csv');c=pd.read_csv(HERE/'proxy-comparison.csv');success=bool((m.straight_retention>=.8).all() and (m.turn_false_acceptance<=.2).all());fmt=lambda x:'계산 불가' if pd.isna(x) else f'{x:.2f}'
    lines=['# 영상만으로 회전 구간을 걸러내는 조건 점검','','직선 보행에서는 걸음 빈도 대용값의 차이가 작아져, 실제 영상만으로 직선 조건을 가려낼 수 있는지 확인했다. 같은 14명의 직선·회전 2초 영상과 두 카메라를 유지했다. 영상에서 복원한 어깨·골반 좌우 벡터의 방향 변화가 30도 이내이고 관절 가시성이 0.5 이상이면 통과시킨다. 별도 학습이나 결과에 따른 문턱 조정은 없다.','',
      '사전 탐색 목표는 각 카메라에서 직선 보행 80% 이상 유지와 회전 80% 이상 거부를 함께 달성하는 것이다. 결과는 '+('충족이다.' if success else '미충족이다.'),'',
      '| 카메라 | 직선 통과/전체 | 직선 유지율 | 회전 통과/전체 | 회전 오통과율 |','| --- | ---: | ---: | ---: | ---: |']
    for _,r in m.iterrows():lines.append(f'| {r.camera} | {r.straight_accepted}/{r.straight_people} | {100*r.straight_retention:.1f}% | {r.turn_accepted}/{r.turn_people} | {100*r.turn_false_acceptance:.1f}% |')
    lines+=['','| 조건 | 전체 사람 | 두 시점 모두 통과 | 시점 판정 불일치 | 두 시점 모두 거부 |','| --- | ---: | ---: | ---: | ---: |']
    for _,r in p.iterrows():lines.append(f'| {r.condition} | {r.people} | {r.both_accepted} | {r.one_view_disagrees} | {r.both_rejected} |')
    lines+=['','| 조건 | 방법 | 카메라 | 전체 | 통과 | 사건 비교 가능 | 통과·사건 가능 | 통과·사건·주기 품질 | 통과 전 차이 | 통과 후 차이 |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for _,r in c.iterrows():lines.append(f'| {r.condition} | {r.method} | {r.camera} | {r.people} | {r.gate_accepted} | {r.event_available} | {r.gated_event_available} | {r.gated_period_quality} | {fmt(r.ungated_discrepancy)} | {fmt(r.gated_discrepancy)} |')
    lines+=['','차이의 단위는 걸음/분이며 광학 발뒤꿈치 사건 대용값과의 차이다. 전문가 발접촉 정답에 대한 정확도가 아니다. 통과한 사람만의 차이가 작아져도 전체 성능 개선으로 해석하지 않는다. 통과하지 못한 영상은 측정 불가이며 낮은 기능 점수가 아니다.','',
      '방향 변화는 영상 모델이 복원한 좌표에서 계산한 값이다. 실제 몸의 회전 각도로 검증되지 않았고 깊이 흔들림으로 직선 영상을 거부할 수 있다. 같은 14명의 개발 진단으로 일반 휴대폰·다른 질환·새 사람의 성능을 입증하지 않는다. 기존 SPPB/TUG 모델·사람·구간·초기값 5개는 그대로 유지했다.','',
      '56클립의 원본·배열·프레임 연결 및 SHA256, 별도 각도 펼침·분위수·가시성 계산과 모든 지표 재계산을 검산했다. 통과 여부는 완전 일치, 각도 허용오차는 1e-10이다.',f'선택 배열: {STORE}','',
      '[프로토콜](protocol.json) · [지표](metrics.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
