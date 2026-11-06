from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['clips']==56 and v['people']==14 and v['verify_sha256']==sha(HERE/'verify.py')
    assert json.loads((HERE/'main-process-exit.json').read_text())['exit_code']==0
    m=pd.read_csv(HERE/'metrics.csv');p=pd.read_csv(HERE/'paired.csv')
    lines=['# 키를 이용한 영상 속도 측정 진단','','고정 카메라에서 키와 영상상 신체 높이로 바꾼 이동 속도가 촬영 방향에 따라 달라지는지 확인했다. 기존 MoVi14명,같은2초 직선/회전구간,카메라2대의56클립을 재사용했다. 독립 사람은14명이다. 새 점수 학습이나 영상 추론은 하지 않았다.','',
      'PlaneNet은처음·끝 골반 위치,PlaneOLS는시간에 따른 위치 기울기에 키/영상상 신체 높이를 곱한다. PinholeOLS는추가 카메라 내부 보정값을 써서 깊이도 추정하는 진단용 방법이다. 키만 입력받는 조건과 구분한다. 세 방법 모두 검출된33개자세점의 세로 범위를 영상상 키 대신 사용하는 단순 후보다. 머리 꼭대기 누락·자세 변화·가림·검출 흔들림의 영향을 포함한다.','',
      '| 구간 | 카메라 | 방법 | 사람 | 속도 MAE(m/s) | 편향(m/s) | 오차90백분위(m/s) |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for _,r in m.iterrows():lines.append(f'| {r.condition} | {r.camera} | {r.method} | {r.people} | {r.mae:.3f} | {r.bias:.3f} | {r.p90_error:.3f} |')
    lines+=['','| 구간 | 방법 | 카메라간 평균 절대 차이(m/s) | PG2−PG1편향(m/s) |','| --- | --- | ---: | ---: |']
    for _,r in p.iterrows():lines.append(f'| {r.condition} | {r.method} | {r.absolute_view_difference:.3f} | {r.signed_view_difference:.3f} |')
    primary=m[m.condition.eq('straight')&m.method.eq('plane_ols')];mae=primary.mae.mean();diff=float(p[p.condition.eq('straight')&p.method.eq('plane_ols')].absolute_view_difference.iloc[0])
    lines+=['',f'사전에 정한 단순PlaneOLS직선 목표는 평균MAE와카메라간 차이 각각0.10m/s이하다. 실제 평균MAE{mae:.3f},카메라차이{diff:.3f}m/s로 '+('두 조건을 만족했다.' if mae<=.1 and diff<=.1 else '공동 조건을 만족하지 못했다.')+' 이 문턱은 탐색용 공학 목표이며 임상 기준이 아니다.','',
      '기준은 제공된3D골반의 수평 순이동/기울기다. 순이동과기울기방법은해당기준끼리비교했다. 원근추정의3D기울기는수직흔들림도포함하며왜곡·바닥·카메라자세보정은하지않았다. 이정의는Mobilise-D의구간평균보행속도와같지않다.2초결과를30–60초5구간입력으로바로대체할근거는없다.','',
      'MoVi에는이실험의SPPB/TUG정답이없다. 정상점수를가정하거나점수적중률을계산하지않았다. 자료의구간선정에는기준좌표를이용했으므로실제사용시자동가용구간선정성능도입증하지못한다. 사람키는제공된메타데이터이며새사용자의입력오차분포는미확인이다. ±2cm의비례민감도만추가표에기록했다.','',
      f'56클립·3360프레임 원본 연결/SHA256·독립최소제곱 재계산 검산 통과. 최대수치차이{v["max_independent_error"]:.3g},사전허용오차1e-10. 모델재학습없음.',
      f'대량배열:{STORE}','', '[프로토콜](protocol.json) · [클립](main-clips.csv) · [지표](metrics.csv) · [시점차이](paired.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
