from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==28 and v['predictions']==112 and v['verify_sha256']==sha(HERE/'verify.py')
    m=pd.read_csv(HERE/'metrics.csv');c=pd.read_csv(HERE/'camera-metrics.csv');pairs=pd.read_csv(HERE/'paired.csv')
    lines=['# 다른 사람·카메라로 속도 배율 보정 이전하기','','같은14명의 모든 영상을 한 사람씩 평가용으로 빼고 나머지13명의 한 카메라 직선 보행에서 배율 하나를 학습했다. 학습에 없던 사람의 같은 카메라와 반대 카메라에 적용했다. 총28개 배율·112예측이며,독립 사람 수는14명이다.','', '| 구간 | 평가 카메라 관계 | 방법 | MAE(m/s) | 편향(m/s) |','| --- | --- | --- | ---: | ---: |']
    for _,r in m.iterrows():lines.append(f'| {r.condition} | {r.transfer} | {r.method} | {r.mae:.4f} | {r.bias:.4f} |')
    rows=c[c.condition.eq('straight')&c.transfer.eq('opposite_camera')&c.method.eq('ScaledPinhole')];raw=float(m[m.condition.eq('straight')&m.transfer.eq('opposite_camera')&m.method.eq('RawPinhole')].mae.iloc[0]);corrected=float(m[m.condition.eq('straight')&m.transfer.eq('opposite_camera')&m.method.eq('ScaledPinhole')].mae.iloc[0]);gap=float(pairs[pairs.condition.eq('straight')&pairs.method.eq('ScaledPinhole')].absolute_view_difference.mean());passed=raw-corrected>=.02 and (rows.mae<=.1).all() and gap<=.1
    lines+=['',f'주 비교인 다른 카메라 직선 구간의MAE는{raw:.4f}→{corrected:.4f}m/s였다. 보정 후PG1/PG2오차는각각'+ '/'.join(f'{x:.4f}' for x in rows.sort_values('camera').mae)+'m/s다. 같은 학습 카메라의 배율을 양쪽 평가 시점에 적용한 카메라 차이는평균'+f'{gap:.4f}m/s였다. 사전의 탐색용 공동 목표는'+('만족했다.' if passed else '만족하지 못했다.')+' 임상 기준은 아니다.','',
    '평가 사람의 정답이나 회전 구간 정답으로 배율을 조정하지 않았다. 배율은원점을지나는직선의기울기이며기준속도와추정속도의곱합/추정속도제곱합으로계산했다. 한 카메라에서 학습해 다른 카메라에 적용했지만 같은 연구 장비와 자료 안의 비교다. 카메라 내부 보정값도 사용했으므로 키만 입력하는 일반 휴대폰 완성 방법이 아니다.','',
    '이 결과는 속도 측정의 개선이며 SPPB/TUG 적중률 개선이 아니다.2초 골반 순이동의 기울기와 Mobilise-D의30–60초 구간 평균 보행 속도 정의도 다르다.14명은이전부터확인한탐색자료이고새임상검증표본이아니다. 회전 구간 PG2의 보정 후MAE는0.114m/s로남았다. 배율 보정만으로 어려운 관측을 모두 해결하지 못했다.','',
    f'원본/코드지문·28배율 재현·28독립최소제곱 재학습·112예측/지표 재계산 검산 통과. 최대배율차이{v["max_scalar_difference"]:.3g},고정허용오차1e-12.',f'모델·예측:{STORE}','',
    '[조건](protocol.json) · [전체 지표](metrics.csv) · [카메라별](camera-metrics.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
