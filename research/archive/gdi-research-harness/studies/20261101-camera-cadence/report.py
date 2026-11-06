from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());p=json.loads((HERE/'protocol.json').read_text());n=p['expected']['main_people'];assert v['passed'] and v['tasks']==n*2 and v['estimates']==n*16 and v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py');m=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'paired.csv');length=pd.read_csv(HERE/'length-matched.csv');availability=json.loads((HERE/'availability.json').read_text())
    lines=['# 고정 카메라 보행 영상의 주기 추정 진단','',f'추가 입력인 걸음 빈도에 활용할 주기 정보를 RGB에서 일관되게 얻는지 확인했다. 기존30명을 조사해 2/4/6초 가용 사람은각각{availability["available_counts"]["2"]}/{availability["available_counts"]["4"]}/{availability["available_counts"]["6"]}명이었다. 같은4초가 가능한{n}명의 앞2초와전체4초를두카메라에서비교했다. 걷기구간내회전을포함할수있다.','',
    '**전문가가 표시한 발 접촉 정답이 없는 주기 대용값 진단이다.** 기준3D발목의4초 주기도 같은 알고리즘으로 추정했다. 아래 수치는 그 대용값과의 차이이며 실제 걸음 빈도 정답이나 SPPB/TUG 점수 정확도가 아니다. 같은 알고리즘을 기준에도 쓰면 일치가 과대평가될 수 있다.','',
    '| 길이 | 방법 | 카메라 | 전체 | 기준 주기 가용 | RGB/후보 품질 가용 | 공동 통과 | 대용값 차이(걸음/분) |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'계산 불가' if pd.isna(x) else f'{x:.2f}'
    for _,r in m.iterrows():lines.append(f'| {r.seconds}초 | {r.method} | {r.camera} | {r.selected_people} | {r.reference_valid_people} | {r.candidate_quality_people} | {r.accepted_people} | {fmt(r.accepted_proxy_discrepancy)} |')
    lines+=['','| 길이 | 방법 | 양쪽·기준 공동통과 | 두 카메라 차이(걸음/분) |','| --- | --- | ---: | ---: |']
    for _,r in pairs.iterrows():lines.append(f'| {r.seconds}초 | {r.method} | {r.accepted_people} | {fmt(r.accepted_view_difference)} |')
    lines+=['','## 같은 사람에서2초와4초 비교','','| 방법 | 카메라 | 공동통과 사람 | 2초 차이 | 4초 차이 |','| --- | --- | ---: | ---: | ---: |']
    for _,r in length.iterrows():lines.append(f'| {r.method} | {r.camera} | {r.common_people} | {fmt(r.proxy_discrepancy_2sec)} | {fmt(r.proxy_discrepancy_4sec)} |')
    a=m[m.seconds.eq(4)&m.method.eq('RGB3D')];b=pairs[pairs.seconds.eq(4)&pairs.method.eq('RGB3D')].iloc[0];coverage=bool((a.candidate_quality_coverage>=.8).all());error=bool(a.accepted_proxy_discrepancy.notna().all() and (a.accepted_proxy_discrepancy<=5).all());gap=bool(pd.notna(b.accepted_view_difference) and b.accepted_view_difference<=5)
    lines+=['',f'사전4초RGB3D탐색목표:두시점품질가용률80%이상={coverage},대용값차이5걸음/분이하={error},시점차이5걸음/분이하={gap}. 임상기준이아니며거부된사람을낮은점수로처리하지않았다.','',
    'RGB3D는RGB모델이복원한국소좌표이며실제3D정답을입력하지않았다. ProjectedReference2D는제공기준의투영으로기하진단에만쓰며제품입력이아니다. 기준3D는선정/평가에만사용했다. 신규모델학습·초기값실험·임상점수학습은없다. 기존Mobilise-D의사람/구간/5seed는보존했다.','',
    '신호품질조건은주기적합도0.5이상/1.5주기이상/주파수경계여유다. 선택된한구간과일반신체가시성검사는가림이심한실사용가용률과같지않다.2/4초주기가30–60초평균걸음빈도와같다고가정하지않는다. 건강한원자료의반복사용은새로운질환자검증이아니다.','',
    f'{n*2}작업·{n*16}추정의원본연결/지문·독립SVD주기계산·지표검산통과. 주파수격자선택일치,R2최대차이{v["max_r2_difference"]:.3g},고정허용1e-9. 모델학습0. 선택배열:{STORE}','',
    '[프로토콜](protocol.json) · [가용 수](availability.json) · [측정](metrics.csv) · [시점차이](paired.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
