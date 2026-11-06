from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==27 and v['verify_sha256']==sha(HERE/'verify.py');e=pd.read_csv(HERE/'main-events.csv');m=pd.read_csv(HERE/'metrics.csv');pairs=pd.read_csv(HERE/'paired.csv');old=m[m.method.eq('Reference3D')].iloc[0]
    lines=['# 걸음 주기 참조의 광학 사건 진단','','기존27명의같은4초에서광학발뒤꿈치/골반마커의진행방향거리최대값으로사건후보를구했다. 기존주기맞춤과다른계산으로참조가느린포락선·반주기를고르는지확인했다. **이사건은힘판/전문가접촉정답이아닌기구학적대용값이다.**','',
      f'전체27명중사건품질가용은{int(e.quality.sum())}명,회전이적은기하조건은{int(e.straight.sum())}명이다. 기존기준3D주기와사건대용값의공동가용은{int(old.both_quality_people)}명이다.','',
      '| 비교 대상 | 시점 | 전체 | 사건 가용 | 주기 가용 | 공동 가용 | 평균 절대 차이(걸음/분) | 반주기 차이 징후 | 직선·공동 가용 | 직선 차이 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt=lambda x:'계산 불가' if pd.isna(x) else f'{x:.2f}'
    for _,r in m.iterrows():lines.append(f'| {r.method} | {r.camera} | {r.selected_people} | {r.event_quality_people} | {r.period_quality_people} | {r.both_quality_people} | {fmt(r.mean_absolute_discrepancy)} | {r.half_proxy_people} | {r.straight_both_people} | {fmt(r.straight_mean_absolute_discrepancy)} |')
    lines+=['','| RGB 방법 | 사건·양쪽 주기 공동가용 | 시점차이(걸음/분) |','| --- | ---: | ---: |']
    for _,r in pairs.iterrows():lines.append(f'| {r.method} | {r.both_camera_event_quality_people} | {fmt(r.mean_absolute_view_difference)} |')
    consistent=int(e.quality.sum())/27>=.8 and pd.notna(old.mean_absolute_discrepancy) and old.mean_absolute_discrepancy<=5
    lines+=['', '사전참조일관성탐색목표는'+('충족했다.' if consistent else '충족하지못했다.')+' 이것은어느계산이실제발접촉정답인지판정하는검사가아니다. 두대용값이불일치하면기존작은주기차이를실제정확도로해석할수없다.','',
      '[Zeni등2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2384115/)의거리최대값원리를참고했으나천골대신허리중점,고정진행축대신매프레임골반축으로변경했다. 회전/후진/마커결측/비정상보행에서의검증은없다. 원논문의힘판대조정확도를이번구현에적용하지않는다.','',
      '기존RGB주기추정기와품질기준은수정하지않았다. SPPB/TUG점수모델·사람분할·초기값5개도보존했다. 새임상표본/영상점수정확도/전문가접촉라벨을확보했다고주장하지않는다. 외부정답자료접근문제는프로토콜에기록했다.','',
      '27명원본/마커/시간/지문·독립사건검출·빈도/지표재계산검산통과. 사건인덱스와좌우완전일치,고정수치허용오차1e-10.100걸음/분가상신호검사는계산검사이며실제사건정확도검증이아니다.',f'선택배열:{STORE}','',
      '[프로토콜](protocol.json) · [사건 수](main-events.csv) · [비교](metrics.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
