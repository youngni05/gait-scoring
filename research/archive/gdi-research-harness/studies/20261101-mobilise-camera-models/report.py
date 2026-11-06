from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and (v['tasks'],v['model_replays'],v['refits'])==(50,1500,32)
    assert v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py')
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 속도 중심 입력의 RF·Deep Sets와 내부 적중률 선택','','같은 입력에서 모델과 내부 설정 선택이 적중률을 높이는지 확인했다. 속도만/속도+걸음빈도를 각각 고정RF·적중률선택RF·적중률선택DeepSets로 비교한다. SPPB2,174명/TUG1,939명,30–60초5구간·사람5분할/내부3분할·초기값5개·4구간경계를 유지했다.','',
      '고정RF는잎20,선택RF는잎5/20중학습내부사람평균적중률로선택한다. DeepSets는1/2→32→평균→16→4,10–600epoch중같은내부기준으로선택한다. 동률은로그손실·큰잎/이른epoch순이다. 평가지표로설정을고르지않았다.','',
      '| 목표 | 방법 | 적중률 ↑ | 구간평균재현율 ↑ | Brier ↓ | 양끝오판 ↓ | 저기능놓침 ↓ | 오경고 ↓ | 같은질환·속도구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target,name in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(name+'_cal')].mean(numeric_only=True);rank=mt[mt.target.eq(target)&mt.method.eq(name+'_cal')].pair_order_accuracy.mean()
        lines.append(f'| {target.upper()} | {name} | {q.accuracy*100:.2f}% | {q.balanced_recall*100:.2f}% | {q.brier:.4f} | {q.extreme_error*100:.2f}% | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {rank*100:.2f}% |')
    choices=[]
    for t,f,s in tasks('main'):
        l=json.loads((STORE/'folds'/f'{t}-f{f}-s{s}.json').read_text())
        for kind in ['Speed','Cadence']:
            for name in [kind+'AccRF',kind+'DS']:
                q=l['models'][name];choices.append(dict(target=t,fold=f,seed=s,method=name,leaf=q.get('selected_leaf'),epochs=q.get('epochs')))
    pd.DataFrame(choices).to_csv(HERE/'inner-selections.csv',index=False)
    lines+=['','## 구간별 재현율','','| 목표 | 방법 | 구간1 | 구간2 | 구간3 | 구간4 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for target,name in itertools.product(['sppb','tug'],METHODS):
        q=m[m.target.eq(target)&m.method.eq(name+'_cal')].mean(numeric_only=True);lines.append('| '+target.upper()+' | '+name+' | '+' | '.join(f'{q[f"recall_{j}"]*100:.2f}%' for j in range(4))+' |')
    lines+=['','SPPB구간0–3/4–6/7–9/10–12,TUG구간≤10/(10,15]/(15,20]/>20초다. 저기능 경고는 별도 내부 오경고10% 규칙이며 전체 최대확률구간 적중과 구분한다.','',
      '후보 내부예측을 설정 선택과 보정에 함께 사용했으므로 내부 성능은 선택용이다. 바깥 평가는 사람을 분리했지만 과거 실험에서 반복 확인한 개발 자료다. 새 영상·임상표본 성능이나 키보정속도 정확도의 검증은 아니다. 모든 모델과초기값결과·사람재표집구간을함께보고한다.','',
      f'검산50작업·1500모델재현·32재학습,독립지표최대차이{v["independent_metric_error"]:.3g}.',f'대량산출물:{STORE}','',
      '[조건](protocol.json) · [초기값별 지표](metrics-by-seed.csv) · [내부 선택](inner-selections.csv) · [사람 재표집](paired-bootstrap.json) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('REPORT_READY',flush=True)
