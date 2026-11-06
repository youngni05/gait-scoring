from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==50 and v['model_scenario_replays']==1400 and v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py')
    summary=json.loads((HERE/'summary.json').read_text());m=pd.read_csv(HERE/'metrics-by-seed.csv')
    lines=['# 속도 입력 오차가 점수 예측에 미치는 영향','','같은 사람·방문·30–60초5구간·초기값5개에서 기존 속도만/속도+걸음빈도의RF·DeepSets를 고정하고 속도 입력만 교란했다. 모델·정규화·보정·경고 문턱을 새로 맞추지 않았다. 실제 영상의 점수 적중률을 측정한 실험이 아니다.','',
      '| 목표 | 오차 조건 | 방법 | 적중률 | 평균재현율 | Brier | 양끝오판 | 저기능놓침 | 오경고 |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target,scenario,name in itertools.product(['sppb','tug'],SCENARIOS,METHODS):
        r=m[m.target.eq(target)&m.scenario.eq(scenario)&m.method.eq(name)].mean(numeric_only=True);lines.append(f'| {target.upper()} | {scenario} | {name} | {r.accuracy*100:.2f}% | {r.balanced_recall*100:.2f}% | {r.brier:.4f} | {r.extreme_error*100:.2f}% | {r.fpr10_miss*100:.2f}% | {r.fpr10_fpr*100:.2f}% |')
    lines+=['','## 사전 주 비교: ±0.10m/s편향 평균에서 걸음 빈도 추가 효과','','| 목표 | 모델 | 적중률 차이(%p) | 개선 초기값 | 놓침 차이(%p) | 오경고 차이(%p) | 공동 목표 |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in summary['primary']:
        passed=r['accuracy_delta']>=.005 and r['miss_delta']<=0 and r['fpr_delta']<=.01
        lines.append(f"| {r['target'].upper()} | {r['model']} | {r['accuracy_delta']*100:.2f} | {r['accuracy_improved_seeds']}/5 | {r['miss_delta']*100:.2f} | {r['fpr_delta']*100:.2f} | {'충족' if passed else '미충족'} |")
    lines+=['','속도 편향·배율·난수 오차는 통제된 합성 조건이다. MoVi2초 골반 속도 오차를 Mobilise-D의 실제 센서 구간 오차 분포로 가져오지 않았다. 걸음 빈도는 정확한 값으로 유지했으므로 추가 입력의 낙관적인 상한이다. 새 사람·자료·임상 정확도를 늘린 실험이 아니다. 기존 개발 사람을 반복 평가한 탐색 한계를 유지한다.','',
      '저기능은SPPB0–3/TUG>20초다. 경고 문턱은 원래 학습 내부에서 정한 오경고10% 규칙을 고정했고,교란 조건에서 다시 맞추지 않았다. 같은 질환·비슷한 속도 비교의 짝도 원래 실제 속도 기준으로 고정했다. 구간 경계는SPPB0–3/4–6/7–9/10–12,TUG≤10/(10,15]/(15,20]/>20초를 유지했다.','',
      f'검산50작업·1400모델×조건재현·새학습0·280지표/280속도짝행 통과,최대재현차이{v["max_prediction_error"]:.3g}. 허용오차1e-12.',f'예측배열:{STORE}','', '[조건](protocol.json) · [초기값별 지표](metrics-by-seed.csv) · [속도짝](matched-by-seed.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
