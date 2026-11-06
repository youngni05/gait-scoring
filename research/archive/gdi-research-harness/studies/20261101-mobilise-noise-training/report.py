from study import *
if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and (v['tasks'],v['model_subset_replays'],v['refits'])==(50,1700,16) and v['verify_sha256']==sha(HERE/'verify.py') and v['analysis_sha256']==sha(HERE/'analyze.py');s=json.loads((HERE/'summary.json').read_text());m=pd.read_csv(HERE/'metrics-by-seed.csv')
    lines=['# 속도 오차 학습의 점수 구간 예측 효과','','같은SPPB2174명/TUG1939명·30–60초5구간·사람5분할·초기값5개·4등급에서 기존Deep Sets와 학습 중에만 속도 오차를 섞은 모델을 비교했다. 구조·초기 가중치·기존내부검증선택epoch를 유지하고 정규화·보정·경고 문턱은 깨끗한 학습자료만으로 계산했다.','',
      '| 목표 | 입력 | 교란6조건 적중률 차이(%p) | 개선seed | 원본 적중률 차이(%p) | +.10놓침 차이(%p) | −.10오경고 차이(%p) | 공동 목표 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in s['primary']:
        passed=r['accuracy_delta']>=.005 and r['clean_accuracy_delta']>=-.005 and r['positivebias_miss_delta']<=-.05 and r['negativebias_fpr_delta']<=.01
        lines.append(f"| {r['target'].upper()} | {r['input']} | {r['accuracy_delta']*100:.2f} | {r['accuracy_improved_seeds']}/5 | {r['clean_accuracy_delta']*100:.2f} | {r['positivebias_miss_delta']*100:.2f} | {r['negativebias_fpr_delta']*100:.2f} | {'충족' if passed else '미충족'} |")
    lines+=['','| 목표 | 조건 | 방법 | 적중률 | 평균재현율 | Brier | 양끝오판 | 저기능놓침 | 오경고 |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target,scenario,name in itertools.product(['sppb','tug'],SCENARIOS,METHODS):
        r=m[m.target.eq(target)&m.scenario.eq(scenario)&m.method.eq(name)].mean(numeric_only=True);lines.append(f'| {target.upper()} | {scenario} | {name} | {r.accuracy*100:.2f}% | {r.balanced_recall*100:.2f}% | {r.brier:.4f} | {r.extreme_error*100:.2f}% | {r.fpr10_miss*100:.2f}% | {r.fpr10_fpr*100:.2f}% |')
    lines+=['','이는 합성 입력 오차에 대한 연구다. 새로운 사람 수나 실제 카메라 영상의 점수 정확도를 늘린 결과가 아니다. 걸음 빈도는 정확한 값으로 둔 낙관적 조건이고 학습 오차가 실제 촬영·추적 실패를 대변하지 않는다. 기존 개발 사람을 반복 확인한 탐색 한계를 유지한다. 모든 조건·원본 성능·어려운 구간·초기값별 결과를 함께 판단한다.','',f'50작업·1700회모델/부분집합재현·16회지정재학습·280지표/280속도짝 검산 통과. 최대재현차이{v["max_prediction_error"]:.3g},허용오차1e-12. 모델/예측:{STORE}','', '[조건](protocol.json) · [초기값별](metrics-by-seed.csv) · [구분](matched-by-seed.csv) · [검산](main-verification.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
