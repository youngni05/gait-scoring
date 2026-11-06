"""Render verified experiment facts; no post-hoc model selection."""
from study import *

def pct(x): return f'{100*x:.2f}%'

def run():
    v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed'
    alignment=json.loads((HERE/'alignment-verification.json').read_text());assert alignment['passed']
    summary=json.loads((HERE/'summary.json').read_text())
    assert v['analysis_sha256']==sha(HERE/'analyze.py')==summary['analysis_sha256']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');matches=pd.read_csv(HERE/'matched-by-seed.csv')
    boot=json.loads((HERE/'paired-bootstrap.json').read_text())['comparisons']
    names={'SpeedRF_cal':'속도만 RF','MedianRF_cal':'중앙값 4개 RF','BoutRF_cal':'구간별 6개 RF','DeepSets_cal':'구간별 6개 Deep Sets'}
    lines=['# Mobilise-D: 입력 정보와 분류 모델 비교','',
      '**결론: 구간별 확장 정보를 유지한 RF와 Deep Sets는 기존 중앙값4개 RF보다 확률 오차와 저기능 놓침을 줄였다. 대표 조건의 개선은 초기값5개 모두에서 유지됐고, 속도가 비슷한 사람들끼리의 비교에서도 구분이 개선됐다.** 다만 실제 오경고는 약0.25–0.41%p 증가했다. Deep Sets는 확률 오차가 조금 작았지만 저기능 놓침은 확장 RF가 조금 적어 한쪽의 일관된 우위를 주장하지 않는다.','',
      '전체 구간 적중률은 SPPB 약52.5–52.7%, TUG 약58.0–58.3%다. 별도 경고 규칙에서는 놓침이 줄었지만 최대 확률 구간 하나만 고르면 SPPB 저기능 적중률은 여전히11.6–15.5%에 그쳤다. 정보가 더 있다는 근거와 확정 기능 등급으로 사용할 수 있다는 근거를 구분한다.','',
      'SPPB와 TUG를 각각 별도 모델로 예측한다. SPPB는 0–3 / 4–6 / 7–9 / 10–12점, TUG는 ≤10 / (10,15] / (15,20] / >20초의 기존 4구간을 유지했다. 입력에 SPPB·TUG·실제 검사 걷기 점수나 진단명은 포함하지 않았다.','',
      '이번 질문은 속도 외의 구간별 정보가 점수 구간과 저기능을 더 잘 구분하는지다. SPPB 2,174명·7,356방문, TUG 1,939명·5,688방문을 유지했다. 같은 사람의 모든 방문을 같은 바깥 평가 분할에 묶었다. 저기능 기록이 있는 사람은 각각195명·290명이다.','',
      '## 비교 방법','',
      '| 방법 | 입력과 역할 |','| --- | --- |',
      '| 속도만 RF | 선택 구간의 보행 속도 중앙값 하나. 속도만으로 가능한 기준선 |',
      '| 중앙값 4개 RF | 속도·걸음 빈도·보행 주기·보폭의 중앙값. 이전 직접 4구간 분류 RF를 그대로 재사용 |',
      '| 구간별 6개 RF | 기존 4개에 구간 길이·초당 회전 수를 추가. 각 구간을 속도순으로 정렬해 모든 값을 입력 |',
      '| 구간별 6개 Deep Sets | 같은 여섯 값을 구간별로 변환한 뒤 평균으로 모아 네 구간 확률을 출력 |','',
      'RF 세 방법의 나무 수128개·최소 잎 표본20개는 같다. Deep Sets는 6→32→평균→16→4 구조, 교차엔트로피 손실, 사람 동일 학습 비중을 사용했다. 10–600 epoch 중 학습 사람 안쪽3분할의 사람 평균 로그손실이 가장 작은 종료 시점을 선택했다. 바깥 평가 사람은 종료 시점·보정·경고 문턱 선택에 쓰지 않았다.','',
      '초기값은20261030·20262030·20263030·20264030·20265030이다. [4,10)초·[30,60)초 각각1·3·5개를 사용했다. 대표 조건은 결과 확인 전부터 유지한30–60초5개다. 아래 값은 초기값5개의 지표 평균이며 예측을 평균한 앙상블이 아니다.','',
      '확률 보정은 기존 실험과 같은 온도 보정이다. 학습 내부 예측으로 보정하고, 내부 오경고10% 조건의 문턱을 바깥 평가에 그대로 적용했다. 따라서 바깥 평가 오경고가 정확히10%로 고정되지는 않는다. 보정 전 결과도 CSV에 남겼다. Deep Sets 종료 시점 선택에 사용한 내부 예측을 보정에도 사용하므로 내부 성능은 선택용 값이며 최종 성능으로 해석하지 않는다.','',
      '## 대표 조건: 전체 결과','',
      'Brier는 네 구간 확률과 정답의 차이를 제곱해 합한 값으로 낮을수록 좋다. 적중률은 최대 확률 구간의 일치율, 양끝 오판은 두 끝 구간을 서로 바꾼 비율이다. 놓침은 실제 저기능 중 경고하지 않은 비율, 오경고는 비저기능 중 경고한 비율이다.','']
    for target in ['sppb','tug']:
        lines += [f'### {target.upper()}','',
          '| 방법 | Brier ↓ | 구간 적중률 ↑ | 구간별 평균 적중률 ↑ | 양끝 오판 ↓ | 내부 오경고10% 규칙: 놓침 ↓ | 실제 오경고 ↓ |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in summary['metrics']:
            if row['target']==target and row['method'] in names:
                lines.append(f"| {names[row['method']]} | {row['brier']:.4f} | {pct(row['accuracy'])} | {pct(row['balanced_recall'])} | {pct(row['extreme_error'])} | {pct(row['fpr10_miss'])} | {pct(row['fpr10_fpr'])} |")
        lines += ['','| 방법 | 실제 구간1 적중 | 구간2 적중 | 구간3 적중 | 구간4 적중 | 최대 확률 판정의 저기능 놓침 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in summary['metrics']:
            if row['target']==target and row['method'] in names:
                vals=' | '.join(pct(row[f'recall_{j}']) for j in range(4))
                lines.append(f"| {names[row['method']]} | {vals} | {pct(row['low_miss'])} |")
        lines += ['']
    lines += ['## 비슷한 속도에서의 구분','',
      '같은 바깥 평가 분할에서 속도 차이0.05m/s 이내인 저기능 방문과 다른 사람의 비저기능 방문을 가장 가까운 속도로 짝지었다. 동률은 사람ID·방문순으로 정했다. 한 비교 사람이 여러 번 사용될 수 있으며, 같은 저기능 사람의 반복 방문은 사람당 동일 비중으로 평균했다. 짝짓기에는 예측 확률을 쓰지 않았다.','',
      '아래의 위험 순서 적중률은 저기능 쪽에 더 높은 저기능 확률을 주면1, 같으면0.5, 반대면0으로 계산한다. 이는 임상 구간 적중률이나 모집단 전체의 AUROC가 아니다. 같은 질환 집단 내 짝짓기도 보조 분석으로 확인했다. 두 방식 모두 관찰자료의 기술적 비교이며 인과효과를 증명하지 않는다.','',
      '| 목표 | 비교 제한 | 저기능 포함 사람 | 비교에 쓰인 비저기능 사람 | 저기능 방문 포함률 | 평균 속도 차이 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for target in ['sppb','tug']:
        for matching in ['speed','speed_cohort']:
            row=next(r for r in summary['matched'] if r['target']==target and r['matching']==matching)
            lines.append(f"| {target.upper()} | {'속도' if matching=='speed' else '속도+같은 질환'} | {int(row['matched_low_people'])}/{int(row['low_people'])} | {int(row['unique_control_people'])} | {pct(row['low_visit_coverage'])} | {row['mean_speed_gap']:.4f}m/s |")
    lines += ['','| 목표 | 방법 | 속도 짝 위험 순서 적중 | 같은 질환 내 적중 |','| --- | --- | ---: | ---: |']
    for target in ['sppb','tug']:
        for name,label in names.items():
            get=lambda matching:next(r['pair_order_accuracy'] for r in summary['matched'] if r['target']==target and r['method']==name and r['matching']==matching)
            lines.append(f'| {target.upper()} | {label} | {pct(get("speed"))} | {pct(get("speed_cohort"))} |')
    lines += ['','| 목표 | 방법 | 비교 제한 | 중앙값 RF 대비 순서 적중 차이 | 개선 초기값 |','| --- | --- | --- | ---: | ---: |']
    primary_matches=matches[matches.band.eq('long')&matches.k.eq(5)]
    for target in ['sppb','tug']:
        for matching in ['speed','speed_cohort']:
            wide=primary_matches[primary_matches.target.eq(target)&primary_matches.matching.eq(matching)].pivot(index='seed',columns='method',values='pair_order_accuracy')
            for name in ['BoutRF_cal','DeepSets_cal']:
                difference=wide[name]-wide['MedianRF_cal']
                lines.append(f"| {target.upper()} | {names[name]} | {matching} | {100*difference.mean():+.2f}%p | {int((difference>0).sum())}/5 |")
    lines += ['','## 초기값5개와 불확실성','',
      '다음 차이는 확장 입력/Deep Sets − 기존 중앙값4개 RF다. Brier·놓침·오경고는 음수가 개선이다. 신뢰구간은 같은 사람의 초기값별 오차를 평균한 뒤 사람을2,000번 재표집한 조건부95%구간이다. 재학습·다중비교 보정·외부검증은 포함하지 않는다.','',
      '| 목표 | 방법 | 지표 | 평균 차이 | 95%구간 | 개선 초기값 |','| --- | --- | --- | ---: | --- | ---: |']
    for r in boot:
        if r['baseline']=='MedianRF_cal' and r['metric'] in ['brier','fpr10_miss','fpr10_fpr']:
            scale=1 if r['metric']=='brier' else 100
            unit='' if scale==1 else '%p'
            lo,hi=r['ci95']
            lines.append(f"| {r['target'].upper()} | {names[r['method']]} | {r['metric']} | {r['difference']*scale:+.4f}{unit} | [{lo*scale:+.4f}, {hi*scale:+.4f}]{unit} | {r['better_seeds']}/5 |")
    lines += ['','## 여섯 보행 조건의 비교','',
      '다음도 확장 방법 − 중앙값4개 RF의 초기값5개 평균 차이다. 음수가 작은 오차·놓침·오경고를 뜻한다. 대표 조건 외의 결과를 숨기거나 가장 좋은 조건만 골라 전체 성능으로 보고하지 않는다.','',
      '| 목표 | 보행 길이·개수 | 방법 | Brier 차이 | 놓침 차이 | 오경고 차이 |','| --- | --- | --- | ---: | ---: | ---: |']
    mm=m[m.method.isin(names)].groupby(['target','band','k','method']).mean(numeric_only=True)
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        base=mm.loc[(target,band,k,'MedianRF_cal')]
        for method in ['BoutRF_cal','DeepSets_cal']:
            diff=mm.loc[(target,band,k,method)]-base
            lines.append(f"| {target.upper()} | {'4–10초' if band=='short' else '30–60초'}·{k}개 | {names[method]} | {diff.brier:+.4f} | {100*diff.fpr10_miss:+.2f}%p | {100*diff.fpr10_fpr:+.2f}%p |")
    lines += ['','## 검산과 한계','',
      f"전체300개 분할·초기값 작업을 완료했다. 저장 모델 재현은{v['replayed_models']}개이며, 사전 지정 모델{len(v['refits'])}개의 내부/최종 재학습 결과도 일치했다. 전체540행 지표와 속도 짝 비교480행을 검산했다. 최종 확률 최대 재현 차이는{v['final_max_error']:.3g}, 독립 지표 계산 차이는{v['independent_metric_max_error']:.3g}였다. Deep Sets가 최대600 epoch를 선택한 작업은{v['epoch600_count']}/300개다.",'',
      '이 비교의 확장 입력은 구간 평균과 길이·회전 빈도다. 걸음마다의 비대칭·흔들림·관절 시계열을 새로 넣은 실험은 아니다. RF의 입력 비교에는 특징 추가와 구간 요약 손실 감소가 함께 포함되며, 둘의 효과를 따로 분리한 요인 실험은 아니다. 구간별 RF와 Deep Sets는 같은 정보를 다른 방식으로 처리한다.','',
      '여러 후속 실험에서 같은 평가 사람을 사용했다. 이번 내부 선택 절차가 분리되어 있어도 연구 방향 자체의 반복 탐색에 따른 낙관 가능성은 남는다. 센서 특징의 결과를 단일 카메라 임상 점수의 정확도로 해석하지 않는다.','',
      f'작은 코드·요약·검산 기록: `{HERE}`',f'모델·대량 예측·속도 짝 연결: `{STORE}`','',
      '핵심 처리 흐름은 study.py의 work다. 같은 사람 분할과 선택 보행 구간을 읽고 → 입력만 달리해 학습하고 → 학습 내부에서 확률과 경고 문턱을 보정한 뒤 → 평가 사람의 예측을 저장한다. analyze.py는 사람 단위로 성능과 속도 짝을 비교하고, verify.py는 저장 모델·분할·지표·짝짓기를 다시 계산한다.','',
      '[전체 조건 요약](summary-by-condition.csv) · [초기값별 지표](metrics-by-seed.csv) · [속도 짝 비교](matched-summary.csv) · [검산](verification.json) · [사람·방문 연결 검산](alignment-verification.json) · [사전 조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(HERE/'report-completion.json',{'readme_sha256':sha(HERE/'README.md'),'report_sha256':sha(__file__),'verification_sha256':sha(HERE/'verification.json'),'alignment_sha256':sha(HERE/'alignment-verification.json')})
    print('REPORT_READY',flush=True)

if __name__=='__main__':run()
