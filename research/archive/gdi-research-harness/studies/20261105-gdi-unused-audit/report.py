"""Readable remaining-data report and fixed metadata-selected examples."""
from prepare_inputs import *
import csv
sys.path.insert(0,str(CASE));prior_report=module('case_report_labels',CASE/'report.py');LABEL=prior_report.LABEL
LABEL['single_CNN']='Single CNN (first available window)'
def fmt(v):return '해당 없음' if v is None else f'{v:.3f}'
def main():
 r=read(HERE/'results.json');v=read(HERE/'verification.json');assert v['passed'] and sha(HERE/'results.json')==v['results_sha256'];names=r['models'];policy=read(HERE/'label-policy.json');lines=['# 남아 있던 자료에 대한 고정 GDI 모델 예측','', '기존 원자료 5,713기록 중 기존 모델 개발·직전 평가에 쓴 1,438기록을 제외한 **4,275기록**을 확인했다. 같은 94개 체크포인트로 구성한 17개 방식을 적용했다. 재학습·새 보정·결합비율 탐색은 하지 않았다.','', '## 자료를 얼마나 쓸 수 있었나','', '|자료 구분|남은 기록|입력 가능 기록|정답과 비교한 기록|','|---|---:|---:|---:|']
 cn={'primary_remaining':'현재 모델 학습·직전 평가에 없던 사람','additional_final':'직전 평가 사람의 추가 기록','additional_development':'현재 개발 사람의 추가 기록','unlinked':'사람·정답 연결 미확인'}
 for co,z in r['cohort_availability'].items():lines.append(f'|{cn[co]}|{z["inventory_records"]}|{z["input_available_records"]}|{z["accuracy_records"]}|')
 lines+=['',f'입력 가능 {r["available_input_records"]}기록 중 기존 구간별 누락 기준(25% 이하)을 만족한 기록은 {r["input_quality_counts"].get("standard_window_gate_only",0)}개, 그 기준을 넘지만 기존 전처리로 숫자 입력이 만들어진 탐색용 기록은 {r["input_quality_counts"].get("high_missing_exploratory",0)}개다. 입력 불가 101기록도 삭제하지 않고 이유를 남겼다.','', '입력 불가 사유: '+', '.join(f'{k}: {val}' for k,val in r['unavailable_input_reasons'].items())+'.','', '**누락이 많은 자료의 숫자는 신뢰할 수 있는 보행 측정값으로 간주하지 않는다.** 기존 처리에 포함된 보간으로 관절 좌표가 채워져, 실제 관측이 부족해도 예측값이 나올 수 있다. 새로 패딩·시간 늘리기·정답 채우기를 하지는 않았다.','', '## 모델별 결과','', 'MAE는 오차의 절댓값 평균이며 작을수록 좋다. 편향은 예측−정답이며 음수면 과소예측이다. 사람별로 좌우와 해당 집단의 기록을 먼저 평균했다. 입력 품질별 집단은 정답 분포와 인원이 다르므로, 차이를 품질의 인과효과로 해석하지 않는다.']
 for co in ['primary_remaining','additional_final','additional_development']:
  for qu,title in [('standard_window_gate_only','기존 구간 누락 기준 통과'),('high_missing_exploratory','누락 기준 초과·탐색용')]:
   g=r['groups'][co+'__'+qu];z=g['fusion']['all'];lines+=['',f'### {cn[co]} / {title}', '',f'같은 {z["people"]}명·{z["records"]}기록을 17개 구성에서 비교했다.', '', '|구성|전체 MAE|편향|고득점 90 이상 MAE|고득점 편향|','|---|---:|---:|---:|---:|']
   for n in names:
    a=g[n];lines.append('|'+LABEL[n]+'|'+'|'.join(fmt(x) for x in [a['all']['mae'],a['all']['bias'],a['90plus']['mae'],a['90plus']['bias']])+'|')
   high=g['fusion']['90plus'];lines+=['',f'이 표의 고득점 집단은 사람 평균 90 이상 {high["people"]}명·{high["records"]}기록이다.']
 lines+=['', '## 정답이 없는 자료에서도 어떻게 반응하나','', '사람 연결 미확인 자료는 예측값과 모델 사이 차이만 기록했다. 서로 비슷한 점수를 내도 정확하다는 뜻은 아니다. 모델 17개의 최대−최소 점수 차이는 **모델 간 불일치**이며, 임상적 불확실성 구간이 아니다.','', '|자료|기록 수|17개 구성의 평균 최대−최소 차이|','|---|---:|---:|']
 for co,z in r['cohort_availability'].items():lines.append(f'|{cn[co]}|{z["input_available_records"]}|{fmt(z["disagreement_range_mean"])}|')
 lines+=['','### 고정 순서 예시','', '오차가 작거나 큰 사례를 고르지 않았다. 각 자료 구분·품질 조합의 원래 순서상 첫 기록을 보여준다. 모든 기록과 17개 모델별 좌우 예측은 대용량 CSV에 남겼다.','', '|구분|품질|실제 좌/우|Fusion 좌/우|RankSim 추가 좌/우|HuberRank 추가 좌/우|MAERank 추가 좌/우|','|---|---|---|---|---|---|']
 with (BIG/'all-record-predictions.csv').open(encoding='utf-8-sig',newline='') as f:allrows=list(csv.DictReader(f))
 examples=[]
 for co in cn:
  for qu in ['standard_window_gate_only','high_missing_exploratory']:
   selected=next((row for row in allrows if row['cohort']==co and row['quality']==qu),None)
   if selected is None:continue
   examples.append(dict(cohort=co,quality=qu,source_index=int(selected['source_index'])))
   def pair(a,b):return ('—' if not selected[a] else f'{float(selected[a]):.2f}')+' / '+('—' if not selected[b] else f'{float(selected[b]):.2f}')
   values=[pair('true_left','true_right')]+[pair(n+'_left',n+'_right') for n in ['fusion','plus_ranksim','plus_HuberRank1','plus_MAERank1']];lines.append('|'+cn[co]+'|'+('기준 통과' if qu=='standard_window_gate_only' else '탐색용')+'|'+'|'.join(values)+'|')
 lines+=['', '## 해석과 검증 범위','', '- 현재 모델 학습·직전 222명 평가와 사람이 겹치지 않는 원래 후보는 449명·771기록이다. 과거 다른 실험에서 쓰인 37명 검증·38명 시험 표본 등이 포함돼 있어, 전부 완전히 미사용인 독립 최종 시험이라고 부르지 않는다. 과거 표본은 results.json에서 별도로 구분했다.','- 과거 38명 시험용 정답: '+('사용자가 이번 비교를 승인해 개봉했다.' if policy['old38_numeric_labels'] else '보존했다. 이 38명은 예측만 기록했으며 정답·오차는 열지 않았다.'),'- 기존 사람의 추가 기록은 새로운 사람에 대한 일반화 성능과 구분한다. 정답이 연결되지 않은 영상은 사람 수나 정확도를 추정하지 않는다.','- Single CNN은 원래 고정 구간이 준비되지 않은 기록에도 적용하기 위해, 처음 선택 가능한 124프레임 구간을 사용했다. 이 점에서 과거 canonical 입력 선택 방식과 다르다. 나머지 구성은 선택된 구간들의 평균을 사용했다.','- 전처리는 원래와 같고, 먼저 누락 25% 이하인 유한 구간들을 사용했다. 그런 구간이 없을 때만 누락이 많은 유한 구간 전체를 탐색용으로 사용했다. 처음 500프레임 범위의 완전한 124프레임 구간만 사용했으며, 기존 정식 자료의 canonical 적격 기준을 전부 통과했다는 뜻은 아니다.','- 모델·입력·원본 자료 지문과 사람/영상/좌우 연결, 기존 원자료 전처리 재현, 사람별 오차의 독립 계산을 확인했다. 자세한 증거는 verification.json이다.','- 품질이 나쁜 입력의 오답을 새 학습·보정으로 고치거나 유리한 모델을 재선택하지 않았다. 외부 집단·고령자·스마트폰 촬영 정확도 검증이 아니다.','', '전체 4,275기록 예측·누락·정답 상태·실패 이유: D:/gait-research-data/D_P/gdi-unused-audit-20261105/all-record-predictions.csv. 요약 표는 model-comparison.csv, 결과 전체는 results.json이다.','']
 (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(HERE/'report-receipt.json',dict(passed=True,report_sha256=sha(HERE/'README.md'),results_sha256=sha(HERE/'results.json'),verification_sha256=sha(HERE/'verification.json'),fixed_example_membership=examples));print('Remaining-data report saved',flush=True)
if __name__=='__main__':main()
