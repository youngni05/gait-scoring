import json,hashlib,statistics,datetime
from pathlib import Path
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
a=load(S/'result.json');prep=load(S/'preparation.json');inter=load(S/'interpretation.json');verify=load(S/'verification.json');assert verify['verified']
for n,d in inter['source_sha256'].items():assert sha(S/n)==d
prior={}
for folder,names in [(H/'control/desktop-sessions/20261026T041807Z-f050df',['report.md','kinematic_comparison_result.json','internal_cv_protocol.json']),(H/'control/desktop-sessions/20261028T000001Z-753881',['report.md','result.json'])]:
    receipt=load(folder/'receipt.json');sealed={z['path']:z['sha256'] for z in receipt['artifacts']}
    for name in names:
        p=folder/name;rel=p.relative_to(H).as_posix();assert sha(p)==sealed[rel];prior[rel]=sha(p)
save('prior_evidence.json',prior)
save('research_sources.json',{'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':[{'url':'https://arxiv.org/html/2210.06551v5','purpose':'MotionBERT noisy/partial 2D representation learning and downstream adaptation; current masked output pooling is our diagnostic, not a paper-reproduced GDI intervention.'},{'url':'https://roc-hci.com/wp-content/uploads/Ataxia_severity_ML_for_healthcare__UbiComp_.pdf','purpose':'Auto-Gait uses pose-derived gait features and SARA clinical labels on155videos/89people; supports explicit-feature comparison, not our spectral10-feature formula or GDI performance.'},{'url':'https://www.nature.com/articles/s41467-020-17807-z','purpose':'Search-index primary paper text names image-plane joints, knee/ankle angles and inter-ankle distance. Direct page access failed; no new numeric claim based on full-text reading in this turn.'}],'limitations':['The10spectral/coupling descriptors are a project-defined fixed diagnostic inspired by explicit motion features, not a reproduction of the cited feature sets.','Publicly readable papers do not imply dataset acquisition or product-use rights.'],'nonblocking_tool_errors':['Root git status reported not a work tree; no Git change was made.','An assumed uncertainty-script path did not exist; actual fixed protocol and saved OOF files were inspected.','Windows wildcard path search failed once; explicit paths used.','PMC returned a browser check; read the author-hosted Auto-Gait PDF instead.','Nature page redirected to inaccessible identity endpoint; author-hosted PDF fetch timed out. No retry bypass.']})
labels={'baseline':'기존 AI 특징','valid_pool':'누락 위치를 빼고 AI 특징 평균','rhythm':'반복성·좌우관계10개만','combined':'기존 AI 특징+새10개'}
rows=[]
for m,label in labels.items():
    z=a['metrics'][m];e=z['person_mae'];rows.append(f"| {label} | {e['front']:.2f} | {e['middle']:.2f} | {e['last']:.2f} | {e['mean3']:.2f} | {z['range_mean']:.2f} |")
table='\n'.join(rows)
df={m:statistics.mean(z['effective_df'] for z in a['fit_details'] if z['method']==m) for m in labels}
report=f'''# 어느 방향이 지금 자료에서 더 가능한가

**이번에 시험한 두 구현 중에서는 누락을 고려한 집계가 기존 성능을 더 잘 유지했다. 그러나 개선안으로 채택할 정도의 이득은 없었다.** 설명 가능한 반복성 특징은 단독·결합 모두 악화됐다. 처음에는 설명 가능한 특징을 우선 탐색할 가치가 더 높다고 보았지만, 이번 GDI예측 결과는 그 정확도 우위를 지지하지 않는다.

지금의 선택은 ‘누락을 고려한 모델 학습을 후속 연구 후보로 남기고,단순 평균 변경은 채택하지 않는다’이다. 설명 가능한 특징 전체가 쓸모없다는 결론도 아니다. 이번에 실패한 것은 짧은124프레임에서 추출한10개 반복성·좌우관계 숫자를 고정 Ridge에 넣은 조건이다. 새 임상 점수나 실제 속도 측정의 가능성까지 검증한 것은 아니다.

## 두 방향의 장점과 제약

| 비교 항목 | 누락을 고려한 특징 처리 | 설명 가능한 보행 특징 |
|---|---|---|
| 답하려는 질문 | 관절이 보이지 않는 위치가 결과에 미치는 영향을 줄일 수 있는가 | 움직임의 반복성·좌우관계가 GDI예측에 추가 정보를 주는가 |
| 현재 보유 입력 | 모든 관절 좌표와 신뢰도0표시 | 연속 관측된 하체6관절의2D좌표 |
| 이번 변경 | 인코더 출력 중 신뢰도0위치를 최종 평균에서 제외 | 발 사이 거리·양쪽 무릎각의 시간 변화를10개 값으로 요약 |
| 기존 실험과 차이 | 좌표0치환이나 다리만 고정 선택이 아님 | 기존 무릎각 크기·분포16개 요약과 달리 반복성·좌우 시간관계 |
| 실용적 장점 | 기존 특징·모델 구조를 크게 바꾸지 않는 작은 대조 가능 | 계산 항목과 비용을 설명하기 쉬움 |
| 큰 제약 | 인코더 내부에서 이미 관절끼리 섞인 정보까지 제거하지 못함 | 짧은 구간·촬영각·투영·검출 변화에 민감하며 실제 보행 지표 정답이 아님 |
| 이번 판단 | 정확도 거의 동일,구간 차이는 증가: 보류 | 단독·결합 오차와 구간 차이 증가: 이번 구현 보류 |

원래 MotionBERT 연구는 일부가 빠지거나 잡음이 있는2D관절로 표현을 학습하고,다른 과제에 맞게 예측 부분이나 인코더를 적응시킨다. 따라서 최종 평균에서 위치를 빼는 이번 대조와,누락 상황을 포함해 인코더를 학습시키는 방법은 다르다. 후자를 적용한다면 같은 학습량의 일반 미세조정과 누락 증강 미세조정을 비교해야 한다. 우리 GDI자료에서 그 효과를 확인한 것은 아니다. [MotionBERT 원 논문](https://arxiv.org/html/2210.06551v5)

Auto-Gait은155영상·89명에서 관절 좌표로 보행 특징을 만들고,SARA보행 평가와 연결했다. 우리에게 가져온 원칙은 ‘설명 가능한 움직임 특징을 별도 비교군으로 둔다’이다. 이번 주파수·엔트로피10개는 프로젝트에서 고정한 탐색용 계산식이며 그 논문을 그대로 재현한 것이 아니다. 질환·정답·자료가 달라 논문 성능을 우리 성능으로 가져오지 않는다. [Auto-Gait 저자 제공 논문](https://roc-hci.com/wp-content/uploads/Ataxia_severity_ML_for_healthcare__UbiComp_.pdf)

## 이번 실험은 기존 실험과 무엇이 다른가

이전에 실행한 것은243프레임의 무릎각 평균·분산·좌우 차이16개,AI특징의 시간3분할·다리통합·좌우분리,누락좌표0치환이었다. 그 결과와 완료 지문을 확인했다. 이번에는 원래 좌표를 유지한 유효 위치 평균과 새로운 시간 특징을 직접 비교했다. 기존 조건의 단순 재실행은 아니다.

검증37명을 더 맞추지 않기 위해 **원래 학습175명·203기록만** 사용했다. 기존의 사람별5묶음을 그대로 사용해,매번140명으로 표준화·학습하고 남은35명을 예측했다. 각 사람은 자신의 기록을 학습에 넣지 않은 모델로 평가했다.5묶음이 끝나면175명 전체의 분리 예측이 모인다. 학습된 모델의 학습 오차를 보고한 것이 아니다.

기존 검증37명,기존 최종38명,새 보류223명은 계산·모델 선택에 쓰지 않았다. 공용npz에서 학습행을 고르는 과정에서 배열 전체가 일시적으로 해제될 수 있지만 비학습행은 계산에 넣지 않았다. 정답은 학습203행만 있는 기존 OOF파일에서 읽었다. 이175명은 새로 확보한 외부 표본이 아니며,이전 연구에서 사용한 개발 자료다.

입력은 앞·중간·뒤 각각124프레임이고,정규화·원본·GDI정답·사람 분리·Ridge alpha=1을 고정했다. 각묶음 모델은 앞 구간으로만 학습한 뒤 세 위치를 예측했다. 새로운 후보3개×5묶음,기존 모델 재현5회로 비교했고 이후 설정을 바꾸어 재학습하지 않았다. 인코더를 새로 학습하지 않았다.

## 결과

오차는 GDI정답과 예측의 차이를 사람별로 평균한 값(MAE)이다. 구간 폭은 같은 영상의 앞·중간·뒤 예측 중 최대−최소다. 두 값 모두 작을수록 유리하지만,구간 폭이 실제 재촬영 오차나 건강 변화의 기준은 아니다.

| 조건 | 앞 오차 | 중간 오차 | 뒤 오차 | 세 예측 평균 오차 | 구간 점수 폭 |
|---|---:|---:|---:|---:|---:|
{table}

이번 기존 오차8.28은 학습175명의 내부 분리 평가다. 지난 검증37명의7.29와 대상·학습량이 달라 두 숫자를 직접 비교하면 안 된다.

- 누락 위치 제외: 앞 오차 감소0.018점,짝 재표집95% 범위[−0.196,0.234].5묶음 중3묶음만 개선돼 확실한 이득이 아니다. 구간 폭은0.261점 증가했고 감소량 범위[−0.485,−0.042]는 이번 표본에서 악화 쪽이다.
- 시간 특징만: 앞 오차1.400점 증가. 앞 오차는5묶음 중1묶음만 개선됐고 구간 폭도1.440점 증가했다.
- 기존+시간 특징: 앞 오차0.599점 증가,감소량 범위[−0.973,−0.253].앞 오차가5묶음 모두에서 커졌고 구간 폭도1.653점 증가했다.

재표집은 기존seed20261026으로175명의 오차 차이를10000번 뽑았다. 묶음별 학습 자료가 겹치고 여러 후보를 비교했기 때문에 이 범위를 독립적인 임상 유의성 검정으로 해석하지 않는다. 결과가 조금 좋게 나온 위치만 골라 개선이라고 보고하지 않았다.

표준화 후 같은alpha=1을 사용했지만 모델의 유효자유도는 기존평균{df['baseline']:.2f},누락제외{df['valid_pool']:.2f},시간특징{df['rhythm']:.2f},결합{df['combined']:.2f}였다. 결합 결과에는 정보 추가와 복잡성 증가가 함께 들어가며 정확한 원인을 한 가지로 확정하지 않는다. 이번 결과에 맞춰alpha를 새로 고르지 않았다.

## 반복성 특징이 왜 도움이 되지 않았는가

계산식 자체의 오류는 합성 검사와 수치 재현에서 발견하지 못했다. 다만 같은 영상도 구간을 옮기면 분석한 시간 특성이 많이 달랐다. 앞·뒤의 가장 큰 주파수 성분이 바뀐 기록은 발 사이 거리155/203(76.4%),오른쪽 무릎128/203(63.1%),왼쪽 무릎134/203(66.0%)이었다. 주파수 한 칸 차이도 포함한 기술 통계이며,실제 걸음 속도가 그만큼 달라졌다는 뜻은 아니다.

결합 모델의 앞→뒤 예측 변화를 저장된 선형식으로 나누면,새10개 항의 합만으로도 평균 절대2.88점이 움직였다. 전체 예측 변화는4.97점이었다. 기존512개 항의 변화도 함께 있고 서로 상쇄될 수 있으므로2.88/4.97을 ‘원인의 몇%’로 계산하지 않는다. 이 분해는 임상적인 특징 중요도가 아니라 이번 모델의 계산상 기여다.

현재 근거는 **짧은 구간의 반복성 숫자를 붙이는 것만으로 구간 흔들림을 줄이지 못했다**는 것이다. 원인이 실제 움직임 차이인지,시점·검출 변화인지,짧은 분석 길이나 큰 성분을 하나 고르는 계산 방식인지는 분리되지 않았다. 따라서 설명 가능한 지표를 주 결과로 삼으려면 먼저 그 지표 자체의 측정 정확성과 구간 일관성을 확인해야 한다.

## 지금 추천하는 방향

현재GDI예측을 이어간다면 **누락을 고려한 표현 학습을 다음 후보로 남기되,단순 평균 변경을 개선 성과로 채택하지 않는다.** 이번에는 이쪽이 기존 성능을 유지했고 시간 특징보다 나았지만,미세조정까지 성공한다는 보장은 없다. 주요 설계 변경에 앞서 다음 최소 비교를 제안한다.

- 기존동결모델 / 같은 학습량의 일반적인 작은 예측부 적응 / 누락 상황을 추가한 적응을 구분한다. 추가 학습 자체의 효과와 누락 대응 효과를 섞지 않는다.
- 준비 단계에서 관측 위치의 학습용 마스크와 구간을 같은 형식으로 만들 수 있는지 확인한다. 새 가림은 실제 누락 복원 정답으로 취급하지 않는다.
- 기존 내부 사람 분리에서 오차와 구간 차이를 함께 확인한다. 검증37명을 후보 선택에 추가 사용하거나 최종38명·223명을 열지 않는다. 정확한 변경 범위·학습 예산 등 주요 설계는 사용자와 정한다.

설명 가능한 방향은 ‘10개 시간 숫자를 GDI에 더 넣기’보다 **실제로 검증 가능한 개별 지표를 정하고,그 지표의 정답과 비교하는 과제**로 별도 검토할 가치가 있다. 현재 공개 파일에 속도·리듬·무릎각 열이 있다는 것과,지금 사람이 쓸 정확한 지표를 만들 수 있다는 것은 다르다. 임상목표·정답을 바꾸는 선택은 이번에 실행하지 않았다.

실용성에서는 시간 특징 계산 자체가609구간 합계{prep['rhythm_cpu_seconds']:.3f}초,인코더+집계는{prep['encoder_and_pool_cpu_seconds']:.2f}초였다. 같은CPU에서 좌표를 받은 뒤 측정한 참고 시간이며,영상 관절 추정·파일읽기·모바일·서버 대기·전체앱 속도를 포함하지 않는다. 더 빠르다는 이유로 부정확한 점수를 제출하지 않는다.

## 저장한 코드와 확인 범위

핵심 흐름은 temporal_features.py가 시간 특징10개를 계산하고,prepare.py가 원래특징·누락제외특징·시간특징을 같은 학습행에 연결하며,analyze.py가 고정5묶음의 Ridge를 학습·평가하는 것이다. verify.py는 결과표를 별도 계산으로 검사한다. NumPy는 배열·주파수·선형계산,PyTorch는 기존 인코더 추론에 사용했고 새 패키지는 설치하지 않았다. interpret.py는 이미 저장한 예측식을 분해한다.

- 합성신호의 알려진주파수·일정신호 처리,이동·균일크기·평면회전 불변,좌우교환,누락된 하체/길이0 입력 거부를 확인했다. 실제 자료에서 평평한 신호나 상관 계산불가로 대체된 경우는0건이었다.
- 기존특징609회 재현 오차0,유효 위치 평균의 두계산 차이7.75e−9이하,누락없는69구간의 평균 일치를 확인했다.
- 기존 앞 구간OOF예측 재현차이1.14e−13,두 Ridge수식 계수 차이1.52e−11이하였다.
- 예측3248행의 사람별·묶음별 MAE,구간 폭,세 구간 평균식,출력표준편차를 표준라이브러리로 재계산했다. 저장모델도 특성별 곱의 합으로 재현했다.
- 특징609구간 연결·원본지문·고정분리·각기록단1회OOF예측을 확인했다. bootstrap 계산은 독립 구현으로 다시 검사하지 않았다. 영상이나 실측각도 정답으로 특징의 임상 정확성을 검증한 것은 아니다.

초기 파일 검색·Git상태·웹 접근 일부가 실패했으나 실제 실험과 검산은 통과했다. 해당 제한은 research_sources.json에 남겼다. 원본·기존앱·감독프로그램·운영설정·기존증거는 수정하지 않았다. 쓰기는 새 연구폴더와 통합 검토보고에만 했다.

이번 종료 이유는 고정한 두방향 및 결합 비교,독립검산,새 시간 특징의 구간변화 해석을 완료했기 때문이다. 결과를 개선시키려고 다른 주파수대나모델을 계속 시도하지 않았다. 새 후보의 채택을 보류한다는 결과도 그대로 보존한다.
'''
with (S/'report.md').open('x',encoding='utf-8') as f:f.write(report)
old=(H/'RESEARCH_REVIEW.md').read_text(encoding='utf-8')
with (S/'prior_review.md').open('x',encoding='utf-8') as f:f.write(old)
intro='''# 최신 비교: 누락 제외는 거의 동일, 시간 특징 추가는 악화

2026-10-28 사용자 요청에 따라 기존 실험을 확인하고 새 두 방향을 비교했다. 과거 무릎각16개·다리/시간풀링·좌표0치환을 반복하지 않았다. 이번은 원래 입력의 신뢰도0위치를 최종 특징 평균에서 빼는 방법과,발 사이 거리·양쪽 무릎각의 반복성·좌우관계10개를 비교한 것이다.

원래 학습175명203기록의 기존 사람별5묶음만 사용했다. 각사람은 자신을 학습에 넣지 않은 모델로 예측했다. 기존검증37명·최종38명·예약223명을 새 비교에 사용하지 않았다. 따라서 다음8.28을 이전 검증37명의7.29와 직접 비교하지 않는다.

| 조건 | 앞 구간 사람별오차 | 앞·중간·뒤 점수 폭 |
|---|---:|---:|
| 기존AI특징 |8.28|4.72|
| 누락 위치를 빼고 평균 |8.26|4.99|
| 시간특징10개만 |9.68|6.16|
| 기존AI+시간10개 |8.88|6.38|

누락 제외의 오차 감소0.018점은 불확실성 범위[−0.196,0.234]안이고 구간 폭은 증가했다. 시간특징을 더한 조건은 앞 오차가5묶음 모두에서 커졌다. 이번 조건들로 개선을 주장하거나 새 모델을 채택하지 않는다. 초기에는 설명 가능한 특징을 우선 탐색할 가치가 있다고 보았지만,이번 정확도 결과는 그 우위를 지지하지 않았다.

후속 분해에서 발 사이 거리의 가장 큰 주파수 성분은 앞·뒤가203기록 중155기록에서 달랐다. 결합 모델의 새10개 항도 구간 변화에 평균 절대2.88점 기여했다. 실제보행·시점·검출·짧은구간의 영향을 구분하지 못했으므로 임상 원인이나 실제리듬 변화로 해석하지 않는다.

현재GDI개선을 이어갈 때 누락을 고려한 표현 학습을 상대적 후속 후보로 남기고,단순 평균 변경은 보류한다. 이것이 미세조정 성공을 예측하는 근거는 아니다. 설명 가능한 방향은 검증 가능한 개별지표·정답을 선택하는 별도 과제로 검토할 가치가 있다. 새주목표·척도·학습구조의 주요선택은 사용자 몫이다.

다음 독립 작업은 누락 증강을 포함한 작은 적응 비교의 입력 준비성·기존 코드 재사용 범위·필요 학습조건을 점검하고 선택지를 구체화하는 것이다. 새척도·새분할·GPU작업·파인튜닝은 임의 채택하지 않는다. 이번 두구현·기존16개요약·구간품질·누락0치환을 다시 학습하지 않는다.

원래 특징 재현,합성신호·좌표변환 검사,예측3248행 독립집계와 저장모델 재현,원본지문을 확인했다. 실제촬영·임상 정확도나 독립최종 성능은 미검증이다. 종료 이유는 고정비교·검산·변화 원인후보 해석을 마쳤기 때문이며,후속상수를 검증결과에 맞추지 않았다.

[두 방향의 비교·실험 방법·전체 결과·다음 선택지](studies/20261028-rhythm-versus-missing-pooling/report.md)

## 이전 근거와 선택 경과

'''
text=intro+old.rstrip()+'\n\n<!-- session:20261028-rhythm-versus-missing-pooling -->\n'
(H/'RESEARCH_REVIEW.md').write_text(text,encoding='utf-8');(S/'review_snapshot.md').write_text(text,encoding='utf-8')
artifacts=sorted(p.relative_to(H).as_posix() for p in S.rglob('*') if p.is_file() and p.name not in ('response.json','study_receipt.json'))
response={'status':'completed','summary':'기존 학습175명의 고정5묶음에서 누락 위치 제외·시간특징10개·결합을 비교했다. 앞 오차는 기존8.28,누락제외8.26,시간특징9.68,결합8.88이며 구간폭은모두 증가했다. 어느 후보도 채택하지 않았다. 검증37명·최종38명·예약223명은 계산에 사용하지 않았다.','artifacts':artifacts,'checks':['기존OOF재현','합성 주파수 및 좌표변환검사','유효위치 집계 수식 및69무누락 구간 대조','3248예측행 독립집계·저장모델재현','609구간 연결·원본지문·고정사람분리','선형항분해 합과실제예측차이일치','통합보고 보존본 일치'],'next_goal':'누락 증강을 포함한 작은 적응 비교의 입력 준비성·기존 코드 재사용 범위·필요 조건을 점검한다. 새척도·분할·파인튜닝·GPU사용은 자동채택하지 않는다. 완료된 단순pooling·시간10개·무릎16개 비교를 반복하지 않는다.','why_next':'두 새 구현 모두 개선되지 않았다. 누락 제외는 기존성능을 유지하지만 구간 민감도를 해결하지 못했고,시간특징은 추가 흔들림이 있었다. 새로운 구조 변경 전에 최소 대조와 필요한 조건을 구체화해야 한다.','needs_user_reason':''}
save('response.json',response)
save('study_receipt.json',{'kind':'manual_user_requested_study','status':'completed','finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'artifacts':[{'path':p,'sha256':sha(H/p)} for p in artifacts],'response_sha256':sha(S/'response.json')})
assert (H/'RESEARCH_REVIEW.md').read_bytes()==(S/'review_snapshot.md').read_bytes()
print(json.dumps({'completed':True,'artifacts':len(artifacts),'snapshot_equal':True}))
