import json,hashlib
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent;H=HERE.parents[1]
assert not (HERE/'study_receipt.json').exists()
assert json.loads((HERE/'verification.json').read_text(encoding='utf-8'))['verified']
assert json.loads((HERE/'supplement_verification.json').read_text(encoding='utf-8'))['mixed_training_plan_hash_recovered']
review=H/'RESEARCH_REVIEW.md';old=review.read_text(encoding='utf-8')
with (HERE/'prior_review.md').open('x',encoding='utf-8') as f:f.write(old)
intro='''# 최신 결과: 구간 민감도는 줄었지만 정확도 개선은 아직 아니다

2026-10-28 사용자 요청에 따라 구간별 관절 추정 품질·움직임을 분석하고, 저장 모델을 교차 적용한 뒤 여러 구간을 함께 학습하는 모델1개를 시험했다. 현재 우선순위는 앱이 아니라 보유 자료에서 유망한 알고리즘 방향을 찾는 것이다. GDI는 그 검증 기준 중 하나다.

**뒤쪽 관절 추정이 전반적으로 더 나빠진다는 설명은 지지되지 않았다.** 검증37명에서 하체 신뢰도는 앞0.702→뒤0.721로 높아졌고 낮은 신뢰도 비율은0.59%→0.25%로 줄었다. 발목 움직임과 몸통각의 화면상 변화도 있었지만 실제 보행·시점·추정 오류를 구분할 정답이 없어 원인 하나를 확정하지 않는다. 뒤 구간에 맞춰 학습한 기존 모델도 뒤 구간 오차를 개선하지 못했다.

**앞·중간·뒤를 함께 학습하자 점수 폭은5.07→4.27점으로 약15.8% 줄었다.** 하지만 앞/중간/뒤의 오차는 기존7.29/7.75/8.68에서7.60/7.85/8.46으로 바뀌었고, 세 구간 평균 오차는7.69→7.76이었다. 각 정확도 비교의 불확실성 범위는0을 포함한다. 따라서 새 모델을 개선안으로 채택하지 않는다. 모델 출력 자체도 약간 압축됐으므로 안정성 하나만으로 성공을 선언하지 않는다.

구체적인 후속 후보는 누락 관절 처리다. 검사한243프레임 입력의 겹치는 세 구간에서0신뢰도 관절7,558개 관측 모두의 x 또는 y가 정규화 범위 경계에 있었다. 주로 얼굴·손목 등에서 누락이 있었고, 인코더 코드에는 신뢰도0 관절을 명시적으로 제외하는 처리가 없다. 누락이 점수 차이의 원인이라는 증명은 아직 없으므로 같은 관측된 움직임을 유지한 채 누락 처리만 바꾸는 대조 실험으로 판단해야 한다. 이미 실행한 구간 품질 집계·모델3×3 교차 적용·혼합 구간 학습1회를 반복하지 않는다.

GDI 외 대안으로 같은 공개 파일의 머리글에서 보행 리듬(cadence),속도(speed),무릎각 열을 확인했다. 개인별 행은 압축 해제·파싱하지 않았고 정답 연결·단위·결측은 미확인이다. 설명 가능한 개별 지표를 예측하는 방향이 현재 자료에서 가장 가까운 대안이다. 범주화 의미는 CARE-PD의 임상 보행 등급0~3이 더 직접적이지만 보유43명·90구간/최고등급5명과SMPL입력의 제약이 있다. 새 목표·임상척도·10등급·평가 기준은 임의로 채택하지 않는다. GDI를 바꾸는 것만으로 입력 민감도가 사라지지는 않는다.

입력 품질은 개발212명·243건에서 학습/검증을 나눠 집계했고, 주요 비교는 기존 검증37명·40건이다. 혼합 학습의609구간은175명·203기록에서 나온 것이며 사람 수가 늘지 않았다. 구간당1/3가중치와alpha=1,학습 사람만의 표준화를 유지했다. 기존 평가38명·최종223명은 사용하지 않았다. 결과1,249행의 집계·상관·MAE를 별도 코드로 재계산했고 저장 모델과 두 Ridge 대수식도 확인했다. 기존 자료·모델·앱은 수정하지 않았다.

[실험 방법·수치·원인 후보·GDI 대안·권장 순서](studies/20261028-position-quality/report.md)에 전체 분석을 남겼다. 이번에 실제 한 것은 원인 진단과 제한된 개선 실험이며, 새로운 점수 체계나 임상 개선 효과를 입증한 것은 아니다. 종료 이유는 요청한 입력 점검·학습 조건 비교·대안 검토·검산을 마쳤기 때문이다. 다음은 누락 처리 대조 검사이며, 다른 점수 목표 채택은 사용자 선택으로 남긴다.

## 이전 결과와 선택 경과

'''
text=intro+old.rstrip()+'\n\n<!-- study:20261028-position-quality -->\n'
review.write_text(text,encoding='utf-8')
with (HERE/'review_snapshot.md').open('x',encoding='utf-8') as f:f.write(text)
artifacts=sorted(p.relative_to(H).as_posix() for p in HERE.iterdir() if p.is_file() and p.name not in ('response.json','study_receipt.json'))
response=dict(status='completed',summary='입력 품질·움직임 분석과 구간별 모델 교차 진단, 혼합 구간 학습1회 완료. 뒤 구간 품질 저하 가설은 지지되지 않았고 혼합 학습은 점수폭을5.07→4.27로 줄였으나 정확도 개선은 불확실. 누락 관절 처리와 설명 가능한 보행 지표를 후속 후보로 제시.',artifacts=artifacts,
 checks=['원본·특징·모델 지문 불변','기존 사람 분리 유지 및 평가38명·최종223명 미사용','품질729행·교차예측360행·혼합예측160행 독립 집계 및 순위상관 검산','Ridge SVD/직접방정식 일치 및 저장모델 재현','대안 CSV 머리글만 확인: 개인별 행 압축 해제0건','통합 검토 보고와 보존본 일치'],
 next_goal='같은 관측된 움직임을 유지하며 신뢰도0 관절 좌표 처리만 바꾸는 대조 실험을 준비하고 실행해 현재 구간 민감도에 미치는 영향을 구분한다. 새 임상 기준이나 자료 제외를 도입하지 않고 기존 분리와 정확도/안정성 비교를 유지한다. 개별 보행 지표나 임상 보행 등급을 새 주 목표로 삼는 선택은 사용자에게 남긴다.',
 why_next='혼합 구간 학습의 안정성 개선이 정확도로 이어지지 않았고, 누락된 얼굴·손목 좌표가 입력에 남는 구체적인 처리 지점을 확인했다.',needs_user_reason='')
with (HERE/'response.json').open('x',encoding='utf-8') as f:json.dump(response,f,ensure_ascii=False,indent=2)
receipt=dict(kind='manual_user_requested_study',status='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
 artifacts=[dict(path=p,sha256=hashlib.sha256((H/p).read_bytes()).hexdigest()) for p in artifacts],
 response_sha256=hashlib.sha256((HERE/'response.json').read_bytes()).hexdigest())
with (HERE/'study_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
assert review.read_bytes()==(HERE/'review_snapshot.md').read_bytes()
print(json.dumps(dict(completed=True,artifacts=len(artifacts),review_snapshot_equal=True)))
