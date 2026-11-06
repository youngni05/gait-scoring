# 논문 관련 실험

2026-11-06 로컬 논문 초안에 사용한 근거와 관련 후속 실험을 구분했다. 논문 제출·게재나 최종 모델 채택을 의미하지 않는다. 원고 자체는 이번 모델·실험 정리 범위에 포함하지 않았다.

## GDI 편향과 정상 가정 점수

최근 축약 원고 `GDI_논문_축약수정본`과 기존 `gdi-bias-normal-label-draft`에 연결된 실험이다.

| 논문에서 확인한 질문 | 관련 코드·기록 |
|---|---|
| 고정 모델의 별도 임상자료 성능 | [54개 기준 모델](../selected/frozen-gdi/README.md), [222명 평가](../archive/gdi-research-harness/studies/20261102-final223-frozen/README.md) |
| 낮은 점수의 과대예측·높은 점수의 과소예측 | [개발자료 편향 분석](../archive/studies/20261104-ljmu100-mixed/README.md), [관측 점수별 진단](../archive/gdi-research-harness/studies/20261105-gdi-case-audit/README.md) |
| 성인 정상 자료에100점을 가정해 추가했을 때 | [정상100 추가75개 모델](normal100-models/README.md), [혼합 학습 비교](../archive/studies/20261104-ljmu100-mixed/) |
| 임상 학습자료만으로 다시 보정했을 때 | [Stanford 전용 보정 분석](../archive/studies/20261105-unseen-normal-test/stanford-calibration/README.md) |
| 정상 보행 특성 대신 자료 출처를 구분해도 높은 출력이 나오는가 | [출처 대조·외부 전이 검사](../archive/studies/20261105-normal-transfer-followup/README.md) |
| 실제 GDI 없는 외부 아동·직접 촬영 영상의 출력 변화 | [외부 비교](../archive/studies/20261105-unseen-normal-test/README.md) |

가정100점은 측정 GDI가 아니다. 임상자료로만 보정한 정상 추가 결합의 개발 MAE는6.655점으로 기준6.516점보다 커졌다. 외부 정상 보행의 출력 상승을 실제 GDI 정확도 향상으로 해석하지 않는다. 이 정상 추가 모델은 선별 기준 구성과 분리했다.

관련 후속 탐색은 [인원·비중을 맞춘 성인/아동 비교](../archive/studies/20261105-matched-normal100/README.md), [가정 점수·비중 민감도](../archive/studies/20261105-normal-label-sensitivity/README.md), [A~D 사전학습 비교](../archive/studies/20261105-gdi-masked-pretrain-v3/README.md), [고령자 투영좌표 탐색](../archive/studies/20261106-older-normal100/README.md)이다. 후속 결과를 기존 원고에 자동 포함한 것으로 취급하지 않으며, 각 문서의 마지막 확인 시점·미완료 상태를 따른다.

## 평균 오차와 모니터링 적용 조건

`gait-monitoring-short-paper-draft`의 근거다. 센서·투영좌표·실제 RGB 영상의 결과를 구분한다.

| 논문의 비교 | 관련 실험 |
|---|---|
| TUG에서 속도 외 특징의 추가 가치 | [FRAILPOL 개선 비교](../archive/studies/20261103-frailpol-improvement-loop/) |
| SPPB에서 속도 요약5개와 특징15개 비교 | [TabICLv2·TabPFN·LightGBM](../archive/studies/20261103-sppb-tabular-foundation/RESULTS.md) |
| TUG를 다른 자료로 옮겼을 때 | [TUG·SPPB 후속 분석](../archive/studies/20261105-tug-sppb-resume/) |
| 입력 오차와 저기능군 놓침 | [속도 입력 오차](../archive/gdi-research-harness/studies/20261101-mobilise-speed-error/), [분류 비교](../archive/gdi-research-harness/studies/20261031-mobilise-input-classification/) |
| 같은 사람의 방문 간 점수 구간 차이 | [개인 내 비교](../archive/gdi-research-harness/studies/20261101-mobilise-within-person/) |
| 모델 결합의 별도 평가 | [고정 GDI 평가](../archive/gdi-research-harness/studies/20261102-final223-frozen/README.md) |

후속 측정 검증은 [실제 RGB 대 광학 마커 비교](../archive/studies/20261105-planar-measurement-validation/RESULTS.txt), [서 있는 신장 보정](../archive/studies/20261105-standing-height-calibration/)에 분리했다. 이 결과가 기존 모니터링 원고에 이미 포함됐다는 뜻은 아니다. 촬영 각도·걸음 수가 부족한 경우는 측정 불가로 취급한다.
