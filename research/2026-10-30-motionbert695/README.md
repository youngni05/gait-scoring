# MotionBERT 자료 확대·부분학습 실험

**배포 목적은 대학 과목 제출용 비상업적 연구 결과 공유다.** 모델 압축에 Stanford 학습 자료와 MotionBERT 사전학습 모델의 출처·라이선스 원문·부분학습 및 공개 변환 내역을 동봉했다. [출처·조건](../licenses/bundles/motionbert/SOURCES.md), [수정 내역](../licenses/bundles/motionbert/MODIFICATIONS.md), [적용 범위](../licenses/README.md)를 함께 확인한다.

2026-10-30에 수행한 실험이다. 사전학습된 MotionBERT가 움직임을 요약하고 작은 회귀 출력부가 좌우 GDI를 예측한다. 자료를 늘린 뒤 본체 일부도 GDI에 맞게 조정했을 때 오차가 줄어드는지 확인했다. **같은 695명의 MAE는 고정 MotionBERT+Ridge 8.526630점에서 마지막 단계 부분학습 8.354702점으로 약 0.172점 감소했다.**

## 모델이 배우는 것과 실험 순서

입력은 공개자료의 전처리된 124프레임·17관절·x/y/관측 신뢰도다. 정답은 같은 기록의 실제 왼쪽·오른쪽 GDI다. MotionBERT-Lite의 프레임·관절별 표현을 평균한 512개 특징 → 학습자료에서 계산한 표준화 → 좌우 출력부의 순서로 예측한다. 세 번째 입력 채널은 깊이가 아니다.

1. 기존 175명·203기록과 추가 520명·861기록, 총 695명·1,064기록을 준비했다. 기존 사람 분할을 유지하고 추가 사람은 seed 20261030으로 배정했다.
2. 매회 평가 139명을 제외한 556명에서만 표준화와 Ridge(alpha=1)를 적합했다. 이 계수로 각 분할의 출력부를 초기화했다.
3. 동일 초기값에서 `head_only`(출력부), `tail`(출력부·마지막 특징 변환부), `last_stage`(마지막 Transformer 단계까지)의 세 범위를 비교했다. 마지막 범위에서도 본체 전체를 학습한 것은 아니다.
4. 방법별 5개 분할에서 200회씩, 총 15개 학습·3,000회 갱신을 완료했다. 각 사람은 자신이 학습에 포함되지 않은 모델로 평가했다.

배치 32, AdamW, 출력부 학습률 1e-4, 본체 학습률 1e-5, weight decay 0, dropout 비활성화, seed 20261028을 고정했다. 평가 결과로 갱신 횟수나 학습률을 고르지 않았다. 설정은 [protocol.ready.json](protocol.ready.json), 학습 범위는 `run.py`의 `selected()`에 있다.

## 관찰한 결과

| 방법 | 사람별 MAE↓ |
|---|---:|
| 고정 MotionBERT+Ridge | 8.526630 |
| 출력부만 학습 | 8.528939 |
| 마지막 특징 변환부까지 학습 | 8.614338 |
| 마지막 Transformer 단계까지 학습 | 8.354702 |

마지막 단계 부분학습의 Ridge 대비 감소량은 약 2.02%다. 5분할 중 4개, 사람별로는 356명에서 오차가 줄고 339명에서 늘었다. 감소량의 사람 재표집 95% 범위는 0.0358~0.3070점이었다. 고정 예측을 재표집한 탐색적 범위로, 다른 초기값·재학습·반복 탐색의 변동을 모두 포함하지 않는다.

10점 구간 정확 일치율은 35.81%→35.22%로 감소했다. 평균 절대오차의 감소를 모든 구간 판정의 개선으로 해석하지 않는다. [집계](results/summary.json), [전체·집단별 수치](results/metrics.csv), [분할별 수치](results/fold_metrics.csv), [점수대별 수치](results/score_bands.csv)를 함께 확인한다. 보호해 둔 최종 자료나 외부 기관 자료에서 독립 검증한 결과는 아니다.

## 가중치로 예측하기

[motionbert-initial695-weights.tar.gz](https://github.com/youngni05/gait-scoring/releases/download/research-initial-models-20261101/motionbert-initial695-weights.tar.gz)는 `last_stage`의 5개 분할 가중치다. 각 파일은 본체·출력부·표준화 값을 포함하므로 추론 때 원 사전학습 가중치나 Ridge 초기값 파일을 따로 받을 필요가 없다. `head_only`·`tail`의 가중치는 이번 배포에 포함하지 않았다.

```bash
python -m pip install -r research/2026-10-30-motionbert695/requirements.txt
python research/2026-10-30-motionbert695/predict.py --weights /path/to/motionbert/fold0.pt --input /path/to/prepared_windows.npy --output /path/to/gdi_predictions.npy
```

입력은 원 실험과 같은 전처리를 마친 float32 `[기록 수,124,17,3]` `.npy` 배열이며, 관절 순서는 [공통 입력 설명](../2026-10-30-cnn-baseline695/input_contract.json)에 있다. 출력은 `[기록 수,2]` 왼쪽·오른쪽 GDI이며 범위를 자르지 않는다. 평가값을 재현할 때는 해당 사람을 학습하지 않은 분할 모델을 사용한다. 5개 모델을 평균하는 별도 앙상블을 평가한 결과는 아니다.

MotionBERT 코드와 사전학습 모델의 출처는 [공식 MotionBERT 저장소](https://github.com/Walter0807/MotionBERT)이며, 포함한 `lib/`는 원 실행 묶음의 사본이다. [MOTIONBERT_LICENSE](MOTIONBERT_LICENSE)를 보존한다. 원본 데이터 이용 조건과 코드 라이선스는 별도로 확인해야 한다.
