# 초기 1D-CNN 기준 실험

**배포 목적은 대학 과목 제출용 비상업적 연구 결과 공유다.** 모델 압축의 `RESEARCH_USE.md`, `SOURCES.md`, `MODIFICATIONS.md`, `NOTICE.md`, `LICENSES/`에 출처와 조건을 동봉했다. [CNN 묶음의 출처·라이선스](../licenses/bundles/cnn/SOURCES.md)와 [수정 내역](../licenses/bundles/cnn/MODIFICATIONS.md)을 함께 확인한다.

2026-10-30에 수행한 실험이다. 관절 움직임을 작은 CNN이 직접 학습하면 기존 MotionBERT 특징 기반 예측보다 오차가 줄어드는지 확인했다. **사람별 MAE는 8.683496점으로, 같은 조건의 고정 MotionBERT+Ridge 8.526630점을 개선하지 못했다.**

## 입력부터 평가까지

전처리된 관절 좌표 `[기록 수,124,17,3]` → 51개 시계열로 재배치 → 1D 합성곱 6층 → 작은 출력층 → 좌우 GDI 두 값으로 처리한다. 마지막 축은 x·y·관측 신뢰도다. 창별 좌표 정규화·클리핑을 유지했고 새 결측 보간·평활화는 추가하지 않았다. 관절 순서와 입력 범위는 [input_contract.json](input_contract.json)에 있다.

같은 695명을 5묶음으로 나눴다. 바깥 평가 139명을 제외한 556명 안에서 내부 학습 417명·검증 139명을 네 번 바꿔 학습했다. 내부 검증 오차로 epoch를 선택한 뒤 556명으로 새로 학습하고 남겨 둔 139명을 평가했다. 내부 학습 20개와 재학습 5개, 총 25개 학습·52,125회 갱신을 완료했다. 선택 epoch는 97·66·93·89·71이었다.

57,824개 파라미터, 채널 32·커널 8의 합성곱 6층, Pool 2·2·3, Dropout 0.5, Dense 10→2 구조다. RMSprop 시작 학습률 0.001, 배치 32, 최대 100epoch를 사용했다. 정답의 최솟값·범위는 해당 학습 사람에서만 구했다. 전체 설정은 [protocol.json](protocol.json)에 있다.

[Stanford 공개 CNN](https://github.com/stanfordnmbl/mobile-gaitlab/tree/4167ee6213ac81fb9168f4e20c6538d42a41a5a6/training)의 구조를 참고해 PyTorch로 구현했다. 원 구현의 한쪽 12특징·여러 구간 평균 대신 17관절·한 구간·좌우 동시 출력을 사용했으므로 원 논문 성능 재현으로 해석하지 않는다. [출처와 지문](sources/provenance.json), [원 코드 라이선스](sources/LICENSE)를 보존한다.

## 결과와 한계

| 방법 | 같은 695명의 MAE↓ |
|---|---:|
| 고정 MotionBERT+Ridge | 8.526630 |
| MotionBERT 마지막 단계 부분학습 | 8.354702 |
| 초기 CNN | 8.683496 |

CNN은 5개 바깥 분할 모두에서 Ridge보다 오차가 높았다. Ridge 대비 증가량의 사람 재표집 95% 범위는 -0.148~0.476점으로 0을 포함했다. 높은 점수의 과소예측과 학습·평가 오차 차이가 남았으며 원인을 학습시간 하나로 확정하지 않았다. 결과는 [전체·집단별 수치](results/metrics.csv), [분할별 수치](results/fold_metrics.csv), [학습 곡선](results/learning_curves.csv), [점수대별 수치](results/score_bands.csv), [상세 집계](results/summary.json)에 있다.

기존 검산에서는 25개 학습의 분할·예측·곡선 지문, 내부 epoch 선택, 집계 재계산을 확인했다. [당시 검산](results/verification.json)의 `checkpoints_downloaded=false`는 당시 상태다. 이번 공개를 위한 가중치 회수와 예측 확인은 상위 폴더의 공개 검증 기록에 따로 남긴다. 독립 최종 평가·임상 검증 결과는 아니다.

## 가중치로 예측하기

[cnn-initial695-weights.tar.gz](https://github.com/youngni05/gait-scoring/releases/download/research-initial-models-20261101/cnn-initial695-weights.tar.gz)를 받아 압축을 푼다. 다음 명령은 저장소 최상위에서 실행한다. 입력은 원 실험과 같은 전처리를 마친 float32 NumPy `.npy` 배열이다.

```bash
python -m pip install -r research/2026-10-30-cnn-baseline695/requirements.txt
python research/2026-10-30-cnn-baseline695/predict.py --weights /path/to/cnn/fold0.pt --input /path/to/prepared_windows.npy --output /path/to/gdi_predictions.npy
```

출력은 `[기록 수,2]`, 순서는 왼쪽·오른쪽이다. 0~100으로 자르지 않는다. 연구의 평가값을 재현할 때는 각 사람을 학습하지 않은 해당 분할 모델을 사용해야 한다. 이 명령은 원본 영상의 관절 추출·전처리까지 수행하지 않는다.
