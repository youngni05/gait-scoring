# 실제 TUG 시간 예측 — OpenCap 298명 실험

**대학 과목 제출용 비상업적 연구 결과로 배포한다.** 모델 압축에 출처·CC BY 4.0 원문·수정 내역을 동봉했다. [출처](../../licenses/bundles/tug/SOURCES.md), [수정 내역](../../licenses/bundles/tug/MODIFICATIONS.md), [적용 범위](../../licenses/README.md)를 함께 확인한다.

**현재 상태: 후속 실험 중단. 프로젝트가 GDI 예측을 중심으로 전환해 이 연구는 과거 기록으로 보존한다(2026-11-02).** 아래 실험은 2026-10-30 실행·검산을 완료했다.

짧은 보행 구간의 움직임으로 실제 TUG 검사 시간을 예측할 수 있는지 확인했다. 입력은 OpenCap으로 복원된 보행 관절 각도·골반 속도이며 정답은 같은 사람·방문의 실제 TUG 수행 시간이다. TUG 검사 전체의 영상을 입력한 실험은 아니다.

## 입력과 비교 방법

[415명·675방문 공개본](https://zenodo.org/records/22309771)에서 3초 연속 보행과 실제 TUG가 연결된 298명을 사용했다. 한 사람당 한 방문·한 보행 기록을 선택해 반복 방문이 학습·평가 양쪽에 섞이지 않게 했다. 실제 TUG 범위는 4.22~42.59초다.

몸통·골반·다리 각도 17개와 골반 수평 속도·수직 속도, 총 19채널을 30Hz·91시점으로 만들었다. 시간순 입력 → 움직임 요약을 사용하는 Ridge·RF 또는 작은 CNN·TCN → 실제 TUG 초 예측의 흐름이다. 나이·진단·다른 임상점수는 회귀 입력에 넣지 않았다. 채널 순서·선택 조건은 [protocol.json](protocol.json)에 있다.

사람 단위 5분할로 평가하고 각 학습 집단 안의 사람 3분할로만 설정·학습 종료 시점을 선택했다. 정규화도 학습 사람으로만 계산했다.

| 방법 | 298명·3초 입력 MAE(초)↓ |
|---|---:|
| 평균 기준선 | 2.710 |
| 중앙값 기준선 | 2.472 |
| Ridge | 2.542 |
| RF | 2.150 |
| CNN | 2.524 |
| TCN | 2.513 |

RF는 중앙값 기준선보다 약 0.322초 낮았지만 RMSE는 3.811초로 큰 오차도 남았다. 저장 예측의 사람 재표집에서 RF−중앙값 MAE 차이의 95% 범위는 -0.589~-0.055초였다. 이 범위는 재학습·모델 선택·새 기관의 변동을 모두 포함하지 않는다.

이전 129명 공개본과의 중복을 확인해 두 자료의 인원을 더하거나 이전 학습 모델을 그대로 평가하지 않았다. 확대 298명에서 파일 일치·임상 일치 유력 후보 127명과 중복 미확인 171명을 따로 집계했으며, 후자에서 RF 개선은 더 작았다. 중복 미확인자를 모두 새로운 사람으로 확정하지 않는다.

이 실험에는 같은 196명의 1·3·6초 비교도 있었다. RF MAE는 3.077·2.783·2.657초였고, 3초와 6초 차이의 재표집 범위에는 0이 포함됐다. 길이 비교의 196명과 확대 분석의 298명을 섞어 길이 효과를 주장하지 않는다. 전체 표는 [summary.json](results/summary.json), 비교와 중복 집단의 불확실성은 [상세 집계](results/uncertainty-and-overlap.json)에 남겼다. **이번 모델 배포는 298명·3초 조건으로 한정한다.**

## 모델로 실행하기

[TUG 모델 묶음](https://github.com/youngni05/gait-scoring/releases/download/research-functional-archive-20261102/tug-archived-models-20261102.tar.gz)을 받아 푼다. Python 3.12 환경에서 아래 명령을 저장소 최상위에서 실행한다. 입력은 원 실험과 같은 전처리를 마친 float32 `[기록 수,19,91]` NumPy 배열이다. 채널 순서는 [설정](protocol.json)의 `channels` 배열과 같다.

```bash
python -m pip install -r research/archived-sppb-tug/2026-10-30-tug-opencap298/requirements.txt
python research/archived-sppb-tug/2026-10-30-tug-opencap298/predict.py --weights /path/to/tug/RF-fold0.joblib --input /path/to/prepared_walks.npy --output /path/to/tug_predictions.npy
```

CNN·TCN은 `CNN-fold0.pt`·`TCN-fold0.pt`, Ridge는 `Ridge-fold0.joblib`을 같은 명령에 넣는다. 출력은 기록별 예측 시간(초)이며 범위를 임의로 자르지 않는다. 연구의 평가값을 재현할 때는 해당 사람을 학습하지 않은 분할 모델을 사용한다.

[당시 검산](results/verification.json)에서는 원본 CRC·SHA, 입력 재구성, 사람 분리, 저장 모델 재추론과 정규화를 확인했다. 이번 배포도 298기록을 네 방법 모두에서 재추론했으며 [공개 검증 기록](../publication-verification.json)에 별도로 남겼다. 새 학습은 없었다.

입력은 여러 RGB 카메라와 신체 보정을 사용해 미리 복원된 OpenCap 시계열이다. 원본 RGB를 이번에 새로 처리한 실험이나 단일 카메라의 3초 촬영만으로 전 과정을 수행한 성능이 아니다. 개발자료의 탐색 결과이며 임상검사 대체·다른 기관 일반화·고령자 현장 적용은 검증하지 않았다.
