# 실제 SPPB 총점 예측 — 44명 보행 마커 실험

**대학 과목 제출용 비상업적 연구 결과로 배포한다.** 모델 압축에 출처·CC BY 4.0 원문·수정 내역을 동봉했다. [출처](../../licenses/bundles/sppb/SOURCES.md), [수정 내역](../../licenses/bundles/sppb/MODIFICATIONS.md), [적용 범위](../../licenses/README.md)를 함께 확인한다.

**현재 상태: 후속 실험 중단. 프로젝트가 GDI 예측을 중심으로 전환해 이 연구는 과거 기록으로 보존한다(2026-11-02).** 아래 실험은 2026-10-30 실행·검산을 완료했다.

보행의 움직임만으로 실제 SPPB 총점을 예측할 수 있는지 확인했다. 입력은 44명의 Vicon 보행 마커이며, 정답은 같은 사람의 평가표에 제공된 실제 SPPB 총점이다. 점수 정의가 확인되지 않았던 별도 MAT 자료의 등급을 표준 SPPB로 합산한 실험과 구분한다.

## 입력과 비교 방법

12개 3D 마커의 골반 기준 좌표 36개와 골반 속도 3개, 총 39채널을 사용했다. 마커 위치는 선택한 구간 안에서 골반 중심·다리 길이·이동 방향을 기준으로 정리했다. 원래 100Hz 자료 중 연속 3초를 30Hz·91시점으로 맞췄으며, 결측을 채우거나 짧은 기록을 늘리지 않았다. 나이·성별·다른 임상검사는 입력에 넣지 않았다.

44명에게 보행 기록이 5개씩 있어 총 220개 입력이지만, 서로 다른 임상 정답은 44명분이다. 사람 단위 5분할로 평가하고 학습 집단 안의 사람 3분할로 설정과 학습 종료 시점을 선택했다. 같은 사람의 5회 예측을 평균해 실제 점수와 비교한 것이 주결과다. **주결과는 3초 구간 5개, 총 15초 보행을 합친 값이며 단일 3초 결과와 다르다.**

Ridge·RF는 각 채널의 평균·표준편차·평균 절대 시간 변화량을, CNN·TCN은 시간순 시계열을 입력받는다. 정규화와 정답 변환은 해당 학습 사람에서만 계산했다. 상세 조건은 [protocol.json](protocol.json)에 있다.

| 방법 | 사람별 MAE(점)↓ |
|---|---:|
| 평균 기준선 | 1.380 |
| 중앙값 기준선 | 1.318 |
| Ridge | 1.142 |
| RF | 1.001 |
| CNN | 1.236 |
| TCN | 1.194 |

RF의 중앙값 기준선 대비 차이는 -0.318점이며, 저장 예측의 사람 재표집 95% 범위는 -0.634~-0.021점이었다. 작은 44명 개발자료의 탐색 결과로, 재학습·모델 선택의 모든 변동을 포함하는 독립 확증은 아니다. 실제 정답은 5~12점에 한정되며 0~4점의 성능은 확인하지 못했다. 5점·6점은 각각 1명뿐이어서 전체 평균으로 낮은 점수대의 성능을 보장하지 않는다.

수치는 [summary.json](results/summary.json), 점수대별 오차는 [표](results/errors-by-actual-score.csv)에 있다. 보행 1·3·5회 평균의 사후 비교에서 RF MAE는 1.075·1.024·1.001점이었다. [조건](results/aggregation-protocol.json)과 [결과](results/aggregation-sensitivity.csv)를 함께 보존하며 이 사후 비교로 새 모델을 선택하지 않았다.

## 모델로 실행하기

[SPPB 모델 묶음](https://github.com/youngni05/gait-scoring/releases/download/research-functional-archive-20261102/sppb-archived-models-20261102.tar.gz)을 받아 푼다. Python 3.12 환경에서 아래 명령을 저장소 최상위에서 실행한다. 입력은 원 실험과 같은 전처리를 마친 float32 `[기록 수,39,91]` NumPy 배열이다. [설정](protocol.json)의 마커 순서대로 각 마커의 옆·진행·수직 방향 좌표를 연결하고, 마지막에 같은 세 방향의 골반 속도를 넣는다. 원래 카메라 축의 x/y/z를 그대로 넣는 형식과 다르다.

```bash
python -m pip install -r research/archived-sppb-tug/2026-10-30-sppb-older44/requirements.txt
python research/archived-sppb-tug/2026-10-30-sppb-older44/predict.py --weights /path/to/sppb/RF-fold0.joblib --input /path/to/prepared_walks.npy --output /path/to/sppb_predictions.npy
```

CNN·TCN은 `CNN-fold0.pt`·`TCN-fold0.pt`, Ridge는 `Ridge-fold0.joblib`을 같은 명령에 넣는다. 출력은 기록별 실수 점수이며 반올림하거나 0~12로 자르지 않는다. 주평가를 재현할 때는 사람을 학습하지 않은 해당 분할 모델로 예측하고 같은 사람의 5개 출력을 평균해야 한다.

[당시 검산](results/verification.json)은 원본 입력 재구성·사람 분리·저장 모델 재추론·지표 재계산을 확인했다. 이번 공개용 모델도 220기록을 네 방법 모두에서 다시 예측했다. [이번 배포 검산](../publication-verification.json)을 별도로 확인한다. 새 학습은 없었다.

자료는 Caicedo Rodríguez 등의 [공식 OlderAdults44 자료](https://data.mendeley.com/datasets/xgw6bg3g8h/1), CC BY 4.0이며 [원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7701341/)에서 측정 방법을 확인할 수 있다. 보정된 3D Vicon 마커를 이용한 결과이므로 영상 관절 추출·스마트폰 실행·임상검사 대체는 검증하지 않았다.
