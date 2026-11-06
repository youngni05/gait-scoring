# GPU 결과를 CPU 분석으로 연결할 입력 계약

2026-10-30. GPU 연구가 끝났을 때 점수 예측과 특징을 같은 자료에 연결해 비교하려 한다. 이 문서는 받을 파일의 조건을 정리한 것이며, GPU 결과가 이미 생성됐다는 의미는 아니다. 현재 CPU에서 학습한 모델과 캐시 특징으로만 연결 검사를 했다.

## 1. 점수 예측을 받아 같은 표와 그림을 만드는 경우

GPU 담당이 각 사람을 학습에서 제외한 모델의 예측을 모아 NPZ 하나와 메타데이터 JSON 하나를 제공한다.

| NPZ 키 | 모양 | 조건 |
|---|---|---|
| predictions | (203, 2, 2) | 축 순서: 원본 기록, `[front, back]`, `[left_GDI, right_GDI]`. 유한한 실수 |
| person_id | (203,) | 원본 train_only.npz와 같은 값·행 순서 |
| video_id | (203,) | 원본과 같은 값·행 순서. 중복 없는 203기록 |
| fold | (203,) | 원본의 0~4 묶음 배정과 같은 값·행 순서 |
| baseline | (203, 2, 2), 선택 | 제공한다면 원본 baseline_predictions.npz와 정확히 같은 배열. 새 GPU 계산으로 대체하지 않음 |

JSON은 `cpu_predictions.metadata.json`과 같은 구조를 사용한다. 이 CPU 예시는 실제 완료 모델을 검증한 뒤 작성했다. GPU 결과를 평가할 때에는 CPU 예시의 모델 지문을 복사하지 않고 실제 GPU 산출물로 채운다.

- `schema`: `gdi_person_oof_predictions_v1`
- `prediction_sha256`: 제공한 NPZ의 실제 SHA-256
- `dataset_sha256`: 원본 train_only.npz의 SHA-256
- `baseline_sha256`: 원본 baseline_predictions.npz의 SHA-256
- `shape`: `[203, 2, 2]`
- `axes`: `["record", "window", "side"]`
- `windows`: `["front", "back"]`
- `sides`: `["left_GDI", "right_GDI"]`
- `prediction_origin`: `person_oof`
- `training_scope`: `development_175_only`
- `fold_models`: 0~4 순서로 각 모델의 fold, model_sha256, training_people 140명, evaluation_people 35명

평가 코드는 자료·예측 파일 지문, 기록 순서, 중복·누락, 점수의 유한성, 축 선언과 사람 분리 명단을 검사한다. **메타데이터는 생산자의 선언이며, 실제 학습에서 평가 사람을 배제했다는 사실 자체를 증명하지는 않는다.** 조율에서 모델 체크포인트·학습 로그·정답 사용 범위를 함께 대조해야 한다. 배열 값을 몰래 좌우·앞뒤로 바꾸고 선언은 그대로 둔 오류도 메타데이터만으로 완전히 탐지할 수 없으므로 생산 단계의 묶음별 재현 검사가 필요하다.

구간은 A와 같은 `floor(GDI/10)`의 `[10k,10(k+1))`를 쓴다. 0~100으로 자르거나 경계를 다시 정하지 않는다. 전체 오차·구간 일치율은 사람 안에서 좌우·기록을 평균한 뒤 175명 평균이다. 점수대별 값은 해당 구간에 들어온 값만 사람 안에서 평균하고, 그 구간에 값이 있는 사람을 평균한다.

```powershell
# 아래 GPU_PREDICTIONS/GPU_METADATA는 앞으로 받을 실제 파일로 바꾼다.
& 'C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -X utf8 'C:/Users/jun/Desktop/D_P/gdi-research-harness/studies/20261030-cpu-head-study/followup/evaluate.py' --predictions GPU_PREDICTIONS --metadata GPU_METADATA --name gpu_variant --output 'C:/Users/jun/Desktop/D_P/gdi-research-harness/studies/20261030-cpu-head-study/followup/gpu_variant_evaluation'
```

생성 결과는 metrics.csv, score_bands.csv, summary.json, 점수대별 오차·편향 그림과 구간 오류 그림이다. 출력은 이 CPU 후속 폴더 내부로 제한한다. GPU 폴더의 모델이나 결과를 수정하지 않는다.

## 2. GPU에서 조정한 특징으로 Ridge를 다시 학습하는 경우

GDI 예측 파일만으로 Ridge를 다시 학습할 수는 없다. **각 묶음에서 학습한 MotionBERT가 만든 학습 사람의 특징과 평가 사람의 특징**이 모두 필요하다.

```text
fold k의 평가35명을 제외하고 MotionBERT 부분학습
                        ↓ 같은 체크포인트 고정
    학습140명의 앞 특징             평가35명의 앞/뒤 특징
              ↓
    학습 특징으로만 표준화 계산
              ↓
    학습 특징 + 학습 GDI로 Ridge 학습
              ↓
    고정된 표준화와 Ridge로 평가35명 예측
```

사람 전체를 보고 학습한 하나의 MotionBERT에서 특징을 뽑은 뒤 Ridge만 사람 분리하는 방식은 이 비교와 같지 않다. 평가 사람의 정보가 특징 학습에 들어갔다면 Ridge 단계에서 다시 나누어도 되돌릴 수 없다.

각 fold별로 다음 파일을 받는 구성을 생각하고 있다.

| 키 | 모양·내용 |
|---|---|
| train_indices | 원본에서 해당 fold 외 학습 기록의 행 번호, 원본 순서 |
| evaluation_indices | 원본에서 해당 fold의 평가 기록 행 번호, 원본 순서 |
| train_features_front | `(학습기록수, 512)` |
| evaluation_features | `(평가기록수, 2, 512)`, 두 번째 축 `[front, back]` |
| train_person_id / train_video_id | train_indices에 대응하는 원본 식별자 |
| evaluation_person_id / evaluation_video_id | evaluation_indices에 대응하는 원본 식별자 |

이 자료의 fold별 학습/평가 기록 수는 0번 163/40, 1번164/39, 2번162/41, 3번159/44, 4번164/39다. 각 fold의 학습/평가 사람 수는140/35다. 정답 y를 GPU 특징 파일에 중복 저장하지 않고, CPU에서 검증된 원본의 해당 행에 연결한다.

특징의 정의와 생성 근거도 함께 받아야 한다.

- 학습·평가 특징을 **같은 fold의 같은 체크포인트**로 추출했는지.
- 기존과 같은 관절 입력·앞뒤124프레임·관절 순서·좌표 정규화인지.
- MotionBERT 표현을 어떤 단계에서 꺼내고 어떤 축을 평균해512개로 만들었는지.
- 원본 자료 지문, 특징 NPZ 지문, 체크포인트 지문, 생산 코드 지문.
- fold와 시드, 본체 학습에 사용한 사람, 업데이트 수와 손실, 평가 정답으로 설정·체크포인트를 고르지 않았는지.
- 추론 모드와 dropout, 특징 dtype, 유한값 여부.

기존 표준화 값을 새 특징에 그대로 쓰기보다 **해당 fold의 새 학습 특징으로 표준화를 다시 계산**하고 평가 쪽에는 같은 값을 적용한다. Ridge의 alpha를 기존1로 고정하는 것이 첫 비교안이 될 수 있다. alpha를 새로 선택한다면 바깥 평가35명이 아닌 학습140명 안의 사람 분리로 정해야 한다. 새 특징이 평가 사람까지 포함한 표준화·특징 선택을 거쳤다면 이 계약의 결과로 집계하지 않는다.

아직 이 특징 파일을 받거나 Ridge 재학습을 실행하지 않았다. 실제 자료·생성 조건과 비교 설정을 확인한 뒤 조율의 후속 지시에 따라 실행한다. 이 문서는 그 입력 계약이며 새 학습을 승인하거나 GPU 학습률을 확정하는 문서는 아니다.
