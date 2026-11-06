# 같은 695명 입력에서 비교하는 기준 1D-CNN

기존 MotionBERT+Ridge/PLS와 같은 보행 입력을 작은 CNN에 직접 넣어, 시간 흐름을 학습하는 방식의 기준 성능을 확인하려 한다. 이번 묶음은 **CNN 단독 기준 실험**이다. MotionBERT 결합, 구조 탐색, 최종 예약자료 평가는 포함하지 않는다.

팀원 코드는 없다는 사용자 확인에 따라 새로 작성했다. Stanford 공개 CNN의 구조를 참고해 기존 서버의 PyTorch에서 실행하도록 구현했다. **원 논문의 MAE 6.5점 재현이나 팀원 모델 재현이라고 부르지 않는다.** 사람·입력·학습 방식·집계 조건이 다르다.

## 확인한 코드와 입력의 차이

공개 소스: [mobile-gaitlab training](https://github.com/stanfordnmbl/mobile-gaitlab/tree/4167ee6213ac81fb9168f4e20c6538d42a41a5a6/training). 실제 원본과 Apache 2.0 라이선스, 다운로드 URL·SHA-256은 `sources/`에 보관했다. 공개 노트북의 기본 target은 무릎 각도지만, 저장소 설명에서 같은 doublesided 코드를 GDI에도 사용하는 것으로 안내한다.

| 항목 | 공개 doublesided CNN | 이번 기준 CNN |
|---|---|---|
| 입력 | 한쪽 발목·무릎·엉덩이·발끝 x/y, 각도·거리 등 12개 시계열 | 기존 MotionBERT 입력과 같은 17관절 x/y/관측 신뢰도, 총 51개 시계열 |
| 관찰 길이 | 124프레임 여러 구간의 예측을 영상별로 합침 | 기록당 기존 초반 124프레임 한 구간 |
| 좌우 처리 | 좌우를 별도 예제로 쌓아 같은 네트워크에 넣음 | 한 구간에서 좌우 GDI 두 값을 동시에 출력 |
| 출력 2칸 의미 | 첫 칸이 예측, 두 번째 칸은 가중 손실 구현용 보조 칸 | 두 칸 모두 실제 정답: 왼쪽 GDI, 오른쪽 GDI |
| 전처리 | 결측 보간·평활화, 골반 중심/허벅지 길이 정규화 등 | 기존 저장 입력 유지. 새 보간·평활화·방향 반전 없음 |
| 구현·초기화 | Keras/TensorFlow | PyTorch, 새 무작위 가중치. 공개 학습 가중치는 사용하지 않음 |
| 평가 | 원 연구의 분리 및 집계 | 기존 695명 사람 단위 5분할, 사람별 MAE |

입력 파일 `data.npz`는 검산된 `expansion695/train_only.npz`에서 x·y·사람/영상 ID·fold·cohort만 복사했다. 512개 MotionBERT 특징은 CNN 입력에 포함하지 않았다. 수치·기록순서·좌우 정답·분할의 동일성을 검사했다.

- 695명·1,064기록 = 기존 175명·203기록 + 추가 520명·861기록.
- 입력 `1064×124×17×3`을 CNN에서 `1064×51×124`로 바꾼다. **51은 특징 채널, 124가 시간이다.** 세 번째 좌표 채널은 깊이가 아니라 관측 신뢰도다.
- 관절 순서: 골반, 오른쪽 엉덩이/무릎/발목, 왼쪽 엉덩이/무릎/발목, 척추, 가슴, 코, 양눈 중간점, 왼쪽 어깨/팔꿈치/손목, 오른쪽 어깨/팔꿈치/손목. 원래 H36M 이름의 일부를 얼굴 관측점으로 근사한 기존 변환을 유지한다.
- x/y는 관측된 점의 창별 경계 상자로 정규화·클리핑된 값이다. 데이터 전체를 보고 새 정규화를 맞추지 않는다.
- 다리 6개 관절에서 관측 신뢰도 0인 점은 없다. 전체 관절에는 신뢰도 0인 점 72,097개가 있다. 저장된 좌표와 신뢰도를 그대로 사용하며 임의로 채우지 않는다.
- ‘앞 구간’은 카메라 정면이 아니라, 기존 선별 규칙상 연속 관측 구간의 초반 124프레임이다. 반드시 원 영상 첫 프레임부터인 것은 아니다. 촬영 방향을 새로 통일하거나 좌우를 바꾸지 않는다.
- 기존 37/38명과 최종 예약 223명은 묶음에 없다. 보호된 숫자 정답을 이번 준비에서 열지 않았다.

세부 수치와 출처 코드 지문은 `input_contract.json`에 있다. 원 데이터의 이용 조건과 코드 라이선스는 다르며, 원 제공자는 처리된 좌표에 CC BY-NC 2.0을 안내한다. 이 묶음은 현재 연구용 서버 실행을 위한 것이다.

## 고정한 기준 설정

구조는 `(Conv1D → ReLU → BatchNorm)×2 → Pool → Dropout`을 세 번 거친다. 각 합성곱은 채널 32, 커널 길이 8이며 Pool 길이는 2·2·3이다. 124프레임이 62→31→10칸으로 줄어든 뒤 Dense 10→좌우 GDI 2개로 연결한다. 학습 파라미터는 **57,824개**다.

| 설정 | 이번 값·선택 이유 |
|---|---|
| 학습률·최적화 | 공개 노트북을 참고한 RMSprop, 시작 0.001, alpha 0.9, epsilon 1e-8 |
| 학습률 변화 | 공개 함수와 같이 10번째 epoch부터 10회마다 ×0.8 |
| 배치·최대 길이 | 32기록, 최대 100 epoch. epoch는 해당 학습자료를 한 바퀴 보는 단위 |
| 과적합 억제 | Dropout 0.5, 세 번째~여섯 번째 합성곱 kernel에 L2=10^-3.5 |
| 정답 변환 | 해당 학습 사람의 좌우별 최솟값·범위로 변환. 예측은 원래 GDI 단위로 되돌림 |
| 손실 | 기록별 동일 비중, 변환된 좌우 GDI의 평균제곱오차 + L2. 원 GDI 단위 MAE와 구분 |
| 초기값 | seed 20261028에서 fold별 seed 파생. Xavier 초기화. 학습 가중치 가져오지 않음 |
| 후보 수 | 구조·학습률 후보는 1개. 이번에는 epoch만 내부 검증으로 선택 |
| 자원 요청 | GPU 1개, CPU 4개, RAM 12GB, Slurm 4시간. 학습 명령은 약 3시간에서 중단·재개 가능 |

원 노트북의 구조·여러 설정을 참고했지만, PyTorch BatchNorm/RMSprop과 Keras의 계산이 비트 단위로 같다는 검증은 하지 않았다. 새 입력·두 유효 출력·좌우별 정답 변환도 명시적인 차이다. 이 기준 결과를 얻은 뒤 별도 1단계에서 학습률·규제를 비교한다.

## 평가 사람을 보지 않고 epoch를 선택하는 흐름

1. 기존 5묶음 중 1묶음 139명을 바깥 평가로 남긴다. 나머지 556명만 모델 선택과 학습에 사용한다.
2. 그 556명의 나머지 4묶음을 돌려 내부 학습 417명/검증 139명으로 각각 최대 100 epoch 학습한다.
3. epoch별 내부 검증 사람 MAE를 네 묶음에서 평균하고, 가장 낮은 epoch를 고른다. 동률이면 더 이른 epoch를 선택한다.
4. 새 초기값으로 556명 전체를 선택된 epoch만큼 다시 학습한 뒤, 남겨둔 바깥 평가 139명을 한 번 예측한다.
5. 바깥 5분할을 모두 수행한다. **내부 20개 + 재학습 5개 = 25개 학습**이다. 모든 사람은 자신을 학습하지 않은 모델로 평가된다.

정답 변환은 내부 학습 또는 최종 재학습 사람에서만 계산한다. 원 영상 좌표 전처리는 창별 계산을 그대로 사용한다. 선택 코드에 바깥 평가 좌표·정답을 넘기지 않는다. 현재 개발자료를 반복 비교하는 실험이며 독립 최종 성능으로 표현하지 않는다.

## 서버로 보내기 — Windows PowerShell

아래는 Windows에서 실행한다. 서버에 이 폴더가 이미 준비되어 있으면 다시 풀지 않고 기존 명령으로 재개한다.

```powershell
scp -P 10000 "C:\Users\jun\Desktop\D_P\gdi-research-harness\studies\20261030-gpu-motionbert-study\cnn-followup695\gdi-cnn695.zip" RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/
ssh -p 10000 RESEARCH_USER@PRIVATE_HOST
```

서버 ABRM02에서 한 번만 압축을 푼다.

```bash
cd /abr/RESEARCH_USER
python3 -B -m zipfile -e gdi-cnn695.zip .
cd /abr/RESEARCH_USER/gdi-cnn695-20261030
mkdir -p logs
squeue -u RESEARCH_USER
tmux ls
```

기존 GPU 할당이 남아 있다면 그 계산 노드의 tmux 화면에서 아래 파일럿을 실행한다. 기존 할당이 없으면 뒤의 `sbatch` 방법으로 제출한다. 로그인 노드 ABRM02에서는 학습하지 않는다. `launch.sh`도 Slurm 작업 환경이 없으면 거부한다.

## 먼저 실행할 GPU 파일럿

기존 GPU 할당의 계산 노드 쉘에서:

```bash
cd /abr/RESEARCH_USER/gdi-cnn695-20261030
bash launch.sh pilot
```

기존 할당이 없다면 **ABRM02에서**:

```bash
cd /abr/RESEARCH_USER/gdi-cnn695-20261030
mkdir -p logs
sbatch submit.slurm pilot
squeue -u RESEARCH_USER
```

파일럿은 입력 지문 검사 → 가상 GPU 학습·재개 검사 → 실제 개발자료의 내부 학습 417명/검증 139명으로 3 epoch를 수행한다. 바깥 평가 139명은 사용하지 않는다. 새 학습이므로 성능 결과로 발표하지 않는다. 완료 표시는 `CNN_PILOT_FINISHED`다.

```bash
cat checks/gpu_smoke/smoke.json
cat runs/pilot/pilot_summary.json
tail -n 25 logs/pilot.log
```

실제 GPU 시간·최대 메모리는 이 파일럿에서 확인한다. `rough_max_training_seconds`는 전체 학습시간의 거친 추정값이며 보장 시간이 아니다. 3시간 실행 상한과 4시간 Slurm 할당 안에 들어오는지 확인한다. 이전 DIS06에서 CUDA 할당 정지가 있었으므로 새 Slurm 제출은 DIS06을 제외한다. 그 노드가 현재도 고장이라고 확정한 것은 아니다.

## 본 학습과 재개

파일럿이 통과하고 자원을 확인한 후 기존 계산 노드에서:

```bash
cd /abr/RESEARCH_USER/gdi-cnn695-20261030
bash launch.sh train
```

새 배치 작업으로 실행하려면 **ABRM02에서**:

```bash
cd /abr/RESEARCH_USER/gdi-cnn695-20261030
mkdir -p logs
sbatch submit.slurm train
```

배치 작업은 SSH 연결이 끊겨도 Slurm에서 실행된다. 기존 할당과 새 할당을 동시에 신청할 필요는 없다. 같은 `runs/main`에 두 작업을 동시에 실행하지 않는다.

- 진행 확인: `tail -n 20 logs/train.log`, `squeue -u RESEARCH_USER`.
- epoch마다 학습·내부 검증 오차와 가중치·최적화 상태·난수 상태를 저장한다. 중단 직전 저장되지 않은 epoch는 다시 수행한다.
- 중단 또는 시간 상한이면 새 GPU 할당에서 **같은 `bash launch.sh train`**을 실행한다. 완료한 내부 학습은 재학습하지 않고 체크포인트를 읽는다.
- 기존 출력에 다른 코드·설정·자료·사람 분할을 섞으면 거부한다. 설정 변경 실험은 별도 묶음·출력 위치로 만든다.
- 완료: `CNN_TRAINING_AND_EXPORT_FINISHED`, `runs/main/summary.json`, `runs/main-review.tar.gz` 생성.
- 파일 지문 불일치, 비정상 손실·기울기, 잘못된 입력·분할이면 종료한다. CUDA 검사에는 제한 시간이 있어 이전처럼 오래 무출력 대기하지 않는다. 오류 출력을 보내 원인을 확인하고 데이터 검사 조건을 완화하지 않는다.

## 결과와 회수

`runs/main/summary.json`은 사람별 MAE, 기존 175명/추가 520명, fold별 오차, 10점 구간 일치와 큰 구간 오류, 실제 optimizer step 수를 담는다. 같은 사람·분할·정답으로 대조한 기존 기준은 Ridge **8.526630**, PLS **7.974896**, MotionBERT 마지막 단계 부분학습 **8.354702**점이다. 이 참조 예측은 CNN 학습에 들어가지 않는다.

`oof_predictions.npz`는 `[1064기록, 1구간, 2좌우]`와 사람/영상 ID·fold·cohort·정답을 저장한다. `predictions.csv`는 사람/영상/좌우별 실제값·예측값을 읽기 쉽게 기록한다. 내부 학습 곡선, 선택 epoch, 학습/검증/평가 사람 ID, 체크포인트 지문도 남긴다. 10점 구간은 연구용 고정 구간이며 임상 등급으로 해석하지 않는다.

Windows PowerShell에서 결과를 받는다.

```powershell
scp -P 10000 RESEARCH_USER@PRIVATE_HOST:/abr/RESEARCH_USER/gdi-cnn695-20261030/runs/main-review.tar.gz "C:\Users\jun\Desktop\D_P\gdi-research-harness\studies\20261030-gpu-motionbert-study\cnn-followup695\main-review.tar.gz"
```

가벼운 결과 압축에는 코드·설정·로그 역할의 곡선·분할 정보·예측·체크포인트 지문이 포함되지만 `.pt` 가중치는 제외된다. 모델 재추론까지 확인하려면 서버의 `runs/main/fold*/refit/latest.pt`도 별도로 회수한다. 파일럿이나 실패를 진단할 때는 `logs/`의 해당 로그를 함께 확인한다.

## 준비 검사와 아직 남은 확인

로컬에서는 실제 입력 동일성·사람 분리와 Python 3.8 문법을 검사했다. 별도 생성한 가상 좌표·가상 정답으로 손실/기울기·실제 가중치 갱신·중단 재개·저장 후 재예측·전체 25개 학습 흐름과 결과 집계를 검사했다. 비정상 수치·사람 중복·잘못된 축·다른 실행의 체크포인트·불완전/변조 결과도 거부하는지 확인한다.

이 검사는 **CPU의 공학적 동작 검사**다. 서버 Python 3.8.10/PyTorch 2.4.0+cu121 환경의 실제 GPU 실행과 CNN 성능은 실행 후 확인해야 한다. 서버 패키지를 자동 설치하거나 업그레이드하지 않는다. 실행 의존성은 기존 NumPy와 PyTorch이며 TensorFlow·scikit-learn은 필요하지 않다.
