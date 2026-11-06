# CNN 선형 보정에 필요한 학습 예측 회수

현재 회수본에는 각 평가 사람의 예측만 있다. 같은 모델의 학습 사람 예측이 없어 CNN 선형 보정은 아직 실행하지 못했다. 다른 외부 분할의 평가 예측을 모아 보정하면 해당 모델들이 현재 평가 사람을 학습에 포함했을 수 있으므로 그 방법으로 대체하지 않는다.

`export_cnn_train_predictions.py`는 **완료된 CNN의 저장 가중치로 추론만** 한다. 학습·epoch 재선택·GPU 할당은 하지 않는다. CPU로도 실행할 수 있다. 원본 코드·데이터·체크포인트는 읽기만 하고 새 출력 폴더에 저장한다.

원래 CNN 실행 환경에서 아래처럼 실행한다. 경로는 실제 완료 실행의 위치로 지정해야 한다. `--bundle`에는 원본 `run.py`, `model.py`, `manifest.json`, `data.npz`, `protocol.json`이 있어야 하고, `--results`에는 완료된 `fold0`~`fold4`가 있어야 한다.

```bash
python export_cnn_train_predictions.py \
  --bundle /path/to/original/cnn/bundle \
  --results /path/to/completed/results \
  --out /path/to/new/cnn-calibration-export \
  --device cpu
```

각 `fold*/refit/latest.pt`의 지문·학습 사람·epoch·정답 변환을 확인한다. 전체 개발 입력을 예측한 뒤, 이미 회수한 평가 예측과 최대 절대 차이 0.0001점 이내인지 검사한다. 환경 차이로 검사가 실패하면 허용치를 임의로 넓히지 않고 차이를 확인한다. 출력 폴더가 이미 있으면 덮어쓰지 않는다.

생성되는 `fold0.npz`~`fold4.npz`, `complete.json`을 회수하면 각 학습 556명의 예측으로 보정식을 만들고 해당 평가 139명에 적용할 수 있다. 기존 모델의 평가 예측과 연결하는 단계이며 새 CNN 학습은 필요하지 않다.

이 도구는 로컬에서 Python 3.8 문법·도움말·체크포인트 부재 시 출력 생성 전 거부를 확인했다. 실제 체크포인트 추론은 아직 실행하지 못했다. 내부 교차검증 예측을 이용하는 보정에는 선택 epoch의 내부 모델 예측이 추가로 필요하다. 저장된 내부 `latest.pt`는 최대 epoch 상태여서 선택 epoch와 다를 수 있으므로 자동 대용하지 않는다.
