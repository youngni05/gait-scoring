# 고정 GDI 모델과 추론

이미 평가에 사용한54개 구성 모델의 추론용 파라미터를 보존했다. 이 안에서 CNN5개 기준·CNN24개 평균·전체54개 결합을 선택한다. 전체 개발자료로 새 단일 모델을 학습한 것은 아니다.

가중치는 [GitHub Release](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)의 `gdi-frozen54-20261106.tar.gz`에 있다. 압축 안에 코드·설정·출처 고지와 `weights/`를 함께 넣었다. `models.json`은 원 체크포인트와 공개 가중치의 SHA-256, 고정 보정값·결합 비중을 연결한다.

입력은 이미 원 실험 방식으로 전처리한 NumPy 파일이다. `x`의 모양은 `[구간 수,2,124,12]`이며 좌우 다리·124프레임·12개 특징 순서다. 선택 항목 `window_record`는 각 구간이 속한 기록의0부터 연속된 정수 인덱스다. 없으면 각 구간을 별도 기록으로 취급한다. 영상·MediaPipe 원좌표를 그대로 넣는 입력이 아니다.

```bash
python -m pip install -r requirements.txt
python predict.py --input prepared.npz --out cnn-result.npz --mode cnn_bag
```

CNN만 사용하는 두 방식에는 STT 코드가 필요 없다. 결합을 실행하려면 STT 원본 저장소의 커밋 `549d58bc32449b24a0eae0c37ca573f9aa58a1dd`를 별도로 준비하고 그 위치를 지정한다. [원 제공자 저장소](https://github.com/vinuni-vishc/transformer-gait-analysis)에 라이선스 선언이 확인되지 않아 해당 구현 사본은 이 배포에 포함하지 않았다. 원 제공자의 이용 조건을 확인한다.

```bash
python predict.py --input prepared.npz --out fusion-result.npz --mode fusion --stt-source /path/to/transformer-gait-analysis
```

처리 순서는 구간 예측→기록 안 평균→각 모델의 고정 보정→모델 평균·결합이다. 출력 `prediction`은 기록별 좌우 예측이다.0~100으로 자르거나 입력 자료로 다시 보정하지 않는다. 잘못된 모양·빈 입력·결측 수치·잘못된 기록 인덱스는 거부한다. 기존 출력 파일은 덮어쓰지 않는다.

`--weights`로 압축을 푼 폴더를 따로 지정할 수 있다. `cnn_refit`, `cnn_bag`, `fusion`을 지원한다. 이 기준 모델에서는 보정 선택을 기본값 `original`로 유지한다.

이번 공개 검사는 원본과 공개 텐서의 동일성, SHA-256, 재로딩, 가상3구간·2기록의 추론 경로를 확인한다. 실제 사람의 정확도를 새로 평가한 시험이 아니다. 기존 성능은 [선별 목록](../README.md), 이번 검사는 `publication-verification.json`에 있다. 가중치에는 optimizer·난수 상태·학습 사람 명단·개인별 정답을 포함하지 않았다.
