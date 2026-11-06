# 정상100 추가 논문 실험 모델

LJMU 성인64명·881기록에 실제 측정값이 아닌100점 가정 라벨을 부여해 기존 임상자료와 함께 학습한 모델이다. 현재 선별 기준 모델과 분리한 논문 분석용 구성이다.

[GitHub Release](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)의 `gdi-normal100-paper75-20261106.tar.gz`에 CNN·STT·좌우 STT의75개 저장 모델을 담았다. 내부 모델·refit·기존 제외 여부를 구분하며, 공개파일마다 원본과의 텐서 동일성을 확인했다.75개를 모두 같은 비중으로 평균하는 방식이 아니다.

코드와 입력은 [기준 모델 추론 안내](../../selected/frozen-gdi/README.md)를 따른다. 이 폴더의 `models.json`과 가중치를 사용한다. `original`은 혼합 학습자료로 구한 기존 보정, `stanford`는 같은 모델의 임상 학습자료만 사용한 후속 보정이다. 보정계수와 결합 비중은 이미 고정했으며 사용자 입력으로 다시 학습하지 않는다.

```bash
python predict.py --input prepared.npz --out normal100-result.npz --mode fusion --calibration stanford --stt-source /path/to/transformer-gait-analysis
```

이 모델로 외부 정상 보행의 출력은 상승했으나, 임상 개발자료 전체 MAE는 기준보다 커졌다. 실제 GDI 없는 영상에서는 정확도를 계산할 수 없다. 고령자 건강 판정이나 임상적 GDI 측정기로 사용하도록 검증한 모델이 아니다.

기존 CPU/GPU 환경 차이 예외(CNN fold3 bag의 최대 차이0.0012155359점이 당시0.001점 기준을 초과)는 [완료 검산 기록](../../archive/studies/20261104-ljmu100-mixed/completion-verification.json)에 보존한다. 동일 GPU 환경에서는 기존15개 분할의 예측 재현이 일치했다. 이번 공개용 가상 입력 재현과 이 과거 교차환경 검산은 구분한다.
