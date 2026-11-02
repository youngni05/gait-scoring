# 고정 CNN·MotionBERT 잔차 보정

**대학 과목 제출용 비상업적 연구 결과다.** 이번 추가 공개는 새 학습이 아니라 저장된 과거 GDI 모델의 배포다. 원 실험 날짜: 2026-10-30.

한 구간 CNN 점수를 고정하고 MotionBERT 시간 특징으로 잔차를 보완하는 방식이 단순 특징 결합보다 나은지 확인했다.

692명·1,061기록의 사람 5분할·내부 epoch 선택을 사용했다. 공개한 잔차 모델 5개는 동일 분할의 기준 CNN 5개와 함께 사용해야 한다.

기준 한 구간 CNN MAE 7.8389점, 잔차 보정 7.7192점이었다. 소폭 감소한 개발자료 결과이며 안정적인 외부 개선이나 임상 적용을 입증한 결과는 아니다.

주요 수치와 이전 검산은 [집계 결과](results/summary.json)와 [원 검산 기록의 공개본](results/verification-original.json)을 함께 확인한다. 서로 다른 사람·입력·평가 조건의 수치를 단순히 빼서 개선량으로 해석하지 않는다. 원본 실패·엄격 검사 미통과와 별도 예측 검산이 함께 있는 실험은 검산 기록의 범위를 구분한다.

## 공개 모델과 입력

- [모델 압축 다운로드](https://github.com/youngni05/gait-scoring/releases/download/research-gdi-half-20261102/gdi-cnn-frozen-residual692-20261102.tar.gz)
- 배포 범위: Five residual-model refits plus the five matching frozen baseline CNN refits required to interpret the residual scores.
- 입력 계약: Original paper-style CNN baseline score plus cached MotionBERT temporal features. Residual weights and matching baseline CNN states are both provided.
- 원 실험의 핵심 코드는 `source/`에 있다. 출력·전처리·선형 보정의 의미는 해당 코드와 결과를 따른다.

압축을 풀면 실험 이름 폴더 안의 `models/`에 가중치와 `weights-manifest.json`이 나온다. `.pt`는 CPU에서 `torch.load(path, map_location="cpu", weights_only=True)`로 읽을 수 있고 `model`에 모델 상태가 있다. `lo`·`span`이 있으면 원 코드의 정답 역변환에, `calibration`이 있으면 `[기울기, 절편]`의 원 보정에 사용한다. `.npz`는 `numpy.load(path, allow_pickle=False)`로 읽고, 이름의 접두어로 조건·분할을 구분한다. `.joblib`은 원래의 scikit-learn RF 객체다. JSON 전문가는 출력부 계수·정규화·온도만 담고 개인 명단을 제외했다.

저장 상태를 읽는 예시는 [가중치 확인 도구](../../inspect_weights.py)에 있다. 코드가 요구하는 입력은 이미 전처리한 특징·구간이며, 이 묶음만으로 원본 영상부터 예측·전체 재학습까지 자동 실행하는 제품 경로는 제공하지 않는다. 실제 GDI 평가를 재현하려면 사람을 학습하지 않은 해당 분할 모델과 원 실험의 정규화·보정·집계를 함께 사용해야 한다.

## 이번 공개에서 확인한 것

원래 저장 모델의 모든 공개 파라미터와 정규화·보정값을 대조하고, 추론용 텐서를 다시 로딩해 동일성을 확인했다. 압축 안 파일의 SHA-256도 대조했다. 새 학습이나 전 모델의 성능 재평가는 하지 않았다. 기존 성능 검산과 이번 배포 검사는 [이번 검증 기록](../../publication-verification.json)에서 구분한다.

원본 영상·개인별 좌표·정답·사람 명단·optimizer·난수 상태를 공개하지 않았다. [출처](SOURCES.md), [수정 내역](MODIFICATIONS.md), [권리 고지](NOTICE.md), [배포 목적](RESEARCH_USE.md)을 모델 압축에도 동봉했다. 데이터 CC BY-NC 2.0과 참고 코드 Apache 2.0의 적용 범위를 구분하며, 이 목적 표시는 원권리자의 허용 권리를 바꾸는 새 라이선스가 아니다.

이 결과는 해당 개발자료·당시 평가 설계의 탐색 결과다. 단일 RGB·고령자 현장·임상 활용·새 기관의 정확도로 확장하지 않는다. 당시 기록의 최종 평가자료 미사용 표시는 해당 실험 시점의 상태이며 후속 연구의 사용 상태를 보증하지 않는다.
