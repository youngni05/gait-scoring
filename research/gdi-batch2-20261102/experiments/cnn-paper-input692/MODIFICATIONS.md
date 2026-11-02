# 수정·학습·공개 변환 내역

Five original outer refits, recovered as the matching frozen-baseline states used by the verified residual study.

입력 계약: One per-side 12-feature window [N,2,124,12]; lo/span invert targets and calibration=[slope,intercept] returns reported GDI units.

기존에 완료·검산된 모델에서 학습용 optimizer·난수·사람 식별자와 학습 곡선을 제외하고 추론 상태·정규화·보정값을 추출했다. NPZ·RF 파일의 직접 복사 여부와 파라미터 동일성은 weights-manifest.json에 기록했다. 원 실험의 가중치를 새로 학습하거나 성능이 더 좋은 설정을 다시 고르지 않았다. 기존 사전학습 모델을 그대로 배포한 것이라고 표시하지 않는다.
