# 보행 영상에서 걸음의 차이를 읽기까지

집에서 찍은 짧은 보행 영상으로 어떻게 걷는지, 지난 방문과 무엇이 달라졌는지 알아보는 연구다. 보행편차지수(GDI)는 관절 움직임이 기준 보행에서 얼마나 벗어나는지 나타내며, 정상 참조집단의 평균은 100점이다.

| 연구 주제 | 현재까지의 답 |
|---|---|
| [주제 선택 근거](questions/01-assessment.md) | 짧은 걸음은 신체기능검사 결과를 일부 설명했다. 그래서 걷는 동안 관절 움직임이 기준 보행에서 얼마나 벗어나는지를 우선 평가하기로 했다. |
| [GDI 추정 방법](questions/02-gdi-estimation.md) | 측면 영상에서 얻은 관절 움직임을 검사실 GDI와 짝지어 학습하고, 여러 구간과 두 모델의 예측을 합쳐 좌우 GDI를 추정했다. |
| [제약 환경 테스트](questions/03-home-measurement.md) | 짧은 왕복에서도 분석 가능한 구간이 있었지만, 촬영 각도·다리 가림·직선 길이에 따라 일부 측정이 달라지거나 보류됐다. |
| [결과 검증](questions/04-validation.md) | 별도 평가 222명에서 검사실 GDI와의 평균 절대오차는 6.25점이었다. 반복 방문에서는 큰 변화의 일부를 따라갔지만 작은 변화는 구별하기 어려웠다. |
| [GDI 보완 지표](questions/05-complementary-measures.md) | 비슷한 속도에서도 GDI가 기능 차이를 일부 구별했고, GDI가 비슷한 사람도 속도·리듬·무릎 움직임은 달랐다. |

[실험 등록부 · 진행 상태와 상세 기록](experiments/README.md) · [전체 실험 기록](archive/README.md) · [논문 관련 실험](paper-related/README.md) · [모델 비교 결과](selected/README.md) · [모델 다운로드](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)
