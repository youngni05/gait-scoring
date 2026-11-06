# 현재 선별 모델

2026-11-06에 저장된 고정 모델·최종 평가 기록을 기준으로 분리한 목록이다. 모델을 새로 채택하거나 성능 순위를 다시 정한 것은 아니다. 임상 사용을 확정한 최종 모델은 없다.

| 구성 | 역할 | 별도 평가 222명 MAE | 코드·가중치 |
|---|---|---:|---|
| CNN refit 5개 평균 | 단순 비교 기준 | 6.355386 | [고정 모델 묶음](frozen-gdi/README.md)의 `cnn_refit` |
| CNN 내부·refit 24개 평균 | CNN만 사용하는 선별 후보 | 6.266755 | 같은 묶음의 `cnn_bag` |
| CNN·STT·좌우 STT 54개 결합 | 논문 기준 구성·기존 고정 결합 | 6.248559 | 같은 묶음의 `fusion` |

세 구성은 동일한 222명·377기록을 평가했다. 결합−CNN 평균의 MAE 차이는 −0.018197점이고 사람 단위 95% 구간은 −0.126983~+0.089553점이다. 결합의 추가 개선은 불확실하다. 이 평가는 같은 공개 임상자료의 미학습 사람을 대상으로 했으며, 고령자·다른 기관·실제 휴대폰 촬영의 검증이 아니다.

이후 RankSim·Huber·MAE를 더한 비교에서 작은 점 추정치 차이가 있었으나, 이미 결과를 본 평가자료의 사후 분석이다. 이를 근거로 새 최종 모델을 자동 선정하지 않았다. [17개 방식의 진단 기록](../archive/gdi-research-harness/studies/20261105-gdi-case-audit/README.md)을 별도로 보존한다.

정상 자료에100점을 가정해 추가한 모델은 [논문 관련 모델](../paper-related/normal100-models/README.md)에 분리했다. 외부 출력이 높아졌다는 이유로 기준 모델을 대체하지 않는다.

근거: [고정 평가 설명](../archive/gdi-research-harness/studies/20261102-final223-frozen/README.md), [집계 결과](../archive/gdi-research-harness/studies/20261102-final223-frozen/results.json).
