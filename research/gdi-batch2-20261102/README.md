# GDI 추가 공개 2차 — 남은 실험의 절반

**대학 과목 제출을 위한 비상업적 연구 결과다. GDI만 대상으로 하며, SPPB·TUG를 이번에 추가하지 않는다.** 이미 공개한 초기 CNN·695명 MotionBERT 실험은 중복 집계하지 않았다.

완료·검산 기록과 저장 모델이 있는 GDI 실험을 계열별로 묶어 남은 **25종 중 13종**을 이번에 공개한다. 홀수이므로 절반을 올림했다. 같은 실험의 비교 조건·분할·초기값은 한 종류로 세고, 파일 수를 모델 종류 수로 부풀리지 않았다. 기존 고정 모델의 영상 적용·사후 집계와 파일럿만 완료되거나 진행 중인 실험은 제외했다. 175명 GPU MotionBERT 초기 비교는 이미 공개한 695명 부분학습과 같은 계열로 묶었다. CPU 출력부의 후속 탐색도 부모 실험 이력으로 묶으며 이번 묶음의 실제 가중치 범위는 각 문서에 명시한다.

선택은 기록된 실험 날짜순이며, 같은 날짜는 원 실험 경로순으로 정했다. 이 날짜는 일 단위 연구기록이고 정밀한 작업 종료 시각 순서를 뜻하지 않는다. 성능이 좋아 보이는 모델만 골라 공개한 것이 아니다. [선택·보류 전체 목록](selection.json)에 나머지 **12종**과 제외 기준을 함께 기록했다. 목록은 이번 공개를 위한 보유 기록의 스냅샷이며 이후 완료되는 실험이 자동 포함되는 것은 아니다.

| 원 실험 날짜 | 이번 공개 실험 | 가중치 파일 수¹ |
|---|---|---:|
| 2026-10-16 | [고정 MotionBERT 특징·243프레임 Ridge](experiments/ridge243/README.md) | 1 |
| 2026-10-25 | [사람별 학습 비중 Ridge](experiments/weighted-ridge/README.md) | 1 |
| 2026-10-27 | [구간 위치·정규화 Ridge 비교](experiments/window-ablation/README.md) | 1 |
| 2026-10-28 | [누락 증강과 일반 추가 학습](experiments/missing-augmentation/README.md) | 1 |
| 2026-10-28 | [반복성 특징·유효 관절 집계](experiments/rhythm-pooling/README.md) | 1 |
| 2026-10-30 | [175명 고정 특징 출력부 학습](experiments/cpu-head175/README.md) | 5 |
| 2026-10-30 | [695명 RF·시간별 특징 CNN 비교](experiments/fusion695/README.md) | 20 |
| 2026-10-30 | [고정 CNN·MotionBERT 잔차 보정](experiments/cnn-frozen-residual692/README.md) | 10 |
| 2026-10-30 | [MotionBERT 특징·CNN 결합](experiments/cnn-motionbert-fusion692/README.md) | 5 |
| 2026-10-30 | [여러 보행 구간 CNN](experiments/cnn-multiwindow692/README.md) | 5 |
| 2026-10-30 | [원 연구식 입력 한 구간 CNN](experiments/cnn-paper-input692/README.md) | 5 |
| 2026-10-31 | [기록 평균 손실 CNN 비교](experiments/cnn-record-loss692/README.md) | 10 |
| 2026-10-31 | [GDI 10점 구간 확률 출력부](experiments/probability-bands692/README.md) | 20 |

¹ 파일 하나가 여러 Ridge·MLP 파라미터를 담기도 하고, 반대로 한 실험이 여러 분할 파일을 사용하기도 한다. 따라서 이 열은 실험 수나 독립 임상 표본 수가 아니다.

## 모델·코드·검증

[GitHub Release](https://github.com/youngni05/gait-scoring/releases/tag/research-gdi-half-20261102)에 13개 압축을 따로 첨부했다. 저장소에는 실험 목적·입력·비교·결과·한계, 선택한 핵심 코드, 집계 결과와 이전 검산 기록의 공개본을 보존한다. 코드·설정의 원본 대응과 공개용 경로·개인 명단 제거는 [source-files.json](source-files.json)에 있다.

이번 검사는 원본과 공개 파라미터의 동일성, 추론용 텐서 재로딩, 압축 내 파일 지문과 라이선스 동봉을 확인했다. 이전 실험의 예측 재현·지표 검산과 구분하며 새 학습·설정 선택·성능 재평가를 수행하지 않았다. [publication-verification.json](publication-verification.json)에 압축별 SHA-256·파일 수·검증 범위를 기록했다.

각 압축에 `RESEARCH_USE.md`, `SOURCES.md`, `MODIFICATIONS.md`, `NOTICE.md`, `LICENSES/`를 동봉했다. 원 데이터의 CC BY-NC 2.0과 참고 코드의 Apache 2.0을 구분하고, 대학 과목 제출용 비상업적 목적을 명시한다. [기존 라이선스 안내](../licenses/README.md)의 원권리자 권리 보존 원칙을 유지한다. 원자료·사전학습 본체·개인별 정답·사람 명단은 이 묶음에 없다.

실험마다 사람 수·입력 길이·전처리·평가 집단이 다르므로 한 통합 성능 순위로 해석하지 않는다. 임상 타당성·고령자 현장·단일 RGB 적용을 검증한 모델 묶음도 아니다. 사용 방법과 입력 준비의 한계는 각 실험 설명을 따른다.
