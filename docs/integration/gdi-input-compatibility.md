# GDI 모델 입력 호환성 점검 및 다음 통합 작업

검토일: 2026-11-19. 기준 main: `3bc3c251f6f69e439d7bcca52b8dd3c97500dc48`.

## 현재 결론

모델은 이미 공개되어 있다. `research/selected/frozen-gdi`에 추론 코드와 모델 목록이 있으며,
[Release](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)의
`gdi-frozen54-20261106.tar.gz`에 가중치가 있다. 그러나 앱의 MediaPipe 좌표를 그대로 넣을 수는 없다.
이번 작업은 가중치를 다운로드하거나 실제 추론을 실행한 검증이 아니다.

## 확인된 입력과 처리

| 항목 | 선택된 frozen 모델 | 현재 앱과의 차이 |
|---|---|---|
| 입력 | 유한한 float 배열 `[windows,2,124,12]` | 앱은 프레임별 33개 `[x,y,z,visibility]` |
| 좌우 | 왼쪽, 오른쪽 | 화면 좌우가 아닌 해부학적 좌우를 보존해야 함 |
| 특징 12개 | 발목 xy, 무릎 xy, 엉덩이 xy, 엄지발가락 xy, 무릎각, 발목각, 발끝-발목 거리, 반대쪽 발목 x - 해당 발목 x | 마지막 값은 부호 있는 x 차이이며 양발의 유클리드 거리가 아님 |
| 좌표 단위 | 영상 평면 좌표에서 전처리 | x×width, y×height로 축 비율을 먼저 복구; z는 사용하지 않음 |
| 전처리 | 0을 결측 처리, 선형 보간 및 끝부분 평균 대체, sigma=1 Gaussian 필터, 양쪽 hip 중점 중심화, 양쪽 hip-knee 길이 평균으로 정규화 | 현재 앱은 이 처리를 하지 않음; 신장으로 나누는 처리가 아님 |
| 각도 | arccos 기반 unsigned radians | 도 단위가 아님; 퇴화 관절이나 비유한 결과를 허용하면 안 됨 |
| 시간 | 전체 시퀀스를 전처리한 뒤 124프레임 창 추출 | 창별 필터링은 경계 결과가 달라짐; PTS 기반 재샘플링을 임의 추가하면 안 됨 |
| 출력 | 기록마다 좌우 점수 2개, 원래 calibration 사용 | 기존 단일 gdi_score에 평균을 임의 저장하지 않음; 0~100 clipping 금지 |

고정 평가 protocol의 sliding 시작 인덱스는 `range(0,500-124,31)`이고, 요구 좌표별
0/NaN 비율의 최댓값이 25% 이하인지 판정한다. canonical 124프레임의 적격성 및 fallback도
있다. 이 규칙은 원 데이터의 frame ID와 촬영 조건에 의존하므로 앱의 제품 규칙으로 그대로 확정하지 않는다.
원 소스의 `FPS=25.0` 상수만 보고 앱을 25fps로 강제하지 않는다.

`cnn_refit`은 5개 모델, `cnn_bag`는 24개 모델이다. bag은 fold 내부 평균 후 fold 평균이므로
24개 모델의 단순 동일 가중 평균과 다르다. fusion은 외부 STT 코드와 이용 조건 확인이 추가로 필요하다.

## 재현 가능한 근거 위치

아래 경로는 저장소 루트 기준이다.

- `research/selected/frozen-gdi/README.md`, `predict.py`: tensor 검증, 가중치 hash 확인, window_record 집계, calibration 및 추론 모드.
- `research/archive/gdi-research-harness/studies/20261102-final223-frozen/protocol.json`, `prepare.py`: 입력 선택, 창과 결측 기준, LEFT/RIGHT 특징 인덱스.
- `research/archive/gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-paper-input695/bundle/sources/video_process_utils.py`: `fill_nan`, `filter_frames`, `preprocess_frames`, `get_angle`, `get_distance`.
- 같은 `cnn-paper-input695`의 `package.py`, `verify.py`: 특징 순서와 signed ankle gap.
- 같은 study의 `cnn-multiwindow695/prepare_inputs.py`: 원 소스 로딩과 전체 프레임 처리.
- `research/archive/gdi-research-harness/studies/20261101-self-video-gdi/prepare_features.py`: 자체 촬영도 BODY25 입력으로 처리한 사례이며 MediaPipe 등가성 검증은 아님.
- `contracts/coords-1/types.ts`, `validate.ts`: 앱의 좌표/PTS/회전 후 크기 계약.
- [MediaPipe PoseLandmark enum](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/pose.py): 관절 인덱스 명칭. 재현용 비교 실험 시 사용한 MediaPipe 버전과 모델 asset을 별도로 고정한다.

주의: `research/2026-10-30-cnn-baseline695/input_contract.json`의 17관절 계약은 선택된 frozen
모델의 입력 변환 명세로 사용하지 않는다.

## 이번 구현 범위

`contracts/gdi-candidate-1/prepare.ts`의 `prepareGdiCandidate(unknown)`은 기존 coords-1 검증을
통과한 입력에서 다음 픽셀 좌표만 추출한다. 서버/화면의 운영 경로에는 아직 연결하지 않는다.

| 출력 | 왼쪽 MediaPipe 인덱스 | 오른쪽 MediaPipe 인덱스 |
|---|---:|---:|
| ankle | 27 | 28 |
| knee | 25 | 26 |
| hip | 23 | 24 |
| footIndex | 31 | 32 |

footIndex를 bigToe라는 이름으로 바꾸지 않는다. 원본의 시간·프레임 수·null 프레임·visibility를
유지한다. x=0/y=0인 실제 영상 경계점을 결측으로 바꾸지 않으며 낮은 visibility도 임의 제거하지 않는다.
추출 결과는 별도 객체이며 입력을 수정하지 않는다. `readyForInference: false`와 미해결 사유를
항상 반환한다. 따라서 이 함수의 성공은 모델 실행 가능 또는 품질 통과를 의미하지 않는다.

## 모델 실행 전 반드시 해결할 항목

1. **원 전처리 재현:** 로더가 요구하는 `sources/provenance.json`, `process_raw_videos.ipynb`가
   검토한 저장소의 해당 source 폴더에는 없다. Release 또는 연구 담당자에게 원본·hash·실행 환경을
   확보하고, 동일 BODY25 fixture에 대해 원 전처리와 새 구현의 tensor를 비교한다. 현재 확인한
   소스 조각만으로 전체 전처리 등가성을 주장하지 않는다.
2. **관절 추출기 차이:** MediaPipe FOOT_INDEX와 BODY25 BIG_TOE는 동일하다고 확인되지 않았다.
   같은 허가된 영상에 두 추출기를 적용해 좌표·12특징·점수 차이를 검증한다. 대체 불가하면
   BODY25 추출 경로 유지 또는 모델 재검증이 필요하다.
3. **결측 정책:** MediaPipe visibility와 OpenPose confidence는 같지 않다. 경계의 실제 0좌표를
   원 모델의 결측 sentinel과 분리하고 품질 기준을 연구 담당자와 확정한다.
4. **시간 및 보행 구간:** 앱의 stand_start/end는 기립 동작 구간이며 보행 창으로 사용하지 않는다.
   촬영 방향, 보행 구간, 프레임 간격, 길이 부족 시 거절 조건을 결정한다. 124개 프레임을 맞추기
   위한 임의 padding/재샘플링은 금지한다.
5. **결과 계약:** 모델 담당자와 초기 모드 및 calibration을 선택하고, PM은 좌우 점수 표시 방법을
   결정한다. 단일 점수가 필요하면 집계 규칙을 별도 승인·버전 관리한다. 정규화된 거리를 cm 단위
   보폭으로 표시하지 않는다.

## 다음 에이전트의 개발 순서

1. 위 원본과 합성/사용 허가된 기준 fixture를 확보하고 exact preprocessing parity 테스트 작성.
2. MediaPipe 대체 관절과 시간/결측 정책 검증 후 입력 adapter 버전 고정.
3. `cnn_refit` 등 합의한 모드로 가중치 SHA-256 및 유한한 좌우 출력 smoke test 수행.
4. backend에 모델/전처리 버전, 좌우 결과, pending/running/succeeded/failed 상태 및 중복 실행 방지 저장.
5. frontend에 분석 대기/실패/재시도/좌우 결과를 연결하고 실제 추론을 포함한 통합 테스트 추가.

원 영상, 실사용자 좌표, 비밀번호는 fixture나 로그에 넣지 않는다. 커밋 메시지는 한글로 작성하고
최종 PR 검토 및 병합은 PM이 수행한다.
