# 앱 백엔드 통합 기반과 다음 작업

작성일: 2026-11-18. 이 PR은 코드 공존과 회귀 검증 단계이며 실제 HTTP 전송이나 GDI 연결 완료가 아니다.

## 통합 기준

- 앱: `758f6ffba14f9a391448af5eaba46f64b5a75415`
- main: `9215b5eac75e885a2d3dfb9c1035fc1c348f216f`
- 작업 브랜치: `integration/app-backend`
- 기존 앱·백엔드 제품 로직은 변경하지 않았다. .gitignore의 중복 .env 규칙을 제거해 .env.example 예외가 유효하도록 했다.
- 기존 미커밋 stage-9-team-request.txt와 stage-9-verification.txt는 로컬에 그대로 두고 이 PR에서 제외했다. 두 문서는 백엔드 구현 이전의 기록이다.
- 최신 main과 앱 이력을 모두 부모로 보존한다. main 직접 병합은 PM이 수행한다.

## 검증

- 통합 전 npm test: 115/115 통과.
- 통합 후 npm test: 115/115 통과.
- npm run typecheck: 통과.
- npm run build:mobile: 모바일 타입 검사, 고정 모델 SHA 검증, WASM 준비, Vite 빌드 통과.
- Python compileall: 통과. 이는 Python 3.10 로컬 문법 검사이며 목표 Python 3.12 실행 검증이 아니다.
- git diff --check: 통과.
- 빌드 경고: 약 558 KB JS 청크, MediaPipe 정적·동적 import 혼재. 후속 프론트 개편 때 진입점과 지연 로딩을 정리한다.
- 기존 node_modules와 준비된 모델 자산으로 검증했다. 이번 작업에서 npm ci 재설치는 하지 않았다.
- 로컬 Docker 및 백엔드 의존성 부재로 PostgreSQL/API 실행은 미검증. PR의 Backend CI 결과를 별도로 확인한다.
- APK 재빌드, 갤럭시 실기기, 공용 서버 HTTP 요청, 실제 GDI는 이번 검증 범위에 포함되지 않는다.

## 실제 API 연결 작업 목록

아래는 후속 구현 요구사항이며 현재 API가 이미 지원한다고 해석하지 않는다.

| 작업 | 담당 | 변경 위치와 기준 |
|---|---|---|
| 실제 로그인·API 클라이언트 | 프론트 | mobile/src/api 계층 신설. 로그인은 폼 username/password, 이후 Bearer 사용. 401 재로그인, detail 문자열/배열 구분 |
| 대상자 연결 | 프론트·백엔드 | 실제 participant UUID를 경로와 coords.subject.id에 사용. 서버 일치 검증. 키 문자열/null 처리 |
| 좌표 요청 어댑터 | 프론트 | UploadGateway를 재사용. coords → keypoints_json, capture.recorded_at → measured_at. 서버 id를 receiptId에 매핑 |
| 추출기 식별 | 양측 | extractor_name/버전 의미 합의. 엔진·모델 종류·자산 버전은 구분해 보존 |
| metadata 보존 | 백엔드 선행 | processing_metadata 필드·DB 저장 제안. 촬영일 출처, 회전, 검출 상태/버전, 자산 버전을 버리거나 coords-1 내부에 임의 추가하지 않음 |
| 중복 저장 방지 | 백엔드 선행 | Idempotency-Key + 로그인 사용자 + 본문 지문. 같은 키/같은 본문은 동일 measurement_id, 다른 본문 409. DB 유일 제약으로 동시성 보장. 보존 기간과 상태 코드 확정 |
| 오류 규격 | 양측 | 기존 생년월일 누락 409와 멱등 충돌 409를 기계 판독 코드로 구분 |
| 네트워크 정책 | 양측 | 공용 HTTPS origin 확정 후 CORS와 mobile/index.html CSP connect-src 반영. Idempotency-Key 허용 헤더 추가 |
| 서버 좌표 검증 | 백엔드 | coords-1 JSON Schema와 시간 증가·대상자 일치 등을 검증. 현재는 일반 JSON 수준 |
| 결과지 테스트 | 양측 | 별도 테스트 DB에 가상 결과 시드. 완료·부분 null·0·결과 없음. 가상 결과 표기, 신규 측정에 자동 가상 결과 생성 금지 |
| 프론트 CI | 프론트 | PR에서 계약 테스트·타입 검사·모바일 빌드. 자산 준비 네트워크와 SHA 검증 포함 |
| 공용 서버 검증 | 백엔드·PM | 계정/비용/HTTPS/origin/DB 환경 확정. 테스트 데이터만 사용, 배포 SHA 기록 |

## 계약상 막히는 부분

현재 백엔드 MeasurementCreate는 keypoints_json, extractor_name, extractor_version, measured_at 외 최상위 필드를 거부한다. 앱의 {coords,metadata,requestId}를 그대로 전송하면 안 된다. metadata를 버리고 전송 성공만 만드는 우회는 하지 않는다.

현재 앱의 키 범위는 100~220cm, 백엔드 등록은 0 초과다. 프로필 등록은 가능해도 추출이 불가능한 경우를 명확히 안내해야 한다. 범위를 조용히 바꾸거나 값을 잘라 맞추지 않는다.

앱의 날짜 직접 입력은 한국 00:00이며 실제 촬영 시각이 아닐 수 있다. measured_at 매핑과 함께 출처·정확도를 보존한다. 촬영 당시 나이는 서버가 계산한다.

현재 결과 status=not_available은 분석 중이 아니다. 분석 호출과 자동 저장은 없으며 결과 테이블 조회만 구현되어 있다.

기존 연구용 모델은 MediaPipe33 JSON을 바로 받는 것으로 검증되지 않았다. 원본 PTS·null·33관절을 유지하고 모델 변환은 GDI 담당자가 규격을 정한다.

## 다음 PR 순서와 수락 기준

1. 계약: metadata, requestId/멱등성, extractor 식별, 오류 코드, 합성 샘플을 먼저 확정한다.
2. 백엔드: 저장 확장·멱등성·서버 검증·업무 시나리오 테스트를 구현한다.
3. 프론트: 디자인 2안 중 PM 선택 후 로그인·어르신 관리 및 실제 게이트웨이를 연결한다. API 계층 작업은 디자인과 독립 진행 가능하다.
4. 종단 검증: 갤럭시 → 공용 서버 → 한 번 저장 → 기록 재조회 → 가상 결과 API 조회.
5. 실제 모델: 입력 변환·작업 상태·결과 검증 및 저장을 연결한다.

1차 목표는 2026-11-25, 최종 시연은 2026-12-18이다. PR은 PM이 검증 후 병합한다. 공용 서버 비용·디자인은 아직 미정이며 이번 PR에서 배포하거나 확정하지 않는다.
