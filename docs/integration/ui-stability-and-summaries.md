# 저장·화면 이동 안정화 및 목록 요약 조회

## 변경
- 공통 이탈 보호: 측정 준비/저장 중 앱 버튼, 브라우저 뒤로·앞으로, 로그아웃과 새로고침을 보호한다. 취소하면 화면 상태와 요청 본문/키를 유지한다. 브라우저 이탈 확인 문구는 브라우저가 정한다.
- 대상자 등록·수정 완료 후 화면이 이미 종료되었으면 늦은 응답을 무시한다. 서버 저장 취소를 보장하는 기능은 아니다.
- 0값 그래프도 세로축 폭을 확보하여 유한한 SVG 좌표로 표시한다.
- 대상자 목록과 상세는 서버 집계 API를 사용한다. 결과 목록은 20건씩 페이지 조회하고, 좌표/그래프 원문은 조회하지 않는다.
- 기존 API는 유지한다. 추가 API를 포함한 백엔드를 먼저 배포한 다음 프론트를 배포한다. DB 마이그레이션은 추가하지 않았다.

## 추가 API
모두 기존 Bearer 인증이 필요하고 기존 전체 대상자 공유 정책을 유지한다.
- GET /participants/summaries: 기존 대상자 필드 + measurement_count + latest_measured_at
- GET /participants/{id}/summary: 대상자 1명의 동일 요약
- GET /participants/{id}/measurement-summaries?limit=20&offset=0: items, total, offset, limit
  - items: 측정 목록 필드 + has_analysis_result, gdi_score, cadence_spm, stride_length_cm
  - 좌표, 처리 메타데이터, 관절 그래프를 SELECT하거나 반환하지 않는다.
  - 최신일은 analysis_started_at 우선, 없으면 measured_at. 동일 시각은 ID 내림차순.
  - 0은 유효한 수치이며 null과 구별한다.
  - 결과 목록의 '일부 지표 누락'은 목록에 포함된 GDI/케이던스에 대한 안내다. 그래프 검증은 상세 결과지에서 수행한다.

## 검증
- npm test
- npm run typecheck:mobile
- npm run build:mobile
- backend tests/test_measurement_postgres.py: 요약 집계, 기존/신규 시각 정렬, 페이지 경계, 0/null, 인증, 404, 무거운 JSON 제외
- backend tests/e2e/test_design_ui.py: 화면 이탈 거절/허용, 늦은 등록/수정 응답, 페이지별 요청 수, 0값 그래프와 확대 화면

## 범위
승인하고 화면을 나가거나 앱을 종료하면 미확인 측정 요청의 영구 복구는 지원하지 않는다.
기존 저장 중복 방지 계약은 그대로 사용한다. 모델 실행 및 결과 생성은 변경하지 않는다.
offset 페이지 사이에 새 측정이 추가되면 항목 위치가 변할 수 있다. 화면에는 서버의 최신 총건수를 표시한다.

## 이번 작업의 확인 결과
- 로컬 npm test: 157개 통과
- 모바일 TypeScript 검사 및 production 빌드 통과
- production에서 합성 측정 컨트롤 제외 검사 통과
- 실제 JointChart를 SSR 렌더링: 0/양수/음수 상수, 일반/확대 보기 총 6개 경우 유한한 SVG 좌표 확인
- Python 소스와 테스트 문법 검사 통과
- PostgreSQL/Playwright 통합 테스트는 추가했으나 이번 로컬 환경에서는 미실행: Docker 미실행 및 네트워크 제한에 의한 Python 의존성 설치 실패
- 원격 업로드는 자동 승인 검토에서 차단되어 수행하지 못함. 사용자 승인 후 별도 브랜치/PR에서 CI 확인 필요
