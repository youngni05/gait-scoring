# 새 디자인 화면의 API 연동

이 문서는 이전 팀 화면의 계약을 설명한다. 현재 해피워크는 영상에서 읽거나 사용자가 선택한 촬영일을 저장하고 팀 워커가 분석한다.

2026-11-20. `feature/ui-redesign`의 화면을 기반으로 기존 FastAPI 계약에 연결했다.
`npm run dev:legacy` 실행 후 `/design.html`에서 이 화면을, `/`에서 연동 점검 화면을 연다.
분석 시작일을 위한 DB/API 계약을 추가했다. AI 실행 기능은 아직 미연동이다.

**2026-11-20 후속 요구사항:** 촬영 시각 대신 분석한 날짜를 저장·표시한다.
좌표 추출 시작 시각을 `analysis_started_at`으로 저장하며 한국 시간 기준 시작일의 만 나이를 계산한다.
촬영 시각 메타데이터는 요구하지 않는다. [날짜 요구사항 변경](../AI_INTEGRATION_SPEC_DRAFT.md#2026-11-20-날짜-요구사항-변경)을 참고한다.

## 화면과 API

| 화면 | 연결 |
| --- | --- |
| 로그인 | `POST /auth/login`, 폼 데이터, 메모리 JWT |
| 대상자 목록·검색 | `GET /participants`, 이름 검색은 프론트에서 수행 |
| 최근 측정일·측정 수 | 대상자별 측정 목록의 전체 페이지 집계 |
| 대상자 상세 | `GET /participants/{id}` |
| 등록·수정 | `POST /participants`, `PATCH /participants/{id}` |
| 영상 측정 | 기기에서 `prepareRecording` → `prepareAnalysisMeasurement` → `POST /participants/{id}/measurements` |
| 측정 목록 | `GET /participants/{id}/measurements?limit=20&offset=…`와 결과 상세 조회 |
| 결과지 | `GET /measurements/{id}/result`, `available`/`not_available` 구분 |
| 다운로드 | 결과 API 응답을 브라우저에서 JSON 파일로 저장 요청. PDF 생성은 미구현 |

연동 중심 코드는 `mobile/src/ui/data/apiSource.ts`이고 HTTP·인증 헤더·오류 처리는
기존 `mobile/src/api/client.ts`를 재사용한다. 화면에서는 `dataSource`를 사용한다.
목록의 추가 조회는 최대 네 개씩 병렬 실행한다. 대상자가 많아지면 백엔드 집계 API와
UI 페이지네이션을 별도 도입해야 한다.

## 상태와 재시도

- 토큰은 메모리에만 보관한다. 새로고침 시 재로그인이 필요하다.
- 업무 요청의 401은 기존 화면을 유지한 채 재로그인 창을 연다. 같은 계정으로 갱신한 후
  기존 화면에서 다시 시도한다. 저장 요청을 자동 재전송하지 않는다.
- 측정 저장은 준비된 본문과 `Idempotency-Key`를 그대로 재사용한다.
  인증 갱신·응답 유실·취소 이후에도 같은 화면에서 같은 요청으로 재시도한다.
- 등록 API에는 중복 방지 키가 없다. 등록 응답을 확인하지 못하면 재등록 버튼을 막고 목록 확인을 안내한다.
- 화면 종료/새로고침 이후에는 메모리의 미확인 측정 요청이 사라진다. 기록 확인 후 후속 작업을 판단한다.
- 원본 영상·좌표·토큰을 localStorage에 저장하지 않는다. 원본 영상은 서버에 보내지 않는다.

## 결과 표현과 남은 계약

- 나이는 분석 시작일 기준 저장된 만 나이, 이름·성별·키는 현재 대상자 정보다. 기존 촬영 기록은 기존 기준을 명시한다.
- GDI·Cadence·stride 길이의 `0`과 `null`을 구분한다. 좌우 대칭 지표는 API에 없어 실제 화면에 제공하지 않는다.
- 그래프 JSON은 아직 서버 측 축·단위 계약이 없다. 임의 배열을 그래프로 해석하지 않는다.
  프론트는 **향후 협의용** `schema_version: "gait-cycle-degrees-1"`와 유한한 51점의
  `left`/`right` 배열만 표시할 수 있게 준비했다. 이는 백엔드·AI가 합의한 계약이 아니며
  현재 생산자가 이 형식을 생성한다고 가정하지 않는다. 다른 JSON은 안내를 표시하고 JSON 다운로드에는 원형 그대로 포함한다.
- 측정 저장은 분석 시작이 아니다. 실제 AI 호출·결과 저장·진행률 API는 미연동이다.
- 영상 좌표 추출은 실제 MediaPipe 경로를 사용한다. 촬영일 선택·촬영 메타데이터 없이 처리하며 최초 분석 시작 시각을 저장 재시도에도 유지한다.

## PC 실행

백엔드는 `backend/README.md`의 DB 마이그레이션·계정 생성·CORS 설정을 따른다.
프론트 origin `http://127.0.0.1:5173`을 서버 CORS에 등록한다.

```powershell
npm ci
npm run dev:mobile -- --port 5173 --strictPort
```

`http://127.0.0.1:5173/`에서 서버 주소와 실제 복지사 계정으로 로그인한다.
기본 API 주소는 `http://127.0.0.1:8000`이고 `VITE_API_BASE_URL` 또는 로그인 입력으로 지정한다.
외부 API는 HTTPS가 필요하다.

## 검증

```powershell
npm test
npm run typecheck:mobile
npm run build:mobile
node scripts/check-production-e2e.mjs
.venv/Scripts/python.exe scripts/run-local-browser-tests.py
```

마지막 명령은 실행 중인 Compose DB 안에 무작위 이름의 별도 테스트 DB를 만든다.
실제 계정·업무 DB를 수정하지 않으며 종료 시 생성한 DB만 정리한다.
Python 백엔드 의존성·`backend/requirements-e2e.txt` 및 Playwright Chromium 설치가 필요하다.
새 화면 테스트는 등록·수정·합성 측정 저장·결과 없음·저장된 부분 결과·JSON 다운로드,
401 재로그인 후 같은 키 재시도, 등록 응답 유실·로그아웃·잘못된 로그인을 확인한다.
합성 테스트는 실제 영상 추출 품질이나 AI 추론을 검증하지 않는다.
