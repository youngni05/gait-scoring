# 브라우저 통합 테스트

2026-11-18 · `test/browser-integration`

## 검증 범위

Playwright Python 1.56.0의 Chromium이 실제 React 화면을 조작한다.
같은 작업에서 실제 FastAPI와 PostgreSQL 17을 실행하며, 성공 응답을 가짜로 만들어 통과시키지 않는다.
계정은 매 실행마다 임의의 ID·비밀번호로 생성하고 JWT 키도 별도로 생성한다. 실제 계정이나 영상은 사용하지 않는다.

| 사례 | 확인 내용 |
| --- | --- |
| 등록 → 저장 → 수정 → 조회 | 화면 등록·자동 선택, 좌표·메타데이터 DB 저장, 프로필 수정 후 과거 측정·나이·현재 촬영 키 보존, 결과 없음 표시 |
| 저장 응답 유실 | 첫 요청을 실제 API에 전달해 201 저장 후 브라우저에 응답을 전달하지 않음. 재시도 본문·키 동일성과 DB 1건 확인 |
| 인증 만료 | 서버 키로 서명한 만료 토큰을 첫 저장 요청에 사용. 실제 401과 DB 미저장 확인 후 같은 계정으로 인증 갱신·동일 요청 재시도 |
| 대상자 등록 오류 | 잘못된 이름의 화면 검증, 실제 등록 후 응답 유실 시 재등록 버튼 차단, 목록 새로고침과 DB 1건 확인 |

미디어 파일을 준비하는 대신 공유 계약의 합성 좌표를 입력한다.
이후에는 기존 `MeasurementUpload`, `prepareMeasurement`, HTTP 클라이언트, 서버 검증·저장 코드를 그대로 사용한다.
영상 디코딩·MediaPipe·실기기 성능·실제 GDI 실행 검증은 포함되지 않는다.

## 테스트 전용 화면 분리

합성 좌표 버튼은 `import.meta.env.DEV`와 `VITE_E2E_MODE=true`가 모두 참일 때만 로드된다.
일반 개발 서버에는 표시하지 않으며 배포 빌드에는 포함하지 않는다.
CI는 E2E 플래그를 켠 채 production 빌드를 만들고 `scripts/check-production-e2e.mjs`로 테스트 버튼·모듈이 없는지 검사한다.
인증을 생략하거나 운영 API에 테스트 전용 엔드포인트를 추가하지 않는다.

## CI와 로컬 실행

PR에서 기존 Frontend CI/Backend CI와 별도로 **Browser Integration CI**가 실행된다.
Node/Python 의존성, Chromium과 OS 의존성, 전용 PostgreSQL 서비스를 준비한다.
테스트가 API·Vite를 loopback 임의 포트로 실행하며 해당 UI origin만 CORS에 등록한다.

로컬에서도 폐기 가능한 테스트 DB를 준비했다면 실행할 수 있다. 저장소 루트:

```powershell
npm ci
npm run prepare:assets
python -m pip install -r backend/requirements.txt -r backend/requirements-e2e.txt
python -m playwright install chromium
# DATABASE_URL은 본인의 loopback 테스트 DB 접속값으로 설정한다.
# DB 이름은 반드시 _e2e_ci로 끝나야 한다. 운영 .env를 사용하지 않는다.
$env:GAIT_E2E_DB_WRITES = '1'
cd backend
python -m unittest discover -s tests/e2e -v
```

`GAIT_E2E_DB_WRITES=1`, loopback DB 호스트, `_e2e_ci` DB 이름을 모두 확인한 뒤 쓰기를 시작한다.
임의 UUID의 격리 스키마에만 마이그레이션·테스트 데이터를 만들고 종료 시 해당 스키마를 정리한다.
DB 볼륨을 삭제하지 않는다. 강제 종료로 정리가 중단되면 테스트용 스키마가 남을 수 있다.
마이그레이션·서버 시작·브라우저 설치 오류를 테스트 건너뛰기로 숨기지 않고 CI 실패로 처리한다.

## 이번 로컬 검증 결과

- 기존 단위·계약 테스트 141개 통과.
- 모바일 타입 검사와 production 빌드 통과.
- E2E 플래그를 켠 production 빌드에서 테스트 버튼 제외 검사 통과.
- Python E2E 파일 문법 검사 통과.
- 실제 브라우저 통합 테스트 4개는 로컬 미실행: Docker/PostgreSQL 및 Playwright·서버 Python 의존성이 없는 환경이다.

**새 통합 테스트를 통과했다고 간주하지 않는다. PR의 Browser Integration CI 결과를 확인하고 실패 시 수정한 뒤 PM이 병합한다.**

## PR #10 첫 CI 실패와 수정

실행 `44293973510`에서는 설치·DB 마이그레이션·빌드가 통과했지만 네 테스트 모두 로그인 후 대상자 등록 버튼을 찾는 공통 준비 단계에서 실패했다. 원래 로그에는 로그인 HTTP 상태가 없어 네트워크·인증·화면 오류를 구분할 수 없었다.

클라이언트의 기본 transport에 브라우저 `fetch` 함수를 그대로 저장해 `this.transport()`로 호출하던 코드를 수정했다. 전역 객체를 호출 대상으로 유지하도록 감싸 브라우저 네이티브 함수에 ApiClient 인스턴스가 전달되지 않게 했다. 기존 주입형 가짜 transport 테스트 외에 전역 호출 문맥을 확인하는 회귀 테스트를 추가했다.

로그인 테스트는 실제 POST 응답과 200 상태를 먼저 확인하며, 실패 시 API 경로·HTTP 상태와 화면 상태 문구만 남긴다. 비밀번호·토큰·요청 본문은 기록하지 않는다. 로컬 테스트는 142개 통과했으며, 수정 후 실제 E2E 통과 여부는 다음 CI 실행으로 확인해야 한다.
