# Happy Aging Backend

해피에이징 보행 분석 서비스의 백엔드입니다. 원본 영상은 받지 않고 앱에서 추출한 좌표 JSON을 저장합니다.
현재 로그인, 어르신 등록·조회·수정, 측정 저장·목록·결과 조회를 제공합니다. AI 실행·결과 자동 저장은 미연동 상태입니다.

프론트 연동 시 [API 명세](../docs/API_SPEC.md)를 기준으로 사용하세요. 신규 앱은 `processing-1` 요청을 사용하며,
기존 4필드 요청은 호환용으로 유지합니다. 두 요청의 차이는 아래 표에 정리했습니다.

## 기술 스택

- Python 3.12
- FastAPI
- PostgreSQL 17
- SQLAlchemy
- Alembic
- Docker Compose

## PR 자동 검증 (GitHub Actions)

`.github/workflows/backend-ci.yml`은 main 대상 PR 생성·업데이트·재오픈 및 수동 실행 시
`Backend checks` 작업을 실행합니다. 경로 필터 없이 모든 main 대상 PR에 같은 검사를 제공합니다.

- Python 3.12에서 고정된 requirements.txt 설치 및 `pip check`
- 앱·마이그레이션·CI 스크립트·테스트 Python 문법 검사
- 독립 PostgreSQL 17 서비스에서 upgrade → 모델 비교 → downgrade → upgrade → 모델 비교
- API 실행, health·DB 연결·OpenAPI 경로·인증 거부·JWT/비밀번호 해시·CORS 기본 검증
- coords-1·processing-1 스키마와 공유 합성 fixture 검증
- PostgreSQL에서 기존 데이터 보존, 측정 HTTP API·동시 중복 요청·잠금 시간 초과·재시도 검증

실제 `.env`나 GitHub Secrets는 사용하지 않습니다. DB는 작업별 임시 서비스이고
JWT 키는 실행마다 생성합니다. **CI의 downgrade 명령을 개발·운영 DB에서 실행하지 마세요.**
`backend/scripts/ci_smoke.py`는 반복 자동 검증에 필요한 유지 파일이며 일회성 테스트가 아닙니다.
별도 `frontend-ci.yml`은 프론트 테스트·타입 검사·모바일 빌드를 수행하고,
`browser-integration-ci.yml`은 Chromium UI를 실제 API·임시 PostgreSQL에 연결해 검증합니다.
자동 검증 통과가 실제 스마트폰 촬영 품질이나 AI 추론 검증을 의미하지는 않습니다.

테스트 DB를 준비하고 마이그레이션을 적용한 환경에서는 backend 디렉터리에서
`python -m scripts.ci_smoke`로 같은 API 검증을 실행할 수 있습니다.
위 명령은 smoke 검사만 실행합니다. 스키마·PostgreSQL 검사는 이 문서 마지막의 명령을 참고하세요.
병합 필수 검사로 강제하려면 저장소 ruleset/브랜치 보호에서 `Backend checks`를 별도로 지정합니다.

## 개발 환경 실행

모든 명령어는 프로젝트 최상위 폴더(`happy-aging-app/`)에서 실행합니다.

### 1. 환경변수 설정

프로젝트 최상위 폴더의 `.env.example`을 복사하여 `.env` 파일을 생성합니다.

```powershell
Copy-Item .env.example .env
```

`.env`에 로컬 PostgreSQL 사용자, 비밀번호, DB 이름 및 연결 주소를 설정합니다.

JWT 키는 아래 명령으로 생성한 뒤 `.env`의 `JWT_SECRET_KEY` 값을 교체합니다.
생성한 키는 공유하거나 Git에 커밋하지 않습니다.

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

로컬 Python이 없다면 Docker로 생성할 수 있습니다.

```powershell
docker run --rm python:3.12-slim python -c "import secrets; print(secrets.token_urlsafe(32))"
```

키가 없거나 공백, 예시 값, UTF-8 기준 32바이트 미만이면 API 시작을 거부합니다.
길이 검사는 무작위성을 보장하지 않으므로 직접 문구를 만들지 말고 위 생성기를 사용하세요.
키를 변경하면 기존 토큰을 사용할 수 없으므로 다시 로그인해야 합니다.

### 2. Docker 실행

```powershell
docker compose up -d --build
```

### 3. 실행 상태 확인

```powershell
docker compose ps
```

- API 상태: http://localhost:8000/health
- DB 연결 확인: http://localhost:8000/db-health
- API 문서: http://localhost:8000/docs

### 4. 최초 DB 준비와 복지사 계정 생성

새 개발 환경에서는 먼저 마이그레이션으로 테이블을 생성합니다.

```powershell
docker compose exec api alembic upgrade head
```

그다음 관리자가 아래 명령으로 첫 복지사 계정을 생성합니다.

```powershell
docker compose exec api python -m app.create_worker
```

프롬프트에 로그인 ID(1~100자)와 비밀번호(12자 이상)를 입력합니다.
비밀번호 입력은 화면에 표시되지 않습니다. 같은 ID는 중복 생성할 수 없습니다.
생성한 계정으로 `/docs`의 Authorize 또는 `POST /auth/login`을 통해 로그인합니다.
계정 생성은 터미널에서 수행하며 공개 회원가입 API는 제공하지 않습니다.

## 프론트엔드 CORS 설정

프로젝트 루트 `.env`의 `CORS_ALLOW_ORIGINS`에 허용할 프론트엔드 origin을 쉼표로 구분해 지정합니다.

```dotenv
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost,https://localhost,capacitor://localhost
```

위 값은 React 개발 서버 및 Capacitor 앱용 예시입니다. 실제 앱의 `window.location.origin`과
일치하는 값을 사용하세요. scheme·host·port가 모두 일치해야 하며 경로나 끝의 `/`는 넣지 않습니다.
`127.0.0.1`, LAN IP, 다른 포트 또는 live reload 서버를 쓰면 해당 프론트 origin도 별도로 추가합니다.
API 서버 주소가 아니라 요청을 보내는 프론트 주소를 설정합니다.

미설정 또는 빈 값이면 교차 출처를 허용하지 않습니다. `*`는 허용하지 않습니다.
현재 GET·POST·PATCH와 Authorization·Content-Type·Idempotency-Key 헤더를 허용하고 OPTIONS 사전 요청은
CORS 미들웨어가 처리합니다. JWT Bearer 인증이므로 쿠키 credentials는 활성화하지 않습니다.
프론트에서 `credentials: "include"`를 지정할 필요가 없습니다.

`.env` 변경 후에는 환경변수를 다시 읽도록 API 컨테이너를 재생성합니다.
단순 `docker compose restart`만으로는 새 환경변수가 반영되지 않습니다.

```powershell
docker compose up -d --no-deps --force-recreate api
```

CORS는 브라우저의 응답 접근 정책이며 JWT 인증이나 기기에서 API까지의 네트워크 연결을 대체하지 않습니다.

## 데이터베이스 마이그레이션

### 마이그레이션 적용

```powershell
docker compose exec api alembic upgrade head
```

### 현재 마이그레이션 버전 확인

```powershell
docker compose exec api alembic current
```

### DB 모델 변경 시 마이그레이션 생성

```powershell
docker compose exec api alembic revision --autogenerate -m "describe changes"
```

생성된 마이그레이션 파일을 확인한 후 적용합니다.

## 로그 확인

```powershell
docker compose logs --tail=50 api
docker compose logs --tail=50 db
```

## 서버 종료

```powershell
docker compose down
```

## 주의사항

- `.env` 파일은 GitHub에 업로드하지 않습니다.
- PostgreSQL 데이터는 Docker 볼륨에 저장됩니다.
- `docker compose down -v`는 DB 데이터를 삭제할 수 있으므로 주의합니다.
- DB 테이블 변경은 SQLAlchemy와 Alembic으로 관리합니다.
- 실제 개인정보 및 보행 영상은 로컬 개발 환경에 임의로 저장하지 않습니다.

## 측정 및 분석 결과 API

어르신 정보는 `PATCH /participants/{participant_id}`로 일부 수정할 수 있습니다.
`name`, `gender`, `birth_date`, `height_cm` 중 변경할 필드만 전송합니다.
빈 요청, 명시적인 null, 미정의 필드는 422입니다. 누락 필드와 최초 등록자는 유지하며,
생년월일 수정도 기존 촬영의 `age`를 변경하지 않습니다. 결과지의 키는 현재 키를 반영합니다.
전체 요청·응답 예시는 [API 명세](../docs/API_SPEC.md)를 참고하세요.

아래 업무 API에는 `Authorization: Bearer <access_token>` 헤더가 필요합니다.
`POST /auth/login`은 인증 없이 폼 데이터(`username`, `password`)로 호출합니다.
토큰은 8시간 유효하며 갱신 API는 없습니다. `/health`, `/db-health`도 인증이 필요하지 않습니다.
로그인한 복지사는 등록자와 관계없이 모든 어르신의 기록에 접근할 수 있습니다.

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | `/participants/{participant_id}/measurements` | 최초 저장 201, 신규 계약의 동일 성공 요청 재전송 200 |
| GET | `/participants/{participant_id}/measurements?limit=20&offset=0` | 촬영 일시 내림차순 목록 |
| GET | `/measurements/{measurement_id}/result` | 선택한 촬영의 결과지 데이터 |

### 기존 요청과 신규 요청

| 항목 | 기존 4필드 요청 | 신규 processing-1 요청 |
| --- | --- | --- |
| 본문 | keypoints_json, extractor_name, extractor_version, measured_at | 기존 필드 + processing_metadata 객체 |
| Idempotency-Key 헤더 | 생략 | UUID 1개 필수 |
| 좌표 검증 | 비어 있지 않은 JSON 객체/배열, 유한한 숫자 | coords-1 구조와 필드 간 일치 검증 |
| 재전송 | 별도 측정 생성 가능 | 같은 사용자·키·대상자·본문 바이트이면 기존 측정을 200으로 반환 |
| 생년월일 누락 | 409 문자열 detail | 409 객체 detail, 코드 PARTICIPANT_BIRTH_DATE_REQUIRED |
| AI 실행 | 없음 | 없음 |

metadata와 키 중 하나만 보내거나 metadata를 명시적 null로 보내면 422입니다.
신규 요청이 실패했다고 둘을 지워 기존 요청으로 전송하면 중복 방지가 사라집니다.

### 기존 클라이언트 호환용 측정 등록 예시

다음 예시에는 `Idempotency-Key`를 보내지 않습니다. 신규 요청 예시는 [API 명세 12절](../docs/API_SPEC.md#12-processing-1-측정-저장-계약)을 참고하세요.

```json
{
  "keypoints_json": {"frames": [{"t": 0, "points": [[0.1, 0.2]]}]},
  "extractor_name": "example-extractor",
  "extractor_version": "1.0",
  "measured_at": "2026-01-10T14:30:00+09:00"
}
```

위 좌표는 일반 JSON 검증을 통과하는 기존 요청 예시이며 신규 coords-1 형식이나 AI 연동 규격이 아닙니다.
신규 processing-1 요청은 coords-1의 33관절·최소 60프레임·시간·대상자 등을 검증합니다.
원본 영상이나 파일 업로드는 받지 않습니다. `age`, `created_by`, `participant_id`는
요청 본문에 넣지 않습니다. 경로의 어르신 ID와 로그인 사용자, 촬영일로 서버가 결정합니다.

촬영 시각에는 시간대가 필요하며 미래 시각은 거부합니다. 한국 날짜 기준으로
생년월일에서 촬영 당시 만 나이를 계산합니다. 생년월일 미등록은 409,
생년월일 이전 촬영이나 잘못된 입력은 422, 없는 어르신·촬영은 404입니다.

등록 응답·촬영 목록·결과 조회에는 좌표와 processing_metadata 원문을 포함하지 않습니다. 목록에는
`has_analysis_result`가 포함되며 `limit`은 1~100, `offset`은 0 이상입니다.
같은 촬영 시각인 경우 ID 내림차순으로 정렬합니다.

결과 조회는 `participant`(현재 기본 정보와 키), `measurement`(촬영 당시 `age`),
`analysis_result`를 반환합니다. 결과가 없으면 HTTP 200과 함께
`status: "not_available"`, `analysis_result: null`을 반환합니다.
이는 대기·진행·실패를 구별하는 작업 상태가 아닙니다.
결과가 있으면 `status: "available"`이며, 미산출 지표는 null이므로 화면에서 숨깁니다.
키는 Decimal 직렬화에 따라 JSON 문자열로 반환됩니다.

현재 측정 등록은 AI를 호출하거나 분석 결과를 생성하지 않습니다.
분석 결과 저장 및 AI 실행 연동은 추후 구현합니다.

## processing-1 저장과 검증

신규 메타데이터와 요청 키를 함께 보내면 서버에서 coords-1과 필드 일치를 검증하고 DB 트랜잭션으로 중복을 방지합니다.
마이그레이션 `b419d223e901`은 기존 측정의 metadata를 NULL로 유지하고 요청 키 테이블을 추가합니다.
기존 4필드 클라이언트는 그대로 동작합니다. [API 명세](../docs/API_SPEC.md)의 12절을 확인하세요.

- `extractor_name`은 `mediapipe_tasks_vision`, `extractor_version`은 coords의 엔진 버전과 일치해야 합니다.
- 촬영 시각은 coords.capture.recorded_at과 같은 시점, detection.marks는 coords.marks와 같은 값이어야 합니다.
- coords의 키는 100~220 범위의 JSON 숫자입니다. 프로필 등록·수정의 Decimal 키(숫자 문자열도 허용)와 규칙이 다릅니다.
- 같은 키의 본문은 수신 바이트의 SHA-256으로 비교합니다. 공백·키 순서도 바꾸지 말고 직렬화한 본문을 보존합니다.
- `409 REQUEST_IN_PROGRESS`는 `Retry-After` 이후 같은 요청으로 재시도합니다. `409 IDEMPOTENCY_CONFLICT`는 자동 재시도를 중단합니다.
- 네트워크 취소·시간 초과가 저장 취소를 뜻하지 않습니다. 재로그인 시에도 같은 계정·키·본문으로 재시도합니다.
- 오류 detail은 문자열·검증 오류 배열·업무 오류 객체 중 하나입니다. 검증 오류 배열은 loc/type/msg만 반환합니다.

[합성 요청 파일](../contracts/measurement-1/synthetic.request.json)은 method/path/headers/body 묶음입니다.
파일 전체를 HTTP 본문으로 보내지 말고 body만 직렬화하세요. 테스트 어르신 ID를 경로와 coords.subject.id에 동일하게 적용하고,
별도 로그인 토큰을 붙여야 합니다. 실제 데이터나 토큰을 샘플 파일에 저장하지 않습니다.

스키마 단위 테스트 (backend 폴더):

```powershell
python -m unittest discover -s tests -p test_measurement_schema.py -v
```

PostgreSQL/HTTP/동시성 테스트는 DB 이름이 `_ci`로 끝나는 폐기 가능한 테스트 DB에서만 실행합니다.
실제 앱 DB에는 실행하지 마세요. 테스트는 UUID 이름의 독립 스키마에서 마이그레이션과 합성 데이터 쓰기를 수행하고 해당 스키마만 정리합니다.

```powershell
# 먼저 DATABASE_URL을 별도 테스트 DB로 지정합니다. 비밀번호는 Git에 저장하지 않습니다.
$env:GAIT_TEST_DB_WRITES = '1'
python -m unittest discover -s tests -p test_measurement_postgres.py -v
```

GitHub Actions는 PostgreSQL 17에서 위 두 테스트를 실행합니다. 기존 데이터 보존 마이그레이션,
실제 로그인·HTTP 저장/조회, 순차·동시 중복, 실제 유일키 잠금 시간 초과와 재시도, CORS를 검사합니다.
로컬 환경에 PostgreSQL이 없으면 스키마 테스트 통과만으로 DB 통합을 통과했다고 판단하지 않습니다.
