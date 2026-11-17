# Happy Aging Backend

해피에이징 보행 영상 수집 및 분석 서비스의 백엔드입니다.

## 기술 스택

- Python 3.12
- FastAPI
- PostgreSQL 17
- SQLAlchemy
- Alembic
- Docker Compose

## 개발 환경 실행

모든 명령어는 프로젝트 최상위 폴더(`happy-aging-app/`)에서 실행합니다.

### 1. 환경변수 설정

프로젝트 최상위 폴더의 `.env.example`을 복사하여 `.env` 파일을 생성합니다.

```powershell
Copy-Item .env.example .env
```

`.env`에 로컬 PostgreSQL 사용자, 비밀번호, DB 이름 및 연결 주소를 설정합니다.

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
현재 GET·POST·PATCH와 Authorization·Content-Type 헤더를 허용하고 OPTIONS 사전 요청은
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

모든 요청에 `Authorization: Bearer <access_token>` 헤더가 필요합니다.
로그인한 복지사는 등록자와 관계없이 모든 어르신의 기록에 접근할 수 있습니다.

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| POST | `/participants/{participant_id}/measurements` | 관절 좌표와 촬영 정보를 저장 (201) |
| GET | `/participants/{participant_id}/measurements?limit=20&offset=0` | 촬영 일시 내림차순 목록 |
| GET | `/measurements/{measurement_id}/result` | 선택한 촬영의 결과지 데이터 |

### 측정 등록 예시

```json
{
  "keypoints_json": {"frames": [{"t": 0, "points": [[0.1, 0.2]]}]},
  "extractor_name": "example-extractor",
  "extractor_version": "1.0",
  "measured_at": "2026-01-10T14:30:00+09:00"
}
```

좌표 예시는 임시 구조이며 AI 연동 규격이 아닙니다. 현재는 비어 있지 않은
JSON 객체 또는 배열과 유한한 숫자인지만 검증하며, 세부 관절 구조는 추후 확정합니다.
원본 영상이나 파일 업로드는 받지 않습니다. `age`, `created_by`, `participant_id`는
요청 본문에 넣지 않습니다. 경로의 어르신 ID와 로그인 사용자, 촬영일로 서버가 결정합니다.

촬영 시각에는 시간대가 필요하며 미래 시각은 거부합니다. 한국 날짜 기준으로
생년월일에서 촬영 당시 만 나이를 계산합니다. 생년월일 미등록은 409,
생년월일 이전 촬영이나 잘못된 입력은 422, 없는 어르신·촬영은 404입니다.

등록 응답과 촬영 목록에는 좌표 원문을 포함하지 않습니다. 목록에는
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
