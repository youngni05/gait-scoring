# 해피에이징 프론트엔드 연동 API 명세

작성일: 2026-11-17 · 현재 구현 기준

AI 분석 실행은 아직 연결되지 않았습니다. 예시 데이터는 모두 가상입니다.

## 1. 공통 규칙

| 항목 | 내용 |
| --- | --- |
| 로컬 개발 주소 | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| OpenAPI JSON | `http://localhost:8000/openapi.json` |
| 인증 | 로그인 외 아래 업무 API는 JWT Bearer 토큰 필수 |
| 요청 형식 | 로그인은 폼 데이터, 나머지 등록 API는 JSON |
| ID | UUID 문자열 |
| 생년월일 | `YYYY-MM-DD` |
| 촬영·분석 일시 | 시간대가 포함된 ISO 8601 문자열 |
| 키 응답 | 숫자 문자열 또는 `null` — 예: `"160.50"` |
| 미산출 분석값 | `null` |

```http
Authorization: Bearer <access_token>
```

로그인한 복지사는 등록자와 관계없이 모든 어르신·촬영·분석 결과를 조회할 수 있습니다.

실제 스마트폰에서 `localhost`는 스마트폰 자신을 가리킵니다. 연결 가능한 개발 PC 주소를 사용해야 합니다.
CORS 허용 주소는 서버 `.env`의 `CORS_ALLOW_ORIGINS`로 관리합니다.

```dotenv
CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:3000,http://localhost,https://localhost,capacitor://localhost
```

실제 프론트의 `window.location.origin`과 일치하는 값을 등록합니다. 경로나 끝의 `/`는 넣지 않습니다.
다른 포트, LAN IP, Capacitor live reload 주소를 사용하면 해당 origin을 추가해야 합니다.
환경변수 변경 후 `docker compose up -d --no-deps --force-recreate api`로 API 컨테이너를 재생성합니다.
미설정·빈 값은 교차 출처를 허용하지 않으며 `*`는 허용하지 않습니다.
GET·POST·PATCH 및 Authorization·Content-Type 헤더를 허용합니다. 쿠키 인증을 쓰지 않으므로
프론트 요청에 `credentials: "include"`를 지정하지 않습니다.

## 2. API 목록

| 메서드 | 경로 | 기능 | 성공 코드 |
| --- | --- | --- | --- |
| POST | `/auth/login` | 로그인 | 200 |
| GET | `/auth/me` | 로그인 사용자 조회 | 200 |
| POST | `/participants` | 어르신 등록 | 201 |
| GET | `/participants` | 어르신 전체 목록 | 200 |
| GET | `/participants/{participant_id}` | 어르신 상세 | 200 |
| PATCH | `/participants/{participant_id}` | 어르신 정보 일부 수정 | 200 |
| POST | `/participants/{participant_id}/measurements` | 촬영·좌표 기록 저장 | 201 |
| GET | `/participants/{participant_id}/measurements` | 어르신별 촬영 목록 | 200 |
| GET | `/measurements/{measurement_id}/result` | 촬영별 결과지 | 200 |

화면 흐름: 로그인 → 어르신 목록·등록 → 어르신 선택 → 촬영 목록 → 촬영 선택 → 결과지.

새 촬영 등록 흐름: 앱에서 영상 선택 → 앱에서 관절 좌표 추출 → 좌표 JSON 전송 → 측정 저장.
원본 영상은 서버로 전송하지 않으며 앱과 서버에서 별도로 영구 저장하지 않습니다.

## 3. 로그인

### `POST /auth/login`

인증 불필요. **JSON이 아닌 `application/x-www-form-urlencoded`**로 전송합니다.
로그인 ID의 요청 필드 이름은 `username`입니다.

```http
Content-Type: application/x-www-form-urlencoded
```

```text
username=worker01&password=example-password
```

성공 응답 `200`:

```json
{
  "access_token": "<JWT>",
  "token_type": "bearer"
}
```

React/TypeScript 요청 예시 (`API_BASE_URL`은 프론트 환경설정 값):

```ts
const response = await fetch(`${API_BASE_URL}/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/x-www-form-urlencoded" },
  body: new URLSearchParams({ username: loginId, password }),
});

if (!response.ok) {
  throw new Error("로그인에 실패했습니다.");
}

const { access_token } = await response.json();
```

토큰 유효기간은 현재 **8시간**입니다. 토큰 갱신 API는 없으며 만료되면 다시 로그인합니다.

로그인 실패 `401`:

```json
{ "detail": "Invalid credentials" }
```

## 4. 로그인 사용자 조회

### `GET /auth/me`

인증 헤더 외 요청 본문은 없습니다.

성공 응답 `200`:

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "login_id": "worker01"
}
```

## 5. 어르신 등록

### `POST /participants`

```http
Content-Type: application/json
Authorization: Bearer <access_token>
```

요청:

```json
{
  "name": "김예시",
  "gender": "F",
  "birth_date": "1950-03-12",
  "height_cm": "160.50"
}
```

| 필드 | 필수 | 입력 규칙 |
| --- | --- | --- |
| `name` | 예 | 앞뒤 공백 제거 후 1~100자 |
| `gender` | 예 | `"M"`=남, `"F"`=녀 |
| `birth_date` | 예 | 생년월일, 한국 날짜 기준 미래 날짜 불가 |
| `height_cm` | 예 | 0 초과, 전체 최대 5자리·소수 최대 2자리 |

키는 요청에서 숫자 또는 숫자 문자열을 받을 수 있습니다. 응답에서는 문자열로 반환합니다.

성공 응답 `201`:

```json
{
  "id": "22222222-2222-4222-8222-222222222222",
  "name": "김예시",
  "gender": "F",
  "birth_date": "1950-03-12",
  "height_cm": "160.50",
  "created_by": "11111111-1111-4111-8111-111111111111"
}
```

`id`는 서버가 생성하고 `created_by`는 로그인한 복지사로 지정합니다. 프론트에서 보내지 않습니다.

## 6. 어르신 목록·상세 조회

### `GET /participants`

요청 본문은 없습니다. 성공 응답 `200`:

```json
[
  {
    "id": "22222222-2222-4222-8222-222222222222",
    "name": "김예시",
    "gender": "F",
    "birth_date": "1950-03-12",
    "height_cm": "160.50",
    "created_by": "11111111-1111-4111-8111-111111111111"
  }
]
```

- 데이터가 없으면 `[]`입니다.
- 현재 어르신 목록은 페이지네이션·검색·정렬 조건을 지원하지 않습니다.
- 반환 순서를 특정 기준으로 가정하지 않아야 합니다.

### `GET /participants/{participant_id}`

응답 `200`은 위 배열 안의 단일 어르신 객체와 동일합니다.
기존 어르신은 다음처럼 신규 정보가 없을 수 있습니다.

```json
{
  "id": "33333333-3333-4333-8333-333333333333",
  "name": "기존 어르신",
  "gender": null,
  "birth_date": null,
  "height_cm": null,
  "created_by": "11111111-1111-4111-8111-111111111111"
}
```

없는 어르신은 `404`:

```json
{ "detail": "Participant not found" }
```

### `PATCH /participants/{participant_id}` — 어르신 정보 보완·수정

JWT 인증 필수이며 `Content-Type: application/json`으로 전송합니다.
등록자가 아닌 복지사도 수정할 수 있습니다.

요청 예시 — 기존 어르신의 누락 정보 보완:

```json
{
  "gender": "F",
  "birth_date": "1950-03-12",
  "height_cm": "160.50"
}
```

키만 변경하려면 `{"height_cm": "162.00"}`처럼 변경할 필드만 전송합니다.

| 필드 | 필수 여부 | 입력 규칙 |
| --- | --- | --- |
| `name` | 선택 | 공백 제거 후 1~100자 |
| `gender` | 선택 | `"M"` 또는 `"F"` |
| `birth_date` | 선택 | `YYYY-MM-DD`, 한국 날짜 기준 미래 날짜 불가 |
| `height_cm` | 선택 | 0 초과, 전체 최대 5자리·소수 최대 2자리 |

- 최소 한 필드는 필요합니다. `{}`는 422입니다.
- 보내지 않은 필드는 기존 값을 유지합니다. 기존 NULL도 그대로 유지됩니다.
- 명시적으로 `null`을 보내 기존 값을 지우는 것은 허용하지 않습니다(422).
- `id`, `created_by` 등 미정의 필드는 422입니다.
- 최초 등록 복지사와 기존 촬영의 `age`는 변경하지 않습니다.
- 수정된 생년월일은 이후 새로 등록하는 촬영의 만 나이 계산에 사용됩니다.
- 현재 키는 과거 결과지에도 반영되지만 저장된 분석 수치를 재계산하지 않습니다.

성공 응답 `200` — 수정 후 전체 어르신 정보:

```json
{
  "id": "22222222-2222-4222-8222-222222222222",
  "name": "김예시",
  "gender": "F",
  "birth_date": "1950-03-12",
  "height_cm": "160.50",
  "created_by": "11111111-1111-4111-8111-111111111111"
}
```

인증 누락은 401, 없는 어르신은 404, 입력 검증 오류는 422입니다.

## 7. 촬영·관절 좌표 기록 저장

### `POST /participants/{participant_id}/measurements`

인증 필수, `Content-Type: application/json`.
앱에서 영상 선택과 좌표 추출을 완료한 뒤 호출합니다. **영상 파일은 전송하지 않습니다.**

요청:

```json
{
  "keypoints_json": {
    "frames": [
      { "t": 0, "points": [[0.1, 0.2], [0.3, 0.4]] }
    ]
  },
  "extractor_name": "example-extractor",
  "extractor_version": "1.0",
  "measured_at": "2026-01-10T14:30:00+09:00"
}
```

**좌표 JSON은 요청 형태를 보여주는 예시이며 확정된 AI 연동 규격이 아닙니다.**

| 필드 | 필수 | 입력 규칙 |
| --- | --- | --- |
| `keypoints_json` | 예 | 비어 있지 않은 JSON 객체 또는 배열 |
| `extractor_name` | 예 | 앞뒤 공백 제거 후 1~100자 |
| `extractor_version` | 예 | 앞뒤 공백 제거 후 1~100자 |
| `measured_at` | 예 | 실제 촬영 일시, 시간대 필수, 미래 불가 |

- 좌표의 `NaN`, `Infinity`는 허용하지 않습니다. 세부 관절 구조는 아직 검증하지 않습니다.
- 촬영일은 생년월일보다 앞설 수 없습니다.
- 본문에 `age`, `created_by`, `participant_id` 등 정의되지 않은 필드를 넣으면 `422`입니다.
- 서버가 한국 날짜 기준으로 촬영 당시 만 나이를 계산합니다.
- 중복 제출 방지 기능은 아직 없습니다. 같은 요청을 다시 보내면 새 기록이 생기므로 중복 클릭·자동 재전송에 유의합니다.

성공 응답 `201`:

```json
{
  "id": "44444444-4444-4444-8444-444444444444",
  "participant_id": "22222222-2222-4222-8222-222222222222",
  "age": 75,
  "extractor_name": "example-extractor",
  "extractor_version": "1.0",
  "measured_at": "2026-01-10T05:30:00Z",
  "created_by": "11111111-1111-4111-8111-111111111111"
}
```

응답에 좌표 원문은 포함하지 않습니다. 시간은 같은 시점을 나타내는 시간대 포함 문자열로 처리합니다.

생년월일이 없는 기존 어르신은 `409`:

```json
{ "detail": "Participant birth date is required before recording a measurement" }
```

`PATCH /participants/{participant_id}`로 생년월일을 보완한 뒤 측정 등록을 다시 요청합니다.

**등록 성공은 측정 저장 완료를 의미합니다. 현재 AI를 실행하거나 분석 결과를 생성하지 않습니다.**

## 8. 어르신별 촬영 목록

### `GET /participants/{participant_id}/measurements?limit=20&offset=0`

| 파라미터 | 기본값 | 허용 범위 |
| --- | --- | --- |
| `limit` | 20 | 1~100 |
| `offset` | 0 | 0 이상 |

성공 응답 `200`:

```json
[
  {
    "id": "44444444-4444-4444-8444-444444444444",
    "participant_id": "22222222-2222-4222-8222-222222222222",
    "age": 75,
    "extractor_name": "example-extractor",
    "extractor_version": "1.0",
    "measured_at": "2026-01-10T05:30:00Z",
    "created_by": "11111111-1111-4111-8111-111111111111",
    "has_analysis_result": false
  }
]
```

- 최신 촬영일 순이며 촬영일이 같으면 ID 내림차순입니다.
- 촬영 기록이 없으면 `[]`, 어르신이 없으면 `404`입니다.
- 전체 건수 `total`은 반환하지 않습니다.
- `has_analysis_result`는 분석 결과 행의 존재 여부입니다.
- 목록 항목의 `id`로 결과 조회 API를 호출합니다.

## 9. 촬영별 분석 결과지

### `GET /measurements/{measurement_id}/result`

결과가 아직 없어도 촬영 기록이 존재하면 `200`입니다.

```json
{
  "status": "not_available",
  "participant": {
    "id": "22222222-2222-4222-8222-222222222222",
    "name": "김예시",
    "gender": "F",
    "birth_date": "1950-03-12",
    "height_cm": "160.50",
    "created_by": "11111111-1111-4111-8111-111111111111"
  },
  "measurement": {
    "id": "44444444-4444-4444-8444-444444444444",
    "participant_id": "22222222-2222-4222-8222-222222222222",
    "age": 75,
    "extractor_name": "example-extractor",
    "extractor_version": "1.0",
    "measured_at": "2026-01-10T05:30:00Z",
    "created_by": "11111111-1111-4111-8111-111111111111"
  },
  "analysis_result": null
}
```

결과가 저장되어 있으면 `status`는 `"available"`이며 `analysis_result`는 다음 형태입니다.
아래는 **전체 응답 중 `analysis_result` 부분만 보여주는 가상 예시**입니다.

```json
{
  "id": "55555555-5555-4555-8555-555555555555",
  "measurement_id": "44444444-4444-4444-8444-444444444444",
  "gdi_score": 91.2,
  "stride_length_cm": 102.5,
  "cadence_spm": 98.0,
  "hip_graph_json": null,
  "knee_graph_json": null,
  "ankle_graph_json": null,
  "model_version": "example-model-v1",
  "analyzed_at": "2026-01-10T05:31:00Z"
}
```

현재 새 측정을 등록해도 분석 결과는 자동 생성되지 않습니다.
촬영당 분석 결과는 최대 1건이며 재분석 이력 선택 기능은 없습니다.

| 화면 항목 | 응답 필드 |
| --- | --- |
| 성명 | `participant.name` |
| 성별 | `participant.gender` |
| 촬영 당시 만 나이 | `measurement.age` |
| 현재 등록된 키 | `participant.height_cm` |
| 촬영 일시 | `measurement.measured_at` |
| 연구용 GDI 추정 점수 | `analysis_result.gdi_score` |
| 평균 보폭(cm) | `analysis_result.stride_length_cm` |
| 분당 걸음 수 | `analysis_result.cadence_spm` |
| 관절별 그래프 | 각 `*_graph_json` |

- `not_available`은 결과 없음이며 분석 중·실패를 구별하는 상태가 아닙니다.
- 지표가 `null`이면 해당 항목을 숨깁니다.
- `0`도 유효한 값일 수 있으므로 `if (value)` 대신 `value !== null`로 판단합니다.
- 그래프 JSON 세부 구조는 아직 미확정입니다.
- GDI는 임상 진단값이 아닌 연구용 추정값으로 표시합니다.
- 없는 촬영 ID는 `404`와 `"Measurement not found"`를 반환합니다.

## 10. 공통 오류 처리

| HTTP 코드 | 의미 | 프론트 처리 |
| --- | --- | --- |
| 401 | 로그인 실패, 인증 누락, 토큰 만료·오류 | 로그인 실패 안내 또는 재로그인 |
| 404 | 어르신·촬영 없음 | 대상 없음 안내 |
| 409 | 생년월일 미등록으로 측정 불가 | 정보 보완 필요 안내 |
| 422 | 필수값 누락, 형식·범위 오류 | 입력 항목 확인 안내 |

일반 오류의 `detail`은 문자열입니다.

```json
{ "detail": "Capture date must not precede birth date" }
```

입력 검증 오류의 `detail`은 배열일 수 있습니다.

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "gender"],
      "msg": "Field required",
      "input": {
        "name": "김예시",
        "birth_date": "1950-03-12",
        "height_cm": "160.50"
      }
    }
  ]
}
```

프론트 오류 처리에서는 `detail`이 문자열인지 배열인지 구분해야 합니다.
위 예시는 대표 형태이며 검증 오류 종류에 따라 추가 필드가 포함될 수 있습니다.

## 11. 현재 연동 범위와 남은 작업

현재 연동 가능: 로그인, 어르신 등록·조회·수정, 좌표 저장, 촬영 목록, 저장된 결과 조회.

추후 작업: AI 실행·결과 저장 연동, 좌표·그래프 세부 규격 확정,
분석 작업 상태 관리, 중복 제출 방지.

구현 참조:

- [인증 API](../backend/app/routers/auth.py)
- [어르신 API](../backend/app/routers/participants.py)
- [측정 API](../backend/app/routers/measurements.py)
- [어르신 요청·응답 스키마](../backend/app/schemas/participants.py)
- [측정 요청·응답 스키마](../backend/app/schemas/measurements.py)
