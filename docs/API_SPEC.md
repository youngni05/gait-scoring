# 해피에이징 프론트엔드 연동 API 명세

작성일: 2026-11-17 · 갱신일: 2026-11-20 · 현재 구현 기준

AI 분석 실행은 아직 연결되지 않았습니다. 예시 데이터는 모두 가상입니다.

## 1. 공통 규칙

| 항목 | 내용 |
| --- | --- |
| 로컬 개발 주소 | `http://localhost:8000` |
| Swagger | `http://localhost:8000/docs` |
| OpenAPI JSON | `http://localhost:8000/openapi.json` |
| 인증 | 로그인 외 아래 업무 API는 JWT Bearer 토큰 필수 |
| 요청 형식 | 로그인은 폼 데이터, 등록·수정 API는 JSON |
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
GET·POST·PATCH 및 Authorization·Content-Type·Idempotency-Key 헤더를 허용합니다. 쿠키 인증을 쓰지 않으므로
프론트 요청에 `credentials: "include"`를 지정하지 않습니다.
브라우저에서 재시도 간격을 읽을 수 있도록 `Retry-After` 응답 헤더를 노출합니다.

## 2. API 목록

| 메서드 | 경로 | 기능 | 성공 코드 |
| --- | --- | --- | --- |
| POST | `/auth/login` | 로그인 | 200 |
| GET | `/auth/me` | 로그인 사용자 조회 | 200 |
| POST | `/participants` | 어르신 등록 | 201 |
| GET | `/participants` | 어르신 전체 목록 | 200 |
| GET | `/participants/{participant_id}` | 어르신 상세 | 200 |
| PATCH | `/participants/{participant_id}` | 어르신 정보 일부 수정 | 200 |
| POST | `/participants/{participant_id}/measurements` | 촬영·좌표 기록 저장 | 최초 201, 신규 계약의 동일 요청 재전송 200 |
| GET | `/participants/{participant_id}/measurements` | 어르신별 촬영 목록 | 200 |
| GET | `/measurements/{measurement_id}/result` | 촬영별 결과지 | 200 |

화면 흐름: 로그인 → 어르신 목록·등록 → 어르신 선택 → 촬영 목록 → 촬영 선택 → 결과지.

새 촬영 등록 흐름: 앱에서 영상 선택 → 앱에서 관절 좌표 추출 → 좌표 JSON 전송 → 측정 저장.
원본 영상은 서버로 전송하지 않으며 앱과 서버에서 별도로 영구 저장하지 않습니다.

상태 확인용 `GET /health`와 `GET /db-health`는 인증 없이 사용할 수 있습니다.
`/health`는 `200 {"status":"ok"}`, `/db-health`는 연결 성공 시
`200 {"database":"connected"}`, DB 연결 실패 시 `503 {"detail":"Database unavailable"}`입니다.
이 검사는 AI 모델 준비 상태나 마이그레이션 적용 완료를 보장하지 않습니다.

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
현재 등록 스키마는 미정의 필드를 무시합니다. 수정(PATCH)·측정 등록의 미정의 필드 거부와 다르며,
추가 필드를 보내도 저장되지 않습니다. 등록 요청에는 위 네 필드만 보내세요.

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

### 7.1 기존 요청과 신규 요청의 차이

같은 API가 아래 두 요청을 받습니다. 신규 앱 연동에는 **processing-1 요청**을 사용합니다.

| 항목 | 기존 4필드 요청 | 신규 processing-1 요청 |
| --- | --- | --- |
| 본문 | 좌표·추출기 이름·버전·촬영 시각 | 기존 4필드 + `processing_metadata` 객체 |
| `Idempotency-Key` | 보내지 않음 | UUID 헤더 1개 필수 |
| 좌표 검증 | 비어 있지 않은 객체/배열, 유효한 JSON 숫자 | `coords-1` 구조·프레임·시간·대상자와 메타데이터 일치 검증 |
| 중복 저장 방지 | 없음. 반복 호출하면 별도 측정 생성 | 같은 사용자·키·대상자·본문 바이트의 성공 요청 재사용 |
| 성공 응답 | 저장마다 201 | 최초 201, 동일 성공 요청 재전송 200 |
| 생년월일 미등록 | 409, 문자열 `detail` | 409, `PARTICIPANT_BIRTH_DATE_REQUIRED` 객체 |
| 반환 필드 | `MeasurementResponse` | 동일한 `MeasurementResponse` |
| AI 실행 | 없음 | 없음 |

`processing_metadata`와 키는 함께 보내거나 함께 생략해야 합니다. metadata를 `null`로 보내는 것은 생략과 다릅니다(422).
신규 요청 실패 시 metadata나 키를 제거해 기존 요청으로 재전송하면 중복 방지가 사라지므로 그렇게 처리하지 마세요.

### 7.2 기존 클라이언트 호환 요청

다음은 **기존 요청 전용 예시**입니다. `Idempotency-Key` 헤더를 붙이지 않습니다.

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

**이 좌표 예시는 신규 coords-1 검증을 통과하는 예시가 아닙니다.** 신규 요청의 필드와 실행 예시는 [12절](#12-processing-1-측정-저장-계약)을 참고하세요. 두 요청 모두 AI 모델 입력 호환성을 보장하지 않습니다.

| 필드 | 필수 | 입력 규칙 |
| --- | --- | --- |
| `keypoints_json` | 예 | 비어 있지 않은 JSON 객체 또는 배열 |
| `extractor_name` | 예 | 앞뒤 공백 제거 후 1~100자 |
| `extractor_version` | 예 | 앞뒤 공백 제거 후 1~100자 |
| `measured_at` | 예 | 실제 촬영 일시, 시간대 필수, 미래 불가 |

- 좌표의 `NaN`, `Infinity`는 허용하지 않습니다. 신규 processing-1 요청은 coords-1 구조와 관절·프레임·대상자·시간 일치를 검증합니다. 기존 4필드 요청에는 일반 JSON 검증을 유지합니다.
- 촬영일은 생년월일보다 앞설 수 없습니다.
- 본문에 `age`, `created_by`, `participant_id` 등 정의되지 않은 필드를 넣으면 `422`입니다.
- 서버가 한국 날짜 기준으로 촬영 당시 만 나이를 계산합니다.
- 기존 4필드 요청은 중복 제출 방지가 없습니다. 신규 processing-1 요청은 processing_metadata와 Idempotency-Key를 함께 전송하며 아래 12절의 중복 방지 규칙을 따릅니다.

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

등록 응답·촬영 목록·결과 조회의 measurement에는 좌표와 processing_metadata 원문을 포함하지 않습니다.
시간은 같은 시점을 나타내는 시간대 포함 문자열로 처리합니다.

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
| 409 | 생년월일 미등록, 요청 키 충돌, 선행 요청 잠금 대기 초과 | 신규 요청은 `detail.code`별로 분기(12절) |
| 422 | 필수값 누락, 형식·범위 오류 | 입력 항목 확인 안내 |
| 503 | `/db-health`의 DB 연결 실패 | 서버·DB 상태 확인 |

기존 일반 오류의 `detail`은 문자열입니다. 신규 측정 업무 오류는 객체이며, 입력 검증 오류는 배열입니다.

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
      "msg": "Field required"
    }
  ]
}
```

프론트 오류 처리에서는 `detail`의 문자열·배열·업무 오류 객체(`code`, `message`)를 구분해야 합니다. 검증 오류에는 입력 원문을 반환하지 않습니다.
현재 `RequestValidationError` 처리기는 배열 항목에 `loc`, `type`, `msg`만 반환합니다.
모든 422가 배열인 것은 아닙니다. 촬영일이 생년월일보다 이른 경우는 문자열,
키 형식·짝 불일치·대상자 불일치는 업무 오류 객체가 될 수 있습니다.
여러 오류가 동시에 있는 요청에서 특정 오류의 검사 순서에 의존하지 마세요.

## 11. 현재 연동 범위와 남은 작업

현재 연동 가능: 로그인, 어르신 등록·조회·수정, 좌표 저장, 촬영 목록, 저장된 결과 조회.

추후 작업: AI 실행·결과 저장 연동, 모델 입력 변환·그래프 세부 규격 확정, 분석 작업 상태 관리.
신규 측정 요청의 메타데이터 저장·서버 좌표 검증·중복 제출 방지는 12절에 설명합니다.

구현 참조:

- [인증 API](../backend/app/routers/auth.py)
- [어르신 API](../backend/app/routers/participants.py)
- [측정 API](../backend/app/routers/measurements.py)
- [어르신 요청·응답 스키마](../backend/app/schemas/participants.py)
- [측정 요청·응답 스키마](../backend/app/schemas/measurements.py)

## 12. processing-1 측정 저장 계약

### 12.1 본문과 헤더

`POST /participants/{participant_id}/measurements`의 기존 4개 필드는 유지합니다.
신규 클라이언트는 `processing_metadata` 객체와 UUID `Idempotency-Key` 헤더를 함께 전송합니다.
하나만 보내거나 metadata를 명시적 null로 보내면 422입니다. 둘 다 없으면 기존 동작이며 중복 방지를 제공하지 않습니다.

배경과 합성 요청은 [연결 계약](integration/measurement-contract-v1.md),
[합성 샘플](../contracts/measurement-1/synthetic.request.json)을 참고합니다.
연결 계약은 단계별 개발 당시 기록도 포함하므로 현재 서버 동작은 이 명세와 구현을 기준으로 합니다.
본문에는 `keypoints_json` 전체, `extractor_name=mediapipe_tasks_vision`, 엔진 버전 `extractor_version`,
`measured_at`, 그리고 `schema_version=processing-1` 및 기존 추출 메타데이터를 가진 `processing_metadata`가 들어갑니다.
메타데이터와 좌표는 저장하지만 목록·결과 응답에 원문을 추가하지 않습니다.

| 필드 | 필수 | 서버 검증 |
| --- | --- | --- |
| `keypoints_json` | 예 | 아래 coords-1 객체 |
| `extractor_name` | 예 | `"mediapipe_tasks_vision"` |
| `extractor_version` | 예 | `keypoints_json.pose.engine_version`과 일치, 앞뒤 공백 제거 후 1~100자 |
| `measured_at` | 예 | 시간대 포함, 미래 불가. `capture.recorded_at`과 같은 시점 |
| `processing_metadata.schema_version` | 예 | `"processing-1"` |
| `processing_metadata.recordedAtSource` | 예 | `"metadata"` 또는 `"user_entered"` |
| `processing_metadata.rotationAppliedDegrees` | 예 | 유한한 JSON 숫자 또는 `null`. 필드 생략은 불가 |
| `processing_metadata.detection` | 예 | 아래 상태별 객체 |
| `processing_metadata.modelAssetVersion` | 아니요 | 생략 가능, 제공 시 공백만으로 구성되지 않은 문자열. 명시적 `null` 불가 |

`detection` 예시(두 가지 중 하나):

```json
{
  "status": "detected",
  "marks": {"stand_start": 0, "stand_end": 0.5},
  "version": "fixture-1"
}
```

```json
{
  "status": "not_detected",
  "marks": {"stand_start": null, "stand_end": null},
  "version": "fixture-1",
  "reason": "no_standing_interval"
}
```

`version`은 공백만으로 구성되지 않은 1~100자 문자열입니다. `reason`은 not_detected에서 필수인
비공백 문자열이며 detected에는 보내지 않습니다. `detection.marks`는 coords의 `marks`와 일치해야 합니다.
서 있기 미검출은 그 자체로 저장 실패가 아닙니다.

### 12.2 coords-1 필드와 검증 범위

| 필드 | 서버 검증 |
| --- | --- |
| `schema_version` | `"coords-1"` |
| `recording_id` | 하이픈을 포함한 UUID 문자열. 서버 측정 ID와 별개 |
| `subject.id` | 경로의 어르신 UUID를 표준 문자열로 표현한 값과 일치 |
| `subject.height_cm` | JSON 숫자, 100~220. 숫자 문자열 불가 |
| `capture.mode` | `"guided"` 또는 `"fixed"` |
| `capture.recorded_at` | 초까지 포함된 ISO 8601 문자열, `Z` 또는 `±HH:MM` 시간대 포함. 소수 초 허용 |
| `capture.width`, `capture.height` | 양의 정수 |
| `capture.orientation` | `"landscape"` 또는 `"portrait"`. 가로가 더 길면 landscape, 세로가 더 길면 portrait |
| `capture.device` | 생략 가능. 제공 시 비공백 문자열, 명시적 null 불가 |
| `pose.model` | `pose_landmarker_lite`, `pose_landmarker_full`, `pose_landmarker_heavy` 중 하나 |
| `pose.engine_version` | 비공백 문자열. 바깥 extractor_version과 일치 |
| `pose.delegate` | 생략 가능. 제공 시 `"CPU"` 또는 `"GPU"`, 명시적 null 불가 |
| `marks` | stand_start/end가 모두 숫자 또는 모두 null. 숫자이면 `0 <= start < end <= 마지막 t` |
| `frames` | 최소 60개. 각 프레임에 `t`, `lm` 필수 |
| `frames[].t` | 0 이상 유한한 숫자, 엄격 증가. 계약상 원본 영상 시각(초) |
| `frames[].lm` | 미검출이면 null, 아니면 정확히 33개 관절 |
| 관절 배열 | 각 관절은 정확히 4개의 유한한 숫자 `[x, y, z, visibility]`. visibility는 0~1 |

coords와 processing_metadata의 정의된 객체는 미정의 필드를 거부합니다. 엄격한 타입 검증을 사용하므로
숫자 대신 문자열이나 boolean을 보내지 마세요. `x/y/z`에는 현재 서버가 별도 범위를 제한하지 않습니다.
marks는 서 있기 구간이며 AI 보행 구간을 뜻하지 않습니다. 위 조건은 저장 검증이며 분석 가능 품질을 보장하지 않습니다.

어르신 등록의 `height_cm`은 0 초과 Decimal이고 숫자 문자열을 허용하지만,
coords의 키는 100~220 범위의 숫자입니다. coords의 키는 요청 당시 값을 보존하며 현재 프로필 키와
동일한지 검증하거나 프로필을 덮어쓰지 않습니다. `recording_id`를 중복 방지 키로 사용하지도 않습니다.
현재 이 스키마에는 프레임 수의 상한이 정의되어 있지 않습니다.

### 12.3 합성 요청 전송 예시

샘플 파일 전체는 HTTP 본문이 아닙니다. `{method, path, headers, body}` 중 **body만 직렬화**합니다.
샘플에는 토큰이 없으며, 아래 코드는 별도로 생성한 테스트 어르신의 ID와 로그인 토큰을 사용합니다.
실제 촬영 요청에는 샘플이 아닌 앱에서 추출한 좌표·메타데이터를 사용하세요.

```ts
// sample: contracts/measurement-1/synthetic.request.json을 읽은 객체
// participantId: 생년월일을 등록한 테스트 어르신 UUID
const body = structuredClone(sample.body);
body.keypoints_json.subject.id = participantId;
const requestKey = crypto.randomUUID();
const serializedBody = JSON.stringify(body); // 재시도에서도 그대로 보존
const path = `/participants/${participantId}/measurements`;

async function sendMeasurement(accessToken: string) {
  return fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${accessToken}`,
      "Content-Type": "application/json",
      "Idempotency-Key": requestKey,
    },
    body: serializedBody,
  });
}
```

테스트 어르신의 생년월일은 샘플 촬영일 이전이어야 합니다. 같은 요청 재시도 시 위 준비 코드를 다시 실행하지 말고
같은 키·경로·본문으로 `sendMeasurement`만 호출합니다. 성공 응답 필드는 7절과 같으며 최초 201, 재전송 200입니다.

### 12.4 응답·중복 방지와 재시도

| 상황 | 응답 |
| --- | --- |
| 최초 유효 요청 | 201, 기존 MeasurementResponse |
| 같은 사용자·키·대상자·본문 바이트 | 200, 기존 측정 ID와 응답 재사용 |
| 같은 사용자·키, 다른 본문 또는 대상자 (요청 검증을 통과한 경우) | 409 IDEMPOTENCY_CONFLICT |
| 선행 트랜잭션 잠금 대기 2초 초과 | 409 REQUEST_IN_PROGRESS, Retry-After: 2 |
| 생년월일 누락 (신규 요청) | 409 PARTICIPANT_BIRTH_DATE_REQUIRED |
| 경로와 coords.subject.id 불일치 | 422 PARTICIPANT_MISMATCH |
| 키·메타데이터 짝 불일치 | 422 INVALID_MEASUREMENT_CONTRACT |
| 잘못된 요청 키 또는 헤더 중복 | 422 INVALID_REQUEST_KEY |

업무 오류는 `{"detail":{"code":"...","message":"..."}}`입니다.
검증 오류 배열에는 loc/type/msg만 포함해 좌표·개인정보 입력 원문과 NaN을 반사하지 않습니다.
기존 409 생년월일 누락은 문자열 응답을 유지합니다.

예: 잠금 대기로 인한 일시적 실패는 다음과 같습니다.

```http
HTTP/1.1 409 Conflict
Retry-After: 2
Content-Type: application/json
```

```json
{"detail":{"code":"REQUEST_IN_PROGRESS","message":"Retry the same request shortly"}}
```

| 상황 | 클라이언트 처리 |
| --- | --- |
| 네트워크 오류·타임아웃·응답을 받기 전 취소 | 저장 여부를 알 수 없으므로 동일 사용자·키·경로·본문으로 재시도 |
| `REQUEST_IN_PROGRESS` | Retry-After 이후 동일 요청으로 제한된 횟수만 재시도 |
| `IDEMPOTENCY_CONFLICT` | 자동 재시도·자동 새 키 발급 중단. 기존 요청과 변경 내용을 확인 |
| `PARTICIPANT_BIRTH_DATE_REQUIRED` | 어르신 정보 보완 후 재시도 |
| 401 | 같은 계정으로 다시 로그인한 뒤 기존 요청 재시도 |
| 422 | 필드 또는 계약 오류 수정. 성공 저장 여부가 불명확한 원래 요청을 임의로 바꾸지 않음 |

요청 키는 하이픈을 포함한 UUID 문자열이며 헤더는 정확히 1개만 보냅니다. UUID 버전은 제한하지 않습니다.
다른 사용자 또는 새 키로 보내면 별도 요청이므로 동일 좌표라도 중복 생성될 수 있습니다.

키는 로그인 사용자별로 측정 수명 동안 보존합니다. 같은 키의 본문은 UTF-8 원문 SHA-256으로 비교하므로
공백·키 순서 변경도 충돌입니다. 앱은 준비한 본문 문자열을 그대로 재사용해야 합니다.
검증 실패는 키를 예약하지 않습니다. 네트워크 취소가 서버 저장 취소를 뜻하지 않습니다.
정상 재전송도 인증·입력 검증을 거칩니다. 이미 성공한 요청은 생년월일 등 프로필 변경으로
기존 측정의 age를 재계산하지 않고 저장된 측정을 반환합니다.
AI 실행·결과 자동 저장은 이 계약에 포함되지 않습니다.

브라우저에서 재시도 대기 시간을 읽을 수 있도록 CORS는 `Retry-After` 응답 헤더를 노출합니다.
