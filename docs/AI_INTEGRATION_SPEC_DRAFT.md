# 해피에이징 AI 연동 규격 초안

작성일: 2026-11-17 · 갱신일: 2026-11-20 · 문서 버전: 초안 v0.2

이 문서는 백엔드·앱·AI 담당자 간 협의를 위한 문서입니다.
**JSON 예시의 내부 구조와 분석 방법은 확정 규격이 아닙니다.** 예시 수치는 모두 가상입니다.
이 문서 작성으로 코드·DB·API 계약을 변경하지 않습니다.

## 2026-11-20 날짜 요구사항 변경

사용자 결정: **촬영 시각을 저장하지 않고 분석한 날짜를 저장한다.**

- 촬영일 선택·직접 입력 기능을 제공하지 않는다.
- 촬영 시각을 영상 메타데이터에서 추출하여 저장하는 요구사항도 폐기한다.
- 촬영 시각 메타데이터가 없어도 영상 분석을 진행할 수 있어야 한다.
- 기록 목록과 결과지는 촬영일 대신 분석 날짜를 표시한다.
- 영상 내부의 프레임 경과 시간은 좌표 처리에 필요하므로 유지한다. 실제 촬영 일시와 구분한다.

확정: **분석 시작일을 사용하며, 나이도 분석 시작일 기준 만 나이로 계산한다.**
현재 파이프라인에서는 사용자가 좌표 추출을 시작하는 순간을 `analysis_started_at`으로 기록한다.
시간대가 포함된 ISO 8601 값을 UTC로 저장하고 날짜 표시·만 나이 계산에는 한국 시간(Asia/Seoul)을 사용한다.
저장 재시도 시 최초 시작 시각을 유지하며 서버에서 계산한 나이는 기록에 보존한다.
결과의 `analyzed_at`은 별개다. AI 실행은 아직 미연동이며 저장만으로 분석 완료를 의미하지 않는다.
기존 촬영 기록은 원래 날짜·나이를 유지하고 기존 기록으로 구분한다.

아래 이전 JSON 예시는 변경 전 계약을 나타낸다. `measured_at`이나
`capture.recorded_at`에 분석 날짜를 대신 넣어 호환시키는 방식으로 적용하지 않는다.

## 1. 확정된 서비스 요구사항

| 항목 | 확정 내용 |
| --- | --- |
| 영상 처리 | 앱에서 영상을 선택하고 클라이언트 측에서 관절 좌표 추출 |
| 서버 전송 | 좌표 JSON만 전송하며 원본 영상은 전송하지 않음 |
| 영상 보관 | 앱과 서버에서 원본 영상을 별도로 영구 저장하지 않음 |
| 접근 권한 | 로그인한 복지사는 모든 어르신·측정·분석 결과에 접근 |
| 측정 관계 | 어르신 1명에게 여러 촬영 기록 연결 가능 |
| 분석 관계 | 촬영 1건당 최종 결과 최대 1건, 재분석 이력 관리 없음 |
| 추출기 정보 | `extractor_name`, `extractor_version` 저장 |
| 기록 날짜 | 촬영 시각은 저장하지 않으며 분석한 날짜 저장·표시 |
| 결과지 나이 | 한국 시간 기준 분석 시작일의 만 나이를 계산·보존 |
| 결과지 키 | 현재 어르신 정보에 등록된 키 표시 |
| 보폭 | `stride_length_cm`, 평균값, cm 단위 |
| 보행 지표 | GDI 추정 점수, 평균 보폭, 분당 걸음 수 |
| 관절 그래프 | 고관절·무릎·발목 그래프 |
| 부분 산출 실패 | 미산출 지표는 SQL NULL로 저장하고 화면에서 숨김 |
| 모델 식별 | 분석 결과에 `model_version` 저장 |
| GDI 의미 | 임상 진단값이 아닌 연구용 추정값 |

목표 흐름:

앱 좌표 추출 → 백엔드 측정 저장 → AI 분석 요청 → 결과 검증·저장 → 앱 결과 조회

## 2. 현재 구현된 범위

- `POST /participants/{participant_id}/measurements`: 신규 `processing-2` 요청은 좌표와 분석 시작 시각 저장. 촬영 시각은 전송·저장하지 않음.
- 서버가 생년월일과 한국 시간 기준 분석 시작일로 만 나이를 계산하여 `gait_measurements.age`에 저장.
- `GET /participants/{participant_id}/measurements`: 기록 목록 및 결과 존재 여부 조회.
- `GET /measurements/{measurement_id}/result`: 저장된 단일 결과 조회.
- DB의 `measurement_id` UNIQUE 제약으로 촬영당 결과 최대 1건 보장.
- 분석 지표와 그래프 컬럼은 NULL 허용.

**AI 호출과 분석 결과 자동 저장은 아직 구현되지 않았습니다.**
기존 4필드 요청은 일반 JSON만 검증합니다. 신규 processing-1 요청은 coords-1의 33관절·프레임 구조·시간 순서·대상자 및 메타데이터 일치를 검증합니다. 이것이 GDI 모델 입력 호환성이나 촬영 품질을 보증하지는 않습니다.
앱 API의 `available` / `not_available`은 결과 존재 여부이며 분석 진행 상태가 아닙니다.

현재 프론트 API 계약은 [API 명세](API_SPEC.md)를 참고합니다.

## 3. 우선 확인할 분석 가능 조건 — 협의 필요

좌표만으로 필요한 분석을 수행할 수 있는지 AI 담당자의 확인이 필요합니다.
특히 cm 단위 보폭과 관절 움직임 산출에 필요한 좌표계·스케일 정보·촬영 조건을 먼저 정의해야 합니다.

- 지원 추출기 및 버전, 필요한 관절 목록
- 2D 또는 3D 좌표, 좌표 단위와 축 방향
- 실제 길이 변환에 필요한 키·보정 정보 등 추가 입력
- 촬영 방향, 영상 길이, FPS, 해상도, 전신 노출 등 필수 조건
- 좌표 누락·낮은 신뢰도·복수 인물·가림 처리 기준

## 4. 앱 → 백엔드: 관절 좌표 입력

### 4.1 현재 구현된 요청 외부 구조

```http
POST /participants/{participant_id}/measurements
Authorization: Bearer <access_token>
Content-Type: application/json
```

| 필드 | 현재 규칙 |
| --- | --- |
| `keypoints_json` | 비어 있지 않은 JSON 객체 또는 배열, 세부 구조 미확정 |
| `extractor_name` | 필수 문자열, 공백 제거 후 1~100자 |
| `extractor_version` | 필수 문자열, 공백 제거 후 1~100자 |
| `measured_at` | 필수, 시간대 포함 실제 촬영 일시, 미래 시각 불가 |

`measurement_id`는 측정 저장 후 서버가 생성하여 응답의 `id`로 반환합니다.
`age`, `created_by`는 서버가 결정하며 요청 본문에 포함하지 않습니다.

### 4.2 좌표 JSON 내부 구조 제안 — 미확정

```json
{
  "keypoints_json": {
    "schema_version": "1.0",
    "coordinate_system": {
      "dimensions": 2,
      "space": "image_normalized",
      "unit": "normalized",
      "origin": "top_left",
      "x_direction": "right",
      "y_direction": "down"
    },
    "source": {
      "width_px": 1920,
      "height_px": 1080,
      "nominal_fps": 30,
      "duration_ms": 10000
    },
    "joint_names": [
      "left_hip", "left_knee", "left_ankle",
      "right_hip", "right_knee", "right_ankle"
    ],
    "frames": [
      {
        "frame_index": 0,
        "timestamp_ms": 0,
        "joints": [
          {"position": [0.41, 0.48], "confidence": 0.98},
          {"position": [0.42, 0.65], "confidence": 0.96},
          {"position": [0.43, 0.83], "confidence": 0.95},
          {"position": [0.56, 0.48], "confidence": 0.98},
          {"position": [0.57, 0.65], "confidence": 0.94},
          {"position": null, "confidence": 0.12}
        ]
      }
    ]
  },
  "extractor_name": "example-extractor",
  "extractor_version": "1.0",
  "measured_at": "2026-01-10T14:30:00+09:00"
}
```

예시의 6개 관절·1개 프레임은 설명용 축약이며 필수 관절 집합이나 최소 프레임 수를 뜻하지 않습니다.
`confidence` 예시도 추출기의 실제 의미와 범위를 확정한 것이 아닙니다.

| 제안 필드 | 의미 | 협의 사항 |
| --- | --- | --- |
| `schema_version` | 좌표 JSON 구조 버전 | 추출기 버전과 별도로 관리할지 |
| `coordinate_system` | 차원·좌표계·단위·축 방향 | AI가 요구하는 형식 |
| `source` | 영상 크기·FPS·길이 메타데이터 | 필수 항목과 확보 가능 여부 |
| `joint_names` | 관절 이름과 배열 순서 | 필수 관절, 명명 규칙 |
| `frames[].frame_index` | 원본 영상 프레임 번호 | 프레임 생략 시 번호 유지 여부 |
| `frames[].timestamp_ms` | 영상 시작 기준 경과 시간 | 원본 시간 또는 재샘플링 시간 |
| `frames[].joints` | 관절별 좌표 배열 | `joint_names` 순서 사용 여부 |
| `position` | 2D 또는 3D 위치 | 차원, 누락 시 null 표현 |
| `confidence` | 추출기의 관절 신뢰도 | 의미·범위·임계값 |

공통 규칙 제안:

- 좌우는 화면 기준이 아닌 대상자의 해부학적 좌우로 표현.
- 관절 배열의 개수와 순서를 모든 프레임에서 유지.
- 누락 좌표를 `[0, 0]`으로 대체하지 않고 `null`로 표현.
- 숫자에 NaN·Infinity를 사용하지 않음(현재 서버에서도 거부).
- 프레임 시각은 증가해야 하며 간격이 일정하다고 가정하지 않음.
- 실제 촬영 일시 `measured_at`과 영상 내 경과 시간 `timestamp_ms`를 구분.

## 5. 백엔드 → AI: 분석 요청 제안 — 미확정

앱 요청과 AI 요청을 분리하고, AI에는 분석에 필요한 데이터만 전달하는 안입니다.
성명·로그인 정보는 전달하지 않습니다.

```json
{
  "request_schema_version": "1.0",
  "measurement_id": "44444444-4444-4444-8444-444444444444",
  "extractor": {
    "name": "example-extractor",
    "version": "1.0"
  },
  "subject": {
    "age_at_capture": 75,
    "height_cm": 160.5
  },
  "keypoints": {
    "schema_version": "1.0",
    "coordinate_system": {},
    "source": {},
    "joint_names": [],
    "frames": []
  }
}
```

`keypoints` 내부는 지면상 생략한 것으로, 빈 좌표가 유효하다는 뜻이 아닙니다.
`subject.age_at_capture`는 AI 요청 필드 제안이며 DB 컬럼명은 기존 `age`를 유지합니다.

| 항목 | 제안 | 상태 |
| --- | --- | --- |
| `measurement_id` | 요청·응답 연결 및 중복 처리 식별자 | 제안 |
| `subject.age_at_capture` | 저장된 `gait_measurements.age` 사용 | AI 필요 여부 미확정 |
| `subject.height_cm` | 모델에 필요한 경우에만 전달 | 필요 여부 미확정 |
| 성별 | 모델에 필요한 경우 추가 | 미확정 |
| 모델 선택 | 서버 설정 또는 명시적 요청 값 | 미확정 |

AI가 키를 사용한다면 **분석에 실제 사용한 키를 별도로 보존할 필요가 있습니다.**
결과지의 현재 키와 분석 입력 당시 키는 달라질 수 있습니다.
현재 DB에는 분석 입력값 스냅샷이 없으므로 저장 위치와 시점을 별도로 설계해야 합니다.

이 JSON은 논리적 데이터 계약 초안입니다. Python 함수 호출, 별도 HTTP 서비스,
작업 큐 중 어떤 방식으로 실행할지는 확정되지 않았습니다.

## 6. AI → 백엔드: 분석 결과 제안

### 6.1 일부 지표만 산출된 경우의 응답 예시

```json
{
  "response_schema_version": "1.0",
  "measurement_id": "44444444-4444-4444-8444-444444444444",
  "status": "partial",
  "model_version": "gait-model-1.0",
  "gdi_score": 91.2,
  "stride_length_cm": 102.5,
  "cadence_spm": 98.0,
  "hip_graph_json": null,
  "knee_graph_json": null,
  "ankle_graph_json": null,
  "issues": [
    {"target": "hip_graph_json", "code": "INSUFFICIENT_VALID_KEYPOINTS"},
    {"target": "knee_graph_json", "code": "INSUFFICIENT_VALID_KEYPOINTS"},
    {"target": "ankle_graph_json", "code": "INSUFFICIENT_VALID_KEYPOINTS"}
  ]
}
```

### 6.2 확정 필드와 신규 제안 구분

| 필드 | 자료형·의미 | 확정·구현 상태 |
| --- | --- | --- |
| `measurement_id` | UUID 문자열 | DB 결과 FK 존재 |
| `model_version` | 모델 식별 버전 문자열 | DB 필수 컬럼 존재 |
| `gdi_score` | 숫자 또는 null | DB 컬럼 존재, 범위·산출법 미확정 |
| `stride_length_cm` | 평균 보폭(cm), 숫자 또는 null | 명칭·단위·평균값 확정, 집계법 미확정 |
| `cadence_spm` | 분당 걸음 수, 숫자 또는 null | DB 컬럼 존재, 산출 세부 기준 미확정 |
| 각 `*_graph_json` | 그래프 객체 또는 null 제안 | DB JSONB 컬럼 존재, 내부 구조 미확정 |
| `response_schema_version` | 응답 규격 버전 | 신규 제안, 별도 저장 필드 없음 |
| `status` | succeeded / partial / failed 제안 | 신규 제안, 별도 저장 필드 없음 |
| `issues` | 미산출 사유 목록 | 신규 제안, 별도 저장 필드 없음 |

`status`, `issues`, 규격 버전의 채택 여부와 저장 위치는 협의가 필요합니다.
AI의 처리 상태와 앱 API의 `available` / `not_available`은 구분합니다.

### 6.3 결과 수신 처리 제안

- 요청과 응답의 `measurement_id` 일치 여부 확인.
- 숫자·단위·NULL 처리 및 필수 메타데이터 검증 후 저장.
- 결과 UUID와 `analyzed_at`은 백엔드가 생성.
- 미산출 지표는 SQL NULL로 저장하며 0·빈 객체·빈 배열로 대체하지 않음.
- 동일 측정의 결과를 중복 생성하거나 임의로 덮어쓰지 않음.
- 실패 재시도 및 중복 응답 처리는 별도 합의.

모든 지표 실패 시 결과 행을 저장할지, 작업 실패만 기록할지는 아직 미확정입니다.
촬영당 최종 결과 1건이라는 요구사항과 실행 실패 재시도 정책은 구분하여 협의합니다.

## 7. 관절 그래프 JSON 제안 — 미확정

프론트가 같은 컴포넌트로 세 관절을 그릴 수 있도록 공통 구조를 제안합니다.
아래는 `knee_graph_json`의 도식용 예시입니다.

```json
{
  "schema_version": "1.0",
  "quantity": "flexion_extension_angle",
  "x_axis": {
    "name": "gait_cycle",
    "unit": "percent",
    "values": [0, 25, 50, 75, 100]
  },
  "y_axis": {
    "name": "angle",
    "unit": "degree"
  },
  "series": [
    {"side": "left", "values": [5.0, 15.0, 40.0, 20.0, 5.0]},
    {"side": "right", "values": [6.0, 16.0, 38.0, 21.0, 6.0]}
  ],
  "aggregation": "mean_cycle"
}
```

**X축·각도 정의·좌우·평균 처리 방식은 모두 미확정입니다.**

| 협의 항목 | 결정할 내용 |
| --- | --- |
| X축 | 경과 시간 또는 정규화된 보행 주기 0~100% |
| Y축 | 표시할 관절 움직임과 단위 |
| 각도 정의 | 기준 자세, 부호, 회전 평면 |
| 좌우 | 양쪽 필수 여부, 한쪽만 산출될 때의 처리 |
| 곡선 | 단일 주기, 평균 주기, 모든 주기 |
| 표본 수 | 고정 또는 가변 |
| 결측점 | null 허용 및 보간 여부 |
| 요약 정보 | 유효 주기 수·변동성 등 제공 여부 |

최소 검증으로 X값과 각 시리즈 값 개수 일치, 단위 명시, 좌우 식별,
유한한 숫자 사용을 제안합니다. 그래프 전체 산출이 불가능하면 null을 반환합니다.

## 8. 우선순위별 협의 목록

| 우선순위 | 질문 | 결정 목적 |
| --- | --- | --- |
| 1 | 어떤 추출기·버전·관절 세트를 지원하는가? | 앱 추출 방식 결정 |
| 1 | 2D·3D 중 무엇이 필요한가? 좌표계·단위는? | 입력 JSON 확정 |
| 1 | cm 단위 평균 보폭 계산에 어떤 스케일 정보가 필요한가? | 추가 입력 결정 |
| 1 | 촬영 방향·길이·FPS·전신 노출 등 필수 조건은? | 앱 안내와 분석 가능 판정 |
| 1 | 성별·촬영 당시 나이·키가 필요한가? | 입력 계약과 스냅샷 저장 결정 |
| 2 | 평균 보폭에 어떤 구간·좌우·주기를 포함하는가? | 지표 정의 통일 |
| 2 | GDI 추정 점수 범위와 산출 불가 조건은? | 검증·화면 설명 |
| 2 | 그래프 축·각도·좌우·집계 방식은? | 차트 구현 |
| 2 | 전체 지표 실패 시 무엇을 반환·저장하는가? | 실패 화면·DB 정책 |
| 2 | 분석 소요 시간과 호출 방식은? | 동기·비동기 및 상태 관리 |
| 3 | 최대 요청 크기·프레임 수·타임아웃은? | 자원 제한과 앱 처리 |
| 3 | 모델·규격 버전 호환성과 중복 응답 정책은? | 운영·재현성 |

## 9. AI 담당자에게 요청할 자료

- 지원 가능한 입력 규격과 촬영 조건.
- 지표별 산출 방법·단위·허용 범위·산출 불가 조건.
- 정상 분석, 부분 실패, 전체 실패의 입력·출력 샘플 각 1세트.
- 사용할 모델 버전 식별 방식과 예상 실행 시간.
- 필수 입력 메타데이터 및 추후 변경 가능성이 있는 항목.

협의 결과를 반영해 규격을 확정한 다음 백엔드 검증, AI 호출·결과 저장,
프론트 그래프 구현을 진행합니다. 이 초안의 제안 필드는 합의 전까지 고정 계약으로 사용하지 않습니다.
