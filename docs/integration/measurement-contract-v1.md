# 측정 저장 연결 계약 v1

2026-11-18 | 개발 제안 및 프론트 요청 변환 구현 | 실제 서버 지원 전 활성화 금지

기준 main: 2c1faab281d54b96608f0238e2125ddedd1e8e54 (PR #5 병합).
이 문서는 기존 docs/API_SPEC.md를 대체하지 않는다. 현재 서버 계약과 아래 확장 계약은 다르다. 백엔드 담당자가 구현·검증하고 API 명세에 반영한 후 실제 게이트웨이에 연결한다.

## 이번 변경과 다음 변경

이번 브랜치에는 네트워크를 사용하지 않는 prepareMeasurement(), 합성 요청 예시와 테스트만 추가한다. 기존 추출·모의 전송 동작은 유지한다. 실제 로그인, 토큰 저장, HTTP 호출, DB 마이그레이션, AI는 변경하지 않는다.

prepareMeasurement(request, participantId)는 고정된 method/path/headers/body를 생성한다. 입력 객체를 변경하지 않으며 결과와 헤더는 freeze되고 본문은 문자열 스냅샷이다. HTTP 구현은 인증 토큰을 별도로 붙여야 한다. 서버 접수 응답 검증 전 성공 receipt를 만들면 안 된다.

## 요청 매핑

POST /participants/{participant_id}/measurements
Content-Type: application/json
Authorization: Bearer <로그인 토큰>
Idempotency-Key: <requestId UUID>

| 필드 | 원천과 의미 |
|---|---|
| keypoints_json | coords-1 전체. 원래 PTS, null 프레임, 33관절을 보존 |
| extractor_name | 고정 mediapipe_tasks_vision |
| extractor_version | coords.pose.engine_version. MediaPipe 엔진 버전 |
| measured_at | coords.capture.recorded_at을 그대로 사용 |
| processing_metadata | 아래 processing-1 객체. 신규 서버 필드 |

pose.model/delegate는 coords 내부에 보존한다. 모델 파일 버전은 metadata.modelAssetVersion으로 별도 보존한다. 엔진 버전과 모델 자산 버전을 합치지 않는다.

processing_metadata는 schema_version='processing-1'과 기존 ProcessingMetadata의 recordedAtSource, rotationAppliedDegrees, detection, 선택 modelAssetVersion을 포함한다. 새로운 파일 경로·영상·사용자 이름·토큰 필드는 허용하지 않는다. 백엔드도 동일한 허용 필드와 타입, 유한한 수, detection 상태 및 marks 일치를 검증한다.

recordedAtSource=user_entered는 현재 날짜 입력에 따른 한국 00:00일 수 있다. metadata는 MP4 생성 시각 출처를 의미하며 실제 촬영 시각이라는 보증은 아니다. 화면은 출처를 구별하고 추후 정밀도 필드가 필요하면 별도 버전을 추가한다.

participant_id와 coords.subject.id는 같은 실제 어르신 UUID여야 한다. 개발용 문자열 ID는 보내지 않는다. requestId는 UUID를 사용한다. recording_id는 영상 선택 단위 ID이며 서버 measurement_id와 다르다.

age/created_by는 서버 계산 값이다. coords.subject.height_cm은 추출 시작 시의 키를 보존한다. 현재 프로필의 키를 조용히 덮어쓰지 않는다. 현재 앱 허용 범위(100~220cm) 밖이면 추출 불가 안내가 필요하다.

## 서버 저장과 호환성 제안

1. gait_measurements에 processing_metadata JSONB nullable 컬럼을 추가한다. 기존 데이터는 null 그대로 유지한다.
2. 기존 클라이언트의 4필드 요청은 기존 경로에서 계속 받을 수 있도록 하고 신규 계약 요청은 metadata와 Idempotency-Key를 함께 요구한다. 하나만 있는 요청은 422로 거부한다. 신규 어댑터는 항상 둘 다 보낸다.
3. 신규 요청에는 coords-1 스키마와 시간 증가, 대상자·시간·추출기 정보 일치를 서버에서도 검증한다. 프론트 검증은 서버 검증을 대신하지 않는다.
4. 대용량 metadata/좌표를 목록 응답에 추가하지 않는다. 요청 및 좌표 원문을 로그에 출력하지 않는다.
5. 성공 응답은 기존 MeasurementResponse를 유지한다. 응답 id를 receiptId로 매핑하되 UUID/participant_id/필수 필드를 검증한 후 environment='server'를 반환한다.

## 중복 제출 및 경쟁 요청

권장 저장 구조: measurement_requests(user_id, request_key, participant_id, body_sha256, measurement_id). (user_id, request_key)에 UNIQUE, measurement_id에는 FK를 둔다.
본문 지문은 수신된 UTF-8 원문 바이트 SHA-256이다. 따라서 JSON 키 순서·공백 변경도 다른 본문이다. 프론트는 준비된 body 문자열을 그대로 재사용한다. 프록시는 본문을 재직렬화하지 않아야 한다. 인증 토큰은 본문 지문에 포함하지 않는다.

- 처음 유효 요청: 측정과 요청 키를 하나의 트랜잭션으로 저장하고 201 반환.
- 같은 사용자/키/대상자/본문: 기존 measurement_id와 최초 응답을 200으로 반환. 다시 측정이나 분석을 생성하지 않는다.
- 같은 사용자/키지만 다른 대상자 또는 본문: 409 IDEMPOTENCY_CONFLICT.
- 동시 요청: DB 유일 제약으로 직렬화하고 기존 커밋을 조회한다. 프로세스 메모리 캐시만 사용하지 않는다.
- 선행 트랜잭션 대기 제한 초과: 409 REQUEST_IN_PROGRESS, Retry-After: 2. 앱은 같은 키로 재시도하고 자동 무한 재시도하지 않는다.
- 인증 실패·검증 실패는 측정이나 성공 요청 키를 저장하지 않는다.
- 통합 MVP에서는 성공 키를 측정의 수명 동안 유지한다. 이번 범위에는 측정 삭제가 없다. 향후 보존/삭제 정책 변경 시 별도 검토한다.
- 모든 재시도는 인증과 현재 사용자 접근 권한을 먼저 검증한다. 재로그인은 같은 계정이어야 한다.

동시성 테스트는 실제 PostgreSQL과 서로 다른 DB 세션으로 수행한다. 동일 키 동시 요청 2개가 1개 측정으로 귀결되는지 확인한다.

## 신규 오류 형식

기존 FastAPI 422 detail 배열과 기존 오류 문자열 처리는 유지한다. 신규 업무 오류는 다음 객체를 제안한다.

```json
{"detail":{"code":"IDEMPOTENCY_CONFLICT","message":"동일 요청 ID에 다른 데이터가 연결되어 있습니다."}}
```

409 PARTICIPANT_BIRTH_DATE_REQUIRED는 생년월일 보완을 요구한다. 409 IDEMPOTENCY_CONFLICT는 자동으로 새 키를 발급해 재전송하지 않는다. 사용자 확인 후 새 요청을 만든다. 409 REQUEST_IN_PROGRESS는 잠시 후 같은 요청 재시도를 허용한다. 프론트 오류 파서는 문자열·배열·신규 객체 모두 처리해야 한다.

## CORS와 실제 전송 활성화

백엔드 CORS allow_headers에 Idempotency-Key를 추가한다. 공용 테스트 API origin을 확정한 뒤 앱 CSP connect-src에 정확히 추가한다. 현재 CSP는 외부 API 연결을 허용하지 않는다.
서버 미지원 상태에서 processing_metadata를 지우고 성공하는 우회 구현은 금지한다. 백엔드 테스트 통과와 팀 테스트 서버 배포 SHA 확인 후 실제 UploadGateway를 연결한다.
요청 상한·프록시 시간 제한은 실기기 payload 측정 후 양측에서 일치시키며 이 브랜치에서는 임의 숫자를 확정하지 않는다.

## 합성 샘플과 수락 검증

contracts/measurement-1/synthetic.request.json은 기존 coords-1 합성 좌표를 사용하되 어르신 ID만 테스트 UUID로 바꾼 예시다. 실제 개인정보·토큰·영상은 없다. path/headers/body를 분리해 기록하며 이 전체 파일을 HTTP 본문으로 보내지 않는다. body만 JSON.stringify해서 전송하고 그 바이트를 재사용한다. 샘플 대상자는 테스트 DB에 별도로 준비해야 한다.

프론트 테스트: 매핑 보존, 입력 불변, 재시도 스냅샷, 대상자 오연결, 개발 ID·잘못된 요청 키·미래 촬영일·메타데이터 불일치 거부.
백엔드 인수 테스트: 정상/기존 요청 호환, 메타데이터 저장, 잘못된 좌표 거부, 순차·동시 중복, 충돌 코드, 재로그인 재전송, CORS 사전 요청, 결과 없는 상태 조회.
통합 완료 기준: 갤럭시에서 준비된 동일 본문을 재전송해도 DB 측정은 1건이며 이후 목록과 결과 없음 조회가 가능하다.

## 담당자 인계와 PM 결정

백엔드 담당자에게 이 계약과 샘플을 전달해 저장 확장·멱등성·검증 테스트를 구현하도록 한다. 구현 차이는 이 문서와 API_SPEC을 함께 갱신한다. 이번 에이전트는 메시지를 대신 발송하지 않는다.
PM은 디자인 2안 선택 및 공용 서버안/비용을 별도로 결정한다. GDI 입력 계약은 여전히 모델 담당자 확인이 필요하며 이 측정 저장 계약이 모델 호환성을 보증하지 않는다.
