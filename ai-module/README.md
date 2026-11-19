# 팀 개발용 보행 분석 API 사용 안내

[AI module ZIP](./ai-module-integration.zip?raw=true)

2026-11-19 기준. `ai-module-integration.zip`은 프론트·백엔드 담당자가 영상 촬영 전에 합성 좌표로 분석 요청과 개인 이력 비교를 시험하는 묶음이다. ZIP 안의 Python 서버가 자신의 PC에서 응답한다. 기존·정상100 GDI 추론 코드도 들어 있지만 실제 점수 계산에는 별도 연구 코드·가중치·보정 파일이 필요하다. 이 묶음으로 시험하는 범위는 무릎·걸음 계산, GDI 계산 불가 처리, 합성 방문 이력 비교다.

## 1. 서버 실행

Windows에서 ZIP을 풀고 `server.py`와 이 안내가 있는 폴더에서 PowerShell을 연다. Python 3.12가 필요하다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py --port 8877
```

## 2. 두 요청으로 연결 확인

첫 창을 열어 둔 채 같은 폴더에서 두 번째 PowerShell을 연다.

```powershell
$ai = 'http://127.0.0.1:8877'
Invoke-RestMethod "$ai/health" | Select-Object schema,status
$analysis = Invoke-RestMethod -Method Post -Uri "$ai/v1/analyze" -ContentType 'application/json' -InFile fixtures/request-v2.json
$analysis | Select-Object schema,status,request_id
$analysis.segments[0] | Select-Object segment_id,direction,knee,gait,gdi
$history = Invoke-RestMethod -Method Post -Uri "$ai/v1/compare-history" -ContentType 'application/json' -InFile fixtures/history-synthetic.json
$history | Select-Object schema,status,request_id
```

상태 확인은 `ok`다. 첫 분석 예제는 합성 보행 한 구간을 명시해 `ai-analysis/2` 응답을 받는다. `segments[].knee`와 `segments[].gait`에서 관절·걸음 결과를 읽고, `segments[].gdi.models`에서 각 모델의 값 및 불가 사유를 확인한다. 요청에서 `segments` **필드를 생략**하면 AI가 보행 후보 구간과 진행 방향을 찾는다. 빈 배열을 보내면 입력 오류다. 두 번째 요청의 방문 날짜와 수치는 이력 비교 화면 개발용 합성 예시다.

## 3. 팀 앱에서 연결할 부분

| 담당 | 구현할 일 |
|---|---|
| 프론트 | 원본 영상의 프레임 시각, 프레임 번호, 33개 관절 좌표와 관측 여부, 영상 크기·방향, 실제 사용한 추출기와 추적 정보를 전달한다. 화면에는 해부학적 좌우와 단위를 함께 표시하고, 값이 null이면 계산 불가 사유를 표시한다. |
| 백엔드 | 로그인·접근 권한을 확인하고 측정 ID를 대상자·방문에 연결한다. 분석 요청에는 `schema`, `request_id`, `pose`, `provenance`를 넣는다. 응답과 촬영 시각·출처를 함께 저장하고, 이력 조회 때 현재 방문과 같은 사람의 과거 방문을 묶어 비교 API에 전달한다. 상세 규격은 ZIP의 `contract.json`, `pose-contract.json`, `history-contract.json`과 예제 JSON을 참조한다. |

팀 앱에서 보내는 `coords-1`과 AI가 받는 `measurement-pose/1`은 형식이 다르다. 예를 들어 전자의 프레임은 `t`와 `lm`이고 관절당 숫자가 4개다. AI 예제는 원본 시각을 밀리초로 전달하고 관절별 visibility와 presence를 구분한다. 실제 연결 전 추출 단계에서 누락 필드를 확보하고, 시각 단위와 좌우·회전 처리, 미검출 프레임 보존을 예제와 대조한다. 모르는 값을 임의의 1이나 0으로 채우면 품질 판단이 달라진다. [팀 앱 좌표 규격](https://github.com/youngni05/gait-scoring/blob/167f7203/contracts/coords-1/schema.json)

팀 앱 Compose의 API는 8000번, 이 ZIP의 AI 서버는 호스트 PC의 8877번에서 실행한다. 위 명령은 호스트에서 호출하는 절차다. 컨테이너 안의 `127.0.0.1`은 해당 컨테이너 자신을 가리킨다. 팀 Compose에 AI 서비스를 추가하고 내부 연결 주소를 설정하는 작업은 별도로 남아 있다. [현재 팀 Compose 구성](https://github.com/youngni05/gait-scoring/blob/167f7203/compose.yaml)

## 4. 성공 여부 판단

- 연결이 거부되면 첫 창의 서버와 포트 번호를 확인한다.
- HTTP 422이면 응답의 오류 내용을 보고 규격 버전·필수 필드·프레임 시각을 예제와 비교한다.
- HTTP 200의 `partial` 또는 `unavailable`은 서버가 계산 가능한 범위를 알려 준 정상 응답이다. 가능한 값은 그대로 표시하고 누락값은 0점과 구분한다.
- 실제 GDI까지 검증할 때는 전체 모델 실행 환경을 갖추고 고정 입력의 기존·정상100 좌우 결과를 연구 작업본과 대조한다. GitHub에는 연구용 가중치가 올라가 있다. 릴리스만 내려받아 이 ZIP에서 전체 추론을 재현하는 절차는 아직 검증하지 않았다. [모델 릴리스](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)
