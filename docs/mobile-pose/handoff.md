# 앱·서버·분석 담당자 인계

현재 범위는 로컬 영상 → 전체 프레임 → 33관절 → 실험적 서 있기 → coords-1 JSON/mock이다. 실제 인증 서버, 실제 대상자 등록, 분석 점수/GDI는 미구현이다. 개발 화면의 guided 값은 촬영 프로토콜 준수의 증명이 아니다.

## 실행과 재현

저장소 루트에서 Node22.20.0 이상/npm10.9.3으로 실행한다.

```powershell
npm ci
npm test
npm run typecheck:mobile
npm run build:mobile
npm run dev:mobile -- --port 5173 --strictPort
```

브라우저 주소는 http://127.0.0.1:5173/ 이다. 최초 실행/빌드의 prepare:assets는 Google 공식 full 모델을 내려받아 고정 SHA256을 확인하고 npm 패키지의 WASM을 복사한다. 설치/최초 자산 준비에는 네트워크가 필요하다. 자산이 준비된 뒤 영상 처리는 로컬에서 수행한다.
Android는 JDK21/SDK36을 설치한 후 `powershell -NoProfile -File scripts/build-android.ps1`을 실행한다. 사용자별 경로는 `-SdkRoot`/`-JdkRoot`로 지정한다. APK 빌드 성공은 실기기 검증이 아니다. 상세 환경은 [mobile README](../../mobile/README.md)를 따른다.

## 앱 담당자: 다른 화면에서 호출

타입 검사 대상인 [prepareRecording 예시](../../mobile/src/pose/prepare-recording.ts)는 자원 정리까지 포함한다. 웹 DOM/Canvas 및 기기의 H.264/HEVC 디코더가 필요하다. 녹화 ID는 영상 선택 단위, 요청 ID는 고정된 전달 본문 단위로 생성/유지한다.

```ts
import {prepareRecording} from './pose/prepare-recording';
import {MockUploadGateway} from './pose/upload-gateway';
const controller = new AbortController();
const request = await prepareRecording(file, previewMount, {
  subjectId: 'DEV-EXAMPLE', heightCm: 170, day: '',
  recordingId: crypto.randomUUID(), requestId: crypto.randomUUID(),
}, controller.signal);
const gateway = new MockUploadGateway(() => 'success');
const receipt = await gateway.submit(request, controller.signal); // mock only
// 실패 후 같은 request/gateway로 재시도. 취소된 signal은 새 AbortController로 교체.
// 화면 이탈/영상 변경 시 controller.abort(); 이전 결과는 폐기.
```

실제 subjectId는 대상자 등록 화면이 제공해야 한다. 예시 값으로 실제 대상자 연결을 만들지 않는다.
촬영일은 입력 YYYY-MM-DD가 우선이며 한국00:00을 저장한다. 빈 값이면 MP4 생성 시각 메타데이터를 한국시간으로 변환한다. 메타데이터도 없으면 사용자 입력이 필요하다. 파일 수정일/파일명은 사용하지 않는다. 편집된 파일의 생성 시각이 실제 촬영일과 다를 수 있으므로 화면에서 확인한다.

## 서버 담당자: 아직 내부 포트인 계약

[UploadGateway](../../mobile/src/pose/ports.ts)의 입력은 `{coords, metadata, requestId}`, 응답은 `{receiptId, environment: 'mock' | 'server'}`이다. `submit(request, signal)`은 Promise이며 AbortSignal을 전달받는다.
- coords: [coords-1 JSON Schema](../../contracts/coords-1/schema.json)의 본문만. 추가 필드 금지.
- metadata: recordedAtSource, rotationAppliedDegrees, detection(status/marks/version/reason), modelAssetVersion. coords 안에 합치지 않는다.
- requestId: 동일 본문 재시도 시 유지. recording_id도 유지한다. 본문을 변경하면 새 요청 ID를 발급한다.
- MockUploadGateway: 인스턴스 메모리에서만 중복을 방지한다. 실제 DB/인증/영속적 멱등성 보장은 아니다.
- InjectedUploadGateway: 합의한 transport를 명시적으로 주입한다. 기본 URL/토큰/자동 전송 없음.

미정: HTTP URL·메서드·envelope, 인증/토큰 갱신, requestId 헤더/본문 위치, 서버 멱등성·중복 충돌/보존기간, 오류 상태/응답 schema, 용량/시간 제한, 업로드 완료와 분석 완료의 구분. 이 포트를 확정된 서버 API로 간주하지 않는다. 실제 endpoint 사용 전 CSP connect-src 허용 범위를 검토해야 한다. 현재 SDK 외부 통신 시도는 CSP로 차단한다.

## 분석 담당자: coords-1

[합성 샘플](../../contracts/coords-1/synthetic.sample.json)은 실제 인물/영상과 무관하다. 입력 경계에서 [validateCoords](../../contracts/coords-1/validate.ts)를 실행한다.
- frames는 최소60개. t는 원본 영상 PTS(초), 엄격 증가. 평균FPS로 재생성하거나 반올림하지 않는다.
- lm은 null 또는33개의 `[x,y,z,visibility]`. 미검출 프레임도 t와 함께 남긴다.
- x/y는 회전 적용된 표시 캔버스 기준 정규화 좌표이며 범위를 벗어날 수 있다. z는 상대 깊이이며 미터가 아니다. visibility는0~1.
- JSON 좌표만 소수 최대5자리로 직렬화한다. 추론 원본값에는 반전·보간·평활화·클램프를 적용하지 않는다.
- marks는 stand_start/end가 모두 숫자 또는 모두null. 실패 시 처음2초 등 대체 구간을 만들지 않는다.
- subject.height_cm는100~220; capture는 표시 해상도·방향·시간대 포함 recorded_at. 날짜만 직접 입력한00:00을 실제 측정 시각으로 오해하지 않는다.
- pose 모델/엔진/delegate를 기록한다. 현재 full/CPU는 기준 후보이며 최종 실기기 선정이 아니다.

## 실패 사례와 검증 한계

- 이전 영상 A-337:337프레임 보존, 관절 미검출20개는 lm:null로 유지.
- 현재 B-758:758/758, 미검출0. 17프레임 취소 후 재시작 결과 전체PTS/원본좌표 동일. 처리119.28초/111.36초, 시험JSON876399bytes. PC 개발환경 측정이며 기종 일반화 불가.
- 두 영상의 서 있기 marks는null. 수동 정답 미확인이라 위양성/위음성 판정 불가.
- mock 시간초과/네트워크/인증 오류와 취소/재시도 검증. 실제 네트워크 단절·서버 종단 검증과 다르다.

[7단계 기록](stage-7-verification.txt)에 실측 환경·갤럭시 결과표가 있다. Galaxy Wide5/Android13 실기기, 잠금/앱 전환/권한/발열/메모리, 서 있기 수동 정답 비교는 모두 보류. 임상 정확도/GDI 호환성/점수 산출은 완료되지 않았다.

개발 QA HTML들은 production 빌드 진입점에 포함되지 않는다. 단, production index 자체도 개발용 시험 앱이며 mock이 남아 있다. 임상/운영 배포 앱으로 취급하지 않는다. 실제 영상·좌표·개인 환경 파일은 Git에 넣지 않는다.
