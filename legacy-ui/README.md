# 구버전 서버 연동·영상 처리 시험 화면

저장소 루트에서 `npm run dev:legacy`를 실행한다. 기본 주소는 `http://127.0.0.1:5180/`이며, 첫 화면에서 FastAPI 서버와 연결할 수 있다. 같은 주소의 `/dev.html`과 `/*-qa.html`은 이전 영상 처리 시험용이다. [현재 앱](../mobile/)은 별도로 실행한다.

## 이전 영상 처리 시험과 Android 인계 기록

현재 통합 범위·실행법·담당자 계약은 [인계 가이드](../docs/mobile-pose/handoff.md), 실측 결과는 [7단계 기록](../docs/mobile-pose/stage-7-verification.txt)을 참고하세요.
현재 구현은 MP4/H.264 및 HEVC(H.265)를 오프라인 순차 디코딩합니다. 원본 패킷 목록과 모든 출력 프레임의 개수·PTS를 대조하고 일치할 때만 완료 처리합니다. 최신 결과는 [전체 프레임 검증 기록](../docs/mobile-pose/stage-3-offline-verification.txt)을 참고하세요.

실제 계정/서버/분석 기능이 없는 시험 화면입니다. 개인정보를 입력하지 마세요.
대상 기기는 사용자 제공 정보 기준 Galaxy Wide5 / Android 13입니다.

저장소 루트에서 PowerShell로 실행합니다. npm 잠금 파일은 루트 하나만 사용합니다.

```powershell
npm ci
npm run dev:legacy
```

PC에서 <http://127.0.0.1:5180/dev.html>을 엽니다.
종료 후 다시 시작하려면 위 개발 서버 명령을 실행합니다.

## 자동 검사와 Android 빌드

```powershell
npm run typecheck
npm run typecheck:mobile
npm test
npm run build:mobile
powershell -NoProfile -File scripts/build-android.ps1
```

빌드 스크립트는 JDK 21, 사용자 폴더의 Android SDK를 사용하고 전역 환경 변수를 변경하지 않습니다.
다른 위치에 설치했다면 `-SdkRoot '경로' -JdkRoot '경로'` 인수를 추가합니다.
한글 프로젝트 경로로 인한 AGP 사전 검사에는 `android.overridePathCheck=true`를 적용합니다.
Gradle이 실제 경로 오류를 내면 원인을 확인해야 하며 이 옵션 자체가 빌드 성공을 보증하지 않습니다.

필수 환경: Node 22.20.0 이상, JDK 21, SDK platform-tools, platforms;android-36,
build-tools;36.0.0. AGP가 사용하는 build-tools;35.0.0도 자동 설치되었습니다.
Android Studio는 이번 CLI 빌드에는 설치하지 않았습니다.
Capacitor 8.6.2 / React 19.3.1 / Vite 8.3.8 / TypeScript 7.0.2를 잠금 파일로 고정했습니다.
SDK compile/target 36, minSdk 24이며 Android 13은 API 33입니다. OS 지원 범위와 실제 기기 통과는 별개입니다.

## 갤럭시 연결·설치

1. 데이터 전송이 되는 USB 케이블로 PC와 연결하고 휴대폰 잠금을 해제합니다.
2. 설정 → 휴대전화 정보 → 소프트웨어 정보 → 빌드번호 7회 → 개발자 옵션 → USB 디버깅.
3. 휴대폰에 나타나는 이 PC의 디버깅 허용 창을 확인합니다. 본인 PC일 때 허용하세요.
4. 다음 명령의 목록에서 기기 상태가 `device`인지 확인합니다. `unauthorized`는 휴대폰 허용 대기입니다.
   목록이 비어 있으면 케이블/USB 모드/포트 및 필요한 경우 삼성 공식 USB 드라이버를 확인합니다.

```powershell
$adb = "$env:USERPROFILE/AppData/Local/Android/Sdk/platform-tools/adb.exe"
& $adb devices -l
& $adb install -r 'mobile/android/app/build/outputs/apk/debug/app-debug.apk'
& $adb shell am start -n com.gaitscoring.dev/.MainActivity
```

여러 기기가 연결되면 각 명령에 `-s 기기시리얼`을 넣습니다. 시리얼을 공개 문서나 Git에 기록하지 마세요.
앱 이름은 **보행 평가 개발용**, ID는 `com.gaitscoring.dev`, 버전은 `0.6.0` (8)입니다.
debug APK이며 배포용 서명 APK가 아닙니다. 자동 업데이트/서버/인증 기능이 없습니다.

## 직접 확인할 순서

| 순서 | 조작 | 기대 결과 |
|---|---|---|
| 1 | 앱 열기 | 개발 전용, Android 앱, 빌드 0.6.0 표시 |
| 2 | 촬영일을 비우고 입력 형식 확인 | 촬영일 입력 안내 |
| 3 | 키 99로 확인 | 100~220cm 오류 안내 |
| 4 | ID DEV-SUBJECT-001, 키 170, 시각 2026-11-16T10:00:00+09:00 입력 | 형식 확인 완료. 서버 성공으로 표시하지 않음 |
| 5 | 파일 선택 | 선택한 영상 미리보기, 크기·길이·방향 표시. 촬영일 입력 초기화 |
| 6 | MediaPipe 엔진 점검 | 초기화·합성 이미지 실행 성공, 보통 검출 0명 |
| 7 | 엔진 점검을 다시 실행 | 반복 실행 성공 또는 구체적 오류 표시 |
| 8 | 앱 완전히 종료 후 다시 열기 | 동일 화면 실행, 입력값 초기화 |
| 9 | 비행기 모드에서 엔진 점검 | 동봉 자산으로 실행. 실제 기기에서 확인 필요 |

4단계는 영상 전체 프레임에 full/CPU MediaPipe VIDEO 추출을 실행합니다. 미검출은 lm:null로 남깁니다.
프레임 읽기 시작 → 중간 취소 → 다시 시작 → 다른 영상 선택 → 선택 해제를 시험하세요.
파일의 표시 크기와 회전 메타데이터를 캔버스에 적용합니다. 회전 태그가 있는 실제 휴대폰 영상은 후속 검증 대상입니다.
촬영일 출처는 시험값이며 실제 영상 메타데이터를 읽었다는 뜻이 아닙니다.
빈 합성 이미지의 MediaPipe 성공은 실제 영상 속 관절 추출 성공을 뜻하지 않습니다.
관절 추출 시작 후 프레임 번호로 원본+관절 오버레이를 확인합니다. 추출 완료 후 서 있기 초기 규칙을 자동 적용합니다.

## 자산·권한·미정 사항

`prepare:assets`는 고정 URL의 full 모델을 최초에 내려받고 SHA-256을 검사합니다.
`dev:mobile`과 `build:mobile`은 이 작업을 자동으로 먼저 실행합니다.
준비 또는 체크섬 검사 실패 시 실행/빌드를 중단합니다. 최초 실행은 인터넷이 필요합니다.
WASM은 잠금된 npm 패키지에서 복사합니다. 모델/WASM은 APK에 동봉되며 원격 CDN을 실행 중 호출하지 않습니다.
자산은 Git 제외, 출처/버전/해시는 `model-assets.json`에 기록합니다. 재현에는 최초 다운로드가 필요합니다.
full/CPU는 작은 기술 시험용 후보이며 최종 모델·실행 방식 선정이 아닙니다.
앱에 카메라·마이크·광범위 저장소 권한을 추가하지 않았습니다. 파일은 시스템 선택창을 사용합니다.
Capacitor 기본 INTERNET 권한은 존재하나 실제 업로드 코드/주소/토큰은 없습니다.
원본 비전송의 실기기 네트워크 검증은 아직 별도 수행해야 합니다.

공식 참고: [Capacitor 환경](https://capacitorjs.com/docs/getting-started/environment-setup),
[Android 설정](https://capacitorjs.com/docs/updating/8-0), [Vite 환경](https://vite.dev/guide/).

개발용 독립 검증: `python scripts/generate-video-fixtures.py` 실행 후 `/offline-qa.html`에서 생성된 MP4와 expected.json 전체를 선택합니다. QA 페이지는 기본 프로덕션 빌드에 포함되지 않습니다.

HEVC 원본 337프레임 검증 결과: [HEVC 검증 기록](../docs/mobile-pose/stage-3-hevc-verification.txt). HEVC는 현재 브라우저/기기의 디코더 지원이 필요하며 미지원 환경은 명시적 오류로 안내합니다.

## PC의 4단계 확인

1. 로컬 영상을 선택하고 **관절 추출 시작**을 누릅니다. CPU 처리는 영상 재생보다 오래 걸릴 수 있습니다.
2. 완료 후 처리 수, 검출 수, 미검출 수를 확인합니다. 미검출도 원본 시각이 있는 프레임으로 보존됩니다.
3. 프레임 번호 1, 169, 337 등으로 이동하고 원본 시각과 오버레이를 확인합니다. 이전/다음과 오버레이 숨기기도 시험합니다.
4. 원본값 펼치기에서 각 검출 프레임의 33개 [x,y,z,visibility]를 확인합니다. 표시 visibility 기준은 원본값을 수정하지 않습니다.
5. 관절 처리 취소 후 재실행하고, 다른 영상으로 바꿀 때 기존 결과가 사라지는지 확인합니다.
6. **다인 후보 점검**은 별도 numPoses=2 실행입니다. 2인 후보가 나오면 실제 영상 확인이 필요하며, 0건도 단일 인물을 보증하지 않습니다.

GPU는 미제공/미검증이며 CPU를 명시적으로 사용합니다. full은 비교 기준 후보입니다.
실기기 성능·lite 비교·임상 정확도·GDI 호환성은 이번 PC 검사에 포함하지 않습니다.
개발용 `/pose-qa.html`은 원본 MP4와 독립 PyAV PTS JSON을 선택해 전체 시각·33관절·취소/재실행·로컬 요청을 검사합니다. APK에는 포함되지 않습니다.

## PC의 5단계 확인

원본을 선택해 관절 추출을 완료하면 **서 있기 구간 · 실험 규칙**이 나타납니다.
이 원본의 현재 결과는 미검출(시작null/끝null)입니다. 실제 정지 구간의 존재 여부와 같다는 뜻은 아닙니다.
검출 근거를 펼치고, 영상에서 직접 확인한 구간을 **사용자 수동 표시**에 입력해 비교하세요.
구간이 검출되는 다른 영상에서는 구간 시작/중간/끝 버튼으로 원본 프레임을 탐색할 수 있습니다.
규칙과 임계값은 검증 전 후보입니다. 원본 관절값은 변경하지 않으며 JSON 전송은 아직 없습니다.
[5단계 검증 기록·수동 비교 양식](../docs/mobile-pose/stage-5-verification.txt)을 참고하세요.

## PC의 6단계 확인

관절 추출 완료 후 시험 ID·키와 촬영일 적용 값을 확인하고 **JSON 생성·검증**을 누릅니다.
좌표는 최대5자리로직렬화하고 시각/null을보존합니다. 직접 입력과 영상 메타데이터 모두 없으면 생성에 실패합니다.
**개발용 JSON 다운로드**는coords-1본문만 저장합니다. 실제좌표JSON은Git에넣지마세요.
mock 시험상황으로 성공/시간초과/네트워크/인증오류를재현하고, 취소후동일ID로재시도할수있습니다.
mock은실제서버/인증/DB가없습니다. 실제HTTP연결은팀원계약에맞는transport주입이필요합니다.
촬영일출처·검출버전등은별도메타데이터이며coords-1필드를확장하지않았습니다.
[실제두영상검증·PC체크리스트](../docs/mobile-pose/stage-6-verification.txt)를참고하세요.
서 있기 수동확인·갤럭시검증은보류상태입니다.

## 7단계 PC 회귀

개발 전용 `/regression-qa.html`에서 원본 MP4와 독립 PyAV PTS JSON을 선택합니다. 전체 추출·JSON 검증, 17프레임 취소 후 전체 재시작, 오류·정리·mock 재시도를 검사하고 처리 시간을 표시합니다.
제품 버전은 0.6.0을 유지합니다. [실측 결과와 PC/갤럭시 체크리스트](../docs/mobile-pose/stage-7-verification.txt)를 참고하세요. 갤럭시 실기기와 서 있기 수동 정답 비교는 미실행입니다.

## 촬영일 입력 변경

촬영일은 년-월-일(YYYY-MM-DD)로 선택 입력합니다. 직접 입력한 날짜가 우선이며 한국 시간00:00으로 저장됩니다. 비워 두면 MP4 생성 시각 메타데이터를 한국 시간으로 변환하여 사용합니다. 파일 수정일은 사용하지 않으며 메타데이터도 없으면 날짜 입력이 필요합니다. 출처는 자동 결정되고 coords-1의 시간대 포함 timestamp 형식은 유지합니다.
