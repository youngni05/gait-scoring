# 보행 영상 개발용 앱 — 3단계

현재 PC 검증 결과와 상세 확인 순서는 [3단계 기록](../docs/mobile-pose/stage-3-verification.txt)을 참고하세요.
재생 중 관측한 실제 시각을 읽는 기술 검증이며, 전체 원본 프레임을 빠짐없이 추출하는 디코더가 아닙니다.

실제 계정/서버/분석 기능이 없는 시험 화면입니다. 개인정보를 입력하지 마세요.
대상 기기는 사용자 제공 정보 기준 Galaxy Wide5 / Android 13입니다.

저장소 루트에서 PowerShell로 실행합니다. npm 잠금 파일은 루트 하나만 사용합니다.

```powershell
npm ci
npm run dev:mobile -- --port 5173 --strictPort
```

PC에서 <http://127.0.0.1:5173/>을 엽니다. 이 주소는 휴대폰에서 직접 사용할 주소가 아닙니다.
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
앱 이름은 **보행 평가 개발용**, ID는 `com.gaitscoring.dev`, 버전은 `0.3.0` (3)입니다.
debug APK이며 배포용 서명 APK가 아닙니다. 자동 업데이트/서버/인증 기능이 없습니다.

## 직접 확인할 순서

| 순서 | 조작 | 기대 결과 |
|---|---|---|
| 1 | 앱 열기 | 개발 전용, Android 앱, 빌드 0.3.0 표시 |
| 2 | 촬영일을 비우고 입력 형식 확인 | 촬영일 입력 안내 |
| 3 | 키 99로 확인 | 100~220cm 오류 안내 |
| 4 | ID DEV-SUBJECT-001, 키 170, 시각 2026-11-16T10:00:00+09:00 입력 | 형식 확인 완료. 서버 성공으로 표시하지 않음 |
| 5 | 파일 선택 | 선택한 영상 미리보기, 크기·길이·방향 표시. 촬영일 입력 초기화 |
| 6 | MediaPipe 엔진 점검 | 초기화·합성 이미지 실행 성공, 보통 검출 0명 |
| 7 | 엔진 점검을 다시 실행 | 반복 실행 성공 또는 구체적 오류 표시 |
| 8 | 앱 완전히 종료 후 다시 열기 | 동일 화면 실행, 입력값 초기화 |
| 9 | 비행기 모드에서 엔진 점검 | 동봉 자산으로 실행. 실제 기기에서 확인 필요 |

3단계는 영상 미리보기와 프레임 캔버스 읽기를 수행합니다. 관절 추출은 하지 않습니다.
프레임 읽기 시작 → 중간 취소 → 다시 시작 → 다른 영상 선택 → 선택 해제를 시험하세요.
브라우저가 해석한 표시 방향/크기를 캔버스에도 사용하며 원본 회전 각도는 null(미확인)입니다.
촬영일 출처는 시험값이며 실제 영상 메타데이터를 읽었다는 뜻이 아닙니다.
빈 합성 이미지의 MediaPipe 성공은 실제 영상 속 관절 추출 성공을 뜻하지 않습니다.
실제 영상에 대한 MediaPipe 실행은 다음 4단계 범위입니다.

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
