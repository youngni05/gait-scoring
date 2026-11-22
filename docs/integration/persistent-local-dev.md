# 같은 계정으로 PC에서 반복 테스트하기

스마트폰에서 Wi-Fi로 연결하려면 [안드로이드 Wi-Fi 실행 안내](android-wifi.md)를 따른다.

이 실행 방식은 프로젝트 `.env`와 Docker Compose의 `postgres_data` 볼륨을 계속 사용한다.
계정·대상자·측정 기록은 재시작해도 유지된다. 임시 DB를 만드는 자동 테스트 실행기와 별개다.

사전 준비: Docker Desktop, Node.js 22.20.0 이상과 npm이 필요하다.
새 PC에서는 [백엔드 최초 설정](../../backend/README.md#개발-환경-실행)에 따라 `.env`와 계정을 한 번 준비한다.
기존 PC의 `.env`를 예시 파일로 덮어쓰지 않는다. 계정과 DB 데이터는 GitHub에 포함되지 않는다.

## 매번 실행

1. Docker Desktop을 실행한다.
2. PowerShell에서 아래 명령을 실행하고 터미널을 켜 둔다.

```powershell
cd C:\happy-aging-app
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-dev.ps1
```

3. http://127.0.0.1:5173 에서 기존 계정과 비밀번호로 로그인한다.
   서버 주소는 실행 스크립트에서 http://127.0.0.1:8000 으로 설정하므로 화면에서 입력하지 않는다.

현재 이 PC의 영구 DB에서 확인된 기존 아이디는 **worker01**이다 (`work01` 아님).
비밀번호는 이전에 설정한 것을 사용한다. 스크립트는 계정을 만들거나 비밀번호를 바꾸지 않는다.
새로고침·재시작 후 재로그인은 필요하지만 계정을 다시 만들 필요는 없다.

비밀번호를 잊었다면 API 실행 상태에서 PowerShell로 다음 명령을 실행한다.

```powershell
docker compose exec api python -m app.reset_worker_password worker01
```

새 비밀번호(12자 이상)를 두 번 입력한다. 입력 중 문자가 보이지 않는 것이 정상이다.
이 명령은 해당 계정의 비밀번호만 바꾸며 대상자·측정 기록을 유지한다.
취소, 확인 불일치, 길이 미달이면 변경하지 않는다. 기존 발급 토큰은 만료 시까지 유효하다.

스크립트는 API 빌드/실행, DB 마이그레이션, 연결 확인 후 프론트를 실행한다.
이미 설치된 의존성을 재사용한다. package-lock.json 변경 후에는 `npm ci`를 한 번 실행한다.
임시 서버가 같은 포트를 사용 중이면 종료한 뒤 실행한다. 다른 포트로 자동 변경하지 않는다.

## 종료

프론트 터미널에서 Ctrl+C를 누르고, API·DB도 멈추려면 다음을 실행한다.

```powershell
docker compose stop
```

다음 실행도 같은 start-dev.ps1 명령을 사용한다.
DB 볼륨을 지우는 `docker compose down -v`는 사용하지 않는다.
프로젝트의 `.env`와 Docker 볼륨을 유지해야 같은 계정·데이터가 유지된다.

최근 `local-tester`로 사용한 임시 미리보기 데이터는 별도 DB에 있으며 자동으로 합치지 않는다.
AI 자동 분석은 아직 미연동이다. 좌표 추출·저장과 저장된 결과 조회를 테스트할 수 있다.
