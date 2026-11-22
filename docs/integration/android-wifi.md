# 안드로이드에서 같은 Wi-Fi로 테스트

PC와 휴대폰을 같은 Wi-Fi에 연결한다. 기존 `.env`와 DB를 사용하므로 계정·대상자 기록은 PC와 같다.
Docker Desktop, Node.js, Git for Windows의 OpenSSL이 필요하다.

## PC에서 실행

Wi-Fi IPv4 주소를 `ipconfig`로 확인한 뒤 실행한다. 아래 주소는 설정 당시 PC 주소이며 바뀌면 변경한다.

```powershell
cd C:\happy-aging-app
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-phone.ps1 -Address 172.20.104.165
```

PC용 5173 개발 서버와 함께 실행할 수 있다. 휴대폰용은 빌드된 화면을 사용하므로 코드 수정 후에는
Ctrl+C로 종료하고 위 명령을 다시 실행한다. API/DB는 기존 start-dev.ps1의 영구 환경이다.

방화벽 설정은 **관리자 권한 PowerShell**에서 한 번 실행한다. IP가 바뀌면 다시 실행한다.

```powershell
cd C:\happy-aging-app
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\enable-phone-firewall.ps1 -Address 172.20.104.165
```

현재 PC 주소의 5443(HTTPS 앱), 5080(공개 인증서 다운로드)만 같은 서브넷에 허용한다.

## 휴대폰 인증서 설치 (최초 한 번)

1. Chrome에서 `http://172.20.104.165:5080/happy-aging-dev-ca.crt`를 열어 인증서를 다운로드한다.
2. 설정에서 `인증서 설치`를 검색하고 **CA 인증서** 설치를 선택한다. 제조사에 따라
   보안 및 개인정보 보호 → 기타 보안 설정 → 기기 저장공간에서 설치 아래에 있다.
3. 다운로드한 인증서를 선택하고 휴대폰에서 직접 설치를 완료한다.
   이름은 `Happy Aging Local Development CA`이며 이 PC에서 만든 개발용 CA다.
4. Chrome을 다시 열고 `https://172.20.104.165:5443`에 접속한다.
5. 기존 `worker01`과 설정한 비밀번호로 로그인한다.

인증서 경고를 무시해서 진행하지 않는다. 설치 후에도 경고가 나면 주소와 CA 설치 여부를 확인한다.
영상 처리에 쓰는 브라우저 기능은 보안 컨텍스트가 필요하므로 일반 LAN HTTP 주소로 대체하지 않는다.

인증서·키는 Git에서 제외되는 `local-test-data/phone-tls`에 보관한다.
휴대폰에 전달하는 것은 `ca.crt` 공개 인증서뿐이다. `ca.key`와 `server.key`는 공유하지 않는다.
CA는 1년, 서버 인증서는 90일 유효하며 재실행 시 같은 CA로 서버 인증서를 갱신한다.
CA가 만료되면 새 CA 발급과 휴대폰 재설치가 필요하다.

## 연결이 안 될 때 / 종료

- 먼저 인증서 다운로드 주소가 열리는지 확인한다. 안 열리면 Wi-Fi·IP·방화벽부터 확인한다.
- 학교·회사 Wi-Fi는 기기 간 접속을 차단할 수 있다. 이 경우 개인 공유기나 스마트폰 핫스팟에 PC를 연결해서 사용한다.
- 테스트가 끝나면 PC 터미널에서 Ctrl+C. API·DB까지 멈추려면 `docker compose stop`.
- 방화벽 허용을 해제하려면 관리자 PowerShell에서 `Disable-NetFirewallRule -DisplayName 'Happy Aging phone preview'`.
- 개발용 CA가 더 필요 없다면 휴대폰의 사용자 인증서 목록에서 해당 CA만 제거한다.

참고: [Android 인증서 설치](https://support.google.com/pixelphone/answer/2844832),
[보안 컨텍스트가 필요한 randomUUID](https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID).
