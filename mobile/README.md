# 해피워크 · 현재 개발 앱

휴대폰의 브라우저에서 MP4를 열면 MediaPipe가 관절 좌표를 추출한다. 로그인한 앱이 좌표와 추출 조건을 전송하면 팀 API가 PostgreSQL에 촬영 한 건을 저장한다. Python 워커는 대기 중인 촬영을 읽어 분석하고 같은 촬영 ID에 상세 결과지를 저장한다. 접수 후 앱을 닫아도 다른 기기에서 팀 계정으로 다시 열 수 있다.

## 한 서버에서 실행하기

1. 저장소 루트에서 `npm ci`, `npm run build:mobile`을 실행한다.
2. 새 환경에서는 `.env.example`을 `.env`로 복사한다. PostgreSQL 비밀번호와 `DATABASE_URL`을 맞추고 임의로 생성한 32바이트 이상의 `JWT_SECRET_KEY`를 설정한다. 기존 DB를 쓰는 환경에서는 기존 접속값과 Compose 프로젝트 이름을 유지하고 마이그레이션 전 DB를 백업한다.
3. `GAIT_MODEL_DIR`에 [검증된 모델과 STT 소스를 준비한 폴더](../ai-module/README.md)를 지정한다. 이 폴더 아래에 `stanford/`, `gdi-frozen54-20261106/`, `gdi-normal100-paper75-20261106/`, `transformer-gait-analysis/`가 있어야 한다.
4. `docker compose up -d --build`를 실행한다. DB가 준비되면 `migrate`가 테이블을 갱신하고 API와 워커가 시작된다. `docker compose logs worker`에서 `Analysis worker ready`를 확인한다. 최초 계정은 `docker compose exec api python -m app.create_worker`로 생성한다.
5. 서버 PC에서 `http://127.0.0.1:5176/`을 열고 로그인한다. 화면 수정 중에는 `npm run dev:mobile`을 사용할 수 있다. 개발 서버도 같은 PC의 8000번 팀 API에 요청을 전달한다.

### 휴대폰 및 운영 배포

안드로이드 USB 시험은 USB 디버깅 승인 후 PC에서 `adb reverse tcp:5176 tcp:5176`을 실행하고 휴대폰 Chrome에서 `http://localhost:5176/`을 연다. 현장에서는 상시 가동하는 팀 서버 한 곳에 위 구성을 올리고, TLS 인증서를 설정한 HTTPS 프록시가 팀 도메인에서 웹 서비스(기본 127.0.0.1:5176)로 연결되도록 한다. 휴대폰과 PC 모두 그 주소를 연다. 웹은 같은 도메인의 `/api/` 요청을 내부 FastAPI로 전달한다. 서버 기기, 도메인, 인증서와 운영 계정은 배포 환경에서 준비한다.

PostgreSQL 파일은 Compose의 `postgres_data` 볼륨에 보관된다. 각 팀원이 자기 PC에서 별도로 띄우면 각자의 개발 DB가 생긴다. 운영 DB와 모델은 팀 서버에서 관리하고 DB·API 내부 포트는 외부에 공개할 필요가 없다. DB 백업은 별도 저장소에 보관하고 복원도 시험한다. 호스팅 업체의 PostgreSQL을 선택하면 API·워커·마이그레이션의 `DATABASE_URL`을 같은 DB로 설정하고 Compose의 로컬 DB 의존성을 배포 구성에서 조정한다.

### 기록과 캐시

촬영 ID는 좌표를 보낼 때 생성한다. 같은 전송을 재시도하면 서버가 기존 ID와 내용을 확인한다. 워커가 중단되면 처리 중이던 작업은 다시 대기 상태로 남는다. 분석 오류가 난 촬영은 결과 화면에서 같은 좌표로 다시 시도할 수 있다. 촬영일·촬영 당시 키가 알려지지 않은 경우 미확인으로 보존한다. 대상자 등록에는 팀 계정용 API가 요구하는 성별·생년월일·키를 입력한다.

브라우저는 앱 화면 파일을 캐시하고 탭의 로그인 상태와 화면 선택값을 보관한다. 새 측정 기록의 원본은 팀 DB에 있다. 과거에 각 브라우저의 IndexedDB에 저장한 기록은 자동 이관되지 않는다. [구버전 서버 연동 화면](../legacy-ui/)은 `npm run dev:legacy`로 연다.
