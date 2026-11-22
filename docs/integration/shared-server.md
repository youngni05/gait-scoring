# 공용 서버 이전 절차

2026-11-22 main 통합: 웹·API·분석 워커·마이그레이션을 포함하도록 배포 초안을 갱신했다. GAIT_MODEL_DIR 모델 준비와 npm ci 및 npm run build:mobile 실행이 먼저 필요하다. 공용 주소는 앱 루트이며 외부 API 경로는 /api이다. 팀원 스크립트의 -ApiUrl에는 https://실제도메인/api를 지정한다.

상태: 배포 준비만 완료. 서버/도메인 미확보, 실제 백업/복원/외부 접속 미검증.
기존 로컬 DB와 미커밋 작업은 변경하지 않는다.

## 준비

Linux 서버에 Docker Compose와 Node.js를 준비하고 저장소를 복사한다.
API용 도메인의 DNS를 서버 IP로 연결한다. 인바운드는 HTTPS용 443,
인증서 발급/리다이렉트용 80, 관리자 IP에서의 SSH만 허용한다.
5432/5433/8000은 공개하지 않는다. compose.server.yaml은 독립 설정이다.
기존 compose.yaml과 합치지 않는다.
AI 서버 추론의 필요 자원은 담당자 커밋 확인 후 별도로 산정한다.

## 기존 PC 백업

기존 계정이 있는 PC에서 Docker Desktop과 기존 DB를 실행한다.
이전 중 앱의 등록/수정/측정 저장을 중단하고 최종 백업 이후에는 공용 DB로 전환할 때까지 쓰지 않는다.

```powershell
node scripts/backup-local-db.mjs
```

backups/*.dump는 Git 제외이며 계정의 비밀번호 해시와 대상자 데이터를 포함한다.
기존 DB는 읽기만 한다. 출력된 백업을 SSH/SCP로 새 서버의 비공개 경로에 전송한다.
전송 전후 SHA-256을 비교한다. 실패한 .partial 파일은 복원에 사용하지 않는다.
아카이브 목록 검사는 실제 복원 검증을 대신하지 않는다.

## 서버에서 최초 복원 (Linux Bash)

아래 도메인과 백업 경로는 실제 값으로 바꾼다. 새 서버의 비어 있는 DB에만 실행한다.
기존 .env를 복사하지 않고 서버 전용 .env.server를 만든다. 생성기는 기존 파일을 덮어쓰지 않는다.

```bash
node scripts/init-server-env.mjs api.your-domain.com
chmod 600 .env.server
# 이후 명령의 공통 함수. 같은 터미널에서 실행한다.
dc() { docker compose --env-file .env.server -f compose.server.yaml "$@"; }
dc up -d --wait db
```

복원 전에 빈 DB인지 확인한다. 출력이 0이 아니면 중단하고 기존 데이터의 출처를 확인한다.

```bash
dc exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('\''pg_catalog'\'', '\''information_schema'\'');"'
```

API를 시작하거나 마이그레이션을 적용하기 전에 전체 백업을 복원한다.
삭제/덮어쓰기 옵션은 사용하지 않으며 오류 시 전체 복원을 롤백한다.

```bash
dc exec -T db sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-acl --exit-on-error --single-transaction' < /private/path/gait-backup.dump
dc build api worker
dc run --rm api alembic upgrade head
dc up -d --wait
curl --fail https://api.your-domain.com/api/db-health
```

복원 전후 users/participants/gait_measurements/gait_analysis_results/measurement_requests
건수를 비교하고 worker01의 존재와 기존 비밀번호 로그인을 확인한다.
대상자와 과거 측정/결과를 화면에서 확인한 뒤 두 PC로 등록/조회 공유를 검증한다.
새 JWT 키로 인해 재로그인이 필요하지만 계정 비밀번호는 바뀌지 않는다.
검증 실패 시 팀원 전환을 멈추고 기존 PC의 데이터를 유지한다.
공용 DB에 신규 데이터가 생긴 뒤에는 이전 백업으로 무조건 되돌리지 않는다.

## 팀원 PC

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-shared.ps1 -ApiUrl https://api.your-domain.com/api
```

기존 start-dev.ps1은 로컬 서버용이므로 공용 접속 시 위 스크립트를 사용한다.
앱 화면은 http://127.0.0.1:5173 에서 열고 기존 계정으로 로그인한다.
별도 프론트 도메인이 생기면 .env.server의 CORS_ALLOW_ORIGINS에 정확한 origin을 추가하고 API를 재생성한다.
Android는 공용 VITE_API_BASE_URL을 지정해 빌드/동기화해야 한다.

## 운영 전 남은 작업

- 서버/도메인 생성, SSH 접속 및 방화벽 설정
- 실제 백업과 복원 리허설, 데이터/로그인 비교
- 예약 DB 백업과 서버 외부 비공개 보관소, 복구 검증
- 서버 재부팅 후 데이터 유지 및 외부 네트워크 접속 확인

공식 근거:
- https://caddyserver.com/docs/automatic-https
- https://www.postgresql.org/docs/17/app-pgdump.html
- https://www.postgresql.org/docs/17/app-pgrestore.html
