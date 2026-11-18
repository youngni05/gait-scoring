# PR 준비

- 제목: chore: 모바일 관절 추출과 백엔드 통합 기반 구성
- Head: integration/app-backend
- Base: main
- PM 확인 후 병합. 현재 원격 게시와 PR 생성은 GitHub 연결 쓰기 권한 부족(403)으로 미완료.

## 변경 내용

별도 브랜치에 있던 모바일 영상 처리·33관절 추출·coords-1 계약과 최신 main의 FastAPI/PostgreSQL 백엔드를 한 브랜치에 통합합니다. 양쪽 커밋 이력을 유지하며 기존 제품 로직은 유지합니다.

.gitignore의 중복 환경파일 규칙을 제거해 .env.example 예외를 유지하고, docs/integration/app-backend-baseline.md에 실제 API 연결을 위한 요청 변환·메타데이터 보존·중복 저장 방지 등의 후속 작업을 정리했습니다.

## 검증

- 기존 테스트: 통합 전후 모두 115/115 통과
- npm run typecheck: 통과
- npm run build:mobile: 모바일 타입 검사 및 Vite 빌드 통과
- Python compileall 및 git diff --check: 통과
- main 대비 backend 및 앱 브랜치 대비 mobile/contracts/tests 소스 동일 확인
- Git 객체 연결성 검사 통과

## 제한과 후속 확인

- 기존 설치 의존성을 사용했으며 이번에 npm ci 재설치하지 않았습니다.
- JS 청크 크기 및 MediaPipe import 혼재 경고가 있습니다.
- Docker/Python 백엔드 의존성이 없어 로컬 API/DB 실행은 검증하지 못했습니다. PR 게시 후 Backend CI를 확인해야 합니다.
- 실제 서버 전송, 갤럭시 실기기, APK 재빌드, 실제 GDI 연결은 후속 단계입니다.
- 미커밋 stage-9 문서 2개는 로컬에 보존하고 PR에서 제외했습니다.
- 배포 및 DB 마이그레이션은 실행하지 않았습니다. 롤백이 필요하면 앱 통합 변경을 되돌리며 DB downgrade는 필요하지 않습니다.
