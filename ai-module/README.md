# 팀원 PC에서 보행 영상 분석하기

브라우저에서 MP4를 열고 MediaPipe Full로 관절을 추출한다. 같은 PC의 Python 서버가 좌표로 보행 구간, 무릎 움직임, 걸음 지표, 기존 GDI와 정상100 GDI를 계산한다. 대상자와 결과는 현재 브라우저에 저장한다. 팀 계정 및 공용 데이터베이스 연동은 별도 개발 작업이다.

## 최초 설치

Windows에서 Node.js 22.20 이상, Python 3.12, Git, Chrome 또는 Edge를 준비한다. 명령은 이 브랜치를 복제한 저장소의 루트에서 실행한다.

```powershell
npm.cmd ci
npm.cmd run build:mobile
py -3.12 -m venv .venv
$teamPython = '.\.venv\Scripts\python.exe'
& $teamPython -m pip install -r ai-module/requirements.txt
& $teamPython -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
$teamAssets = Join-Path $env:USERPROFILE 'gait-scoring-models'
& $teamPython ai-module/prepare_assets.py --assets $teamAssets
```

설치 도구는 MediaPipe 모델, 기존·정상100 GDI의 공개 가중치, 고정된 Stanford 전처리 원본을 내려받고 파일별 SHA-256을 확인한다. 가중치 압축에 출처와 이용 조건이 함께 들어 있다. [모델 릴리스](https://github.com/youngni05/gait-scoring/releases/tag/research-snapshot-20261106)

두 GDI의 결합 모델에는 원 제공자의 STT 코드가 필요하다. 이용 조건을 확인하고 아래 커밋을 별도로 받아 사용한다. [STT 원본](https://github.com/vinuni-vishc/transformer-gait-analysis/tree/549d58bc32449b24a0eae0c37ca573f9aa58a1dd)

```powershell
$teamStt = Join-Path $teamAssets 'transformer-gait-analysis'
git clone https://github.com/vinuni-vishc/transformer-gait-analysis.git $teamStt
git -C $teamStt checkout --detach 549d58bc32449b24a0eae0c37ca573f9aa58a1dd
```

## 앱 열기

```powershell
$teamPython = '.\.venv\Scripts\python.exe'
$teamAssets = Join-Path $env:USERPROFILE 'gait-scoring-models'
$teamStt = Join-Path $teamAssets 'transformer-gait-analysis'
& $teamPython ai-module/team_server.py --assets $teamAssets --stt-source $teamStt
```

서버를 켜 둔 상태에서 같은 PC의 [앱](http://127.0.0.1:8881/)을 연다. **이 PC에서 시작 → 대상자 등록 → 새 보행 영상 분석**으로 진행한다. 결과에서 두 GDI가 계산되려면 같은 직선 보행 구간에 연속 124프레임과 충분한 관절 좌표가 있어야 한다. 조건을 충족하지 못하면 결과에 측정 보류가 표시된다.

화면 개발 중에는 분석 서버를 켜 두고 다른 창에서 `npm.cmd run dev:mobile`을 실행한다. 콘솔에 표시된 개발 주소에서 분석 서버로 요청이 전달된다. 브라우저 저장 기록은 접속 주소와 브라우저 프로필마다 따로 저장된다.

서버 시작 시 파일이 없거나 SHA-256이 다르면 해당 경로를 표시한다. 모델 설치 경로와 STT 커밋을 확인한다. 앱이 연결되지 않으면 브라우저에서 [서버 상태](http://127.0.0.1:8881/health)를 확인한다.
