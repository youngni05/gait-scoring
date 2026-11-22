# 팀 분석 모델 준비

현재 앱에서는 팀 서버의 Python 워커가 이 모듈을 호출한다. 앱·계정·DB 실행 방법은 [팀 앱 안내](../mobile/README.md)에 있다. 아래 설치 명령은 모델 파일과 소스 준비에 사용한다. Docker 워커에는 Python 의존성이 이미지에 포함된다.

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

## 모델 폴더 연결

준비한 `$teamAssets` 경로를 서버 `.env`의 `GAIT_MODEL_DIR`로 설정한다. 워커가 시작할 때 고정된 모델과 소스의 파일 해시를 검증한다.
