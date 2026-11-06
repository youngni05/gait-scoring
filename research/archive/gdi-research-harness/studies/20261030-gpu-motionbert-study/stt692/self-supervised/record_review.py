"""Publish small verified evidence summaries; preserve other experiment entries."""
from pathlib import Path
import datetime,json,hashlib
H=Path(__file__).resolve().parent;S=H.parent.parent;ROOT=H.parents[4]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
v=read(H/'analysis/verification.json');c=read(H/'analysis/comparison.json');t=read(H/'analysis/transfer_verification.json')
assert v['passed'] and read(H/'analysis/source_audit.json')['passed'] and c['verification_sha256']==sha(H/'analysis/verification.json')
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
prefix=f'''## 본 학습 완료·회수 검산 — {now}

학습 사람만의 가림 복원·미래 예측 사전학습 25개와 GDI 학습 25개가 완료됐다. 확보할당 32844의 자식 단계 32844.1은 COMPLETED, 종료코드 0:0이다. 부모 GPU 확보는 변경하지 않았다. 같은 692명·1,061기록·11,261구간, 사람 분할과 사람 동일 비중 평가를 유지했다.

| 비교 대상 | MAE |
|---|---:|
| 기존 최선: 집계 내부 선택＋평균·좌우 차이 결합 | 6.602250 |
| 기존 평균·좌우 차이 결합 | 6.605589 |
| 기존 CNN·STT 예측 결합 | 6.686592 |
| 여러 구간 CNN | 6.866655 |
| 여러 구간 STT | 6.944988 |
| 이번 사전학습 STT | {c['metrics']['SSL_STT']['mae']:.6f} |

이번 조건은 기존 STT보다 0.015051점, 최선 결합보다 0.357789점 오차가 크다. 원시 예측 MAE는 6.977045로 보정의 개선 폭도 약 0.017점이다. 같은 자료에서 복원·미래 예측을 먼저 학습한 이번 설정으로는 추가 개선이 확인되지 않았다. 사전학습 전체의 무효나 다른 자료·과제에서의 효과 부재로 확대하지 않는다. 한 초기값의 반복 개발자료 비교다.

서버/로컬 SHA256, 원본 환경의 변경하지 않은 audit_fit 25개, 로컬의 모든 4,629epoch 예측·보정·지표와 내부 epoch 선택, 학습 사람만의 사전학습 25개·저장 복원 출력 재추론, 최종 GDI 모델 5개 CPU 재추론을 검산했다. GDI 보정 점수 재추론 최대 차이는 {v['cpu_replay_max_error']:.10f}점으로 기존 0.0001 기준 이내다. CPU float64 참조 계산을 사용했으며 모델 가중치·허용오차는 변경하지 않았다. 원본 환경과 로컬 NumPy의 정확 동등 비교 실패, 이전 파일럿 float32 실패는 보존했다. 독립 수식 검산은 기존 1e-10 기준을 적용했다.

종료 epoch는 143·138·138·87·123이다. 내부 선택 시점 뒤에도 학습 오차는 감소했지만, 마지막 200epoch의 검증 오차는 다섯 분할 모두 증가했다. 사전학습만으로 과적합이 해소되지는 않았다. 고득점 68명 오차와 사람 단위 재표집 비교는 [비교 결과](analysis/comparison.json)에 기록했다.

큰 압축·체크포인트·대량 예측은 D의 대응 경로 `received-full-20261031`에, 작은 검산·비교·기록은 C에 저장했다. 보호된 최종 평가자료는 열지 않았으며 이 결과로 새 결합 비중을 고르지 않았다. [회수 지문](analysis/transfer_verification.json) · [전체 검산](analysis/verification.json) · [원본 환경 검산](analysis/source_audit.json).

'''
p=H/'README.md';old=p.read_text(encoding='utf-8');p.write_text(old.split('\n',1)[0]+'\n\n'+prefix+old.split('\n',1)[1],encoding='utf-8')
status=S/'STATUS.md';old=status.read_text(encoding='utf-8')
note=f'''## 사전학습 STT 검산 완료 — {now}

32844.1 정상 종료(0:0), 사전학습25개＋GDI25개·4,629epoch·바깥5개 완료. 전체 MAE6.960039, 원시6.977045로 기존 STT6.944988과 최선 결합6.602250보다 개선되지 않았다. D로 회수해 SHA256을 대조하고 원본 audit_fit·독립 epoch 검산·사전학습25개 출력과 최종5개 모델 CPU 재추론을 확인했다. [결과·검산](stt692/self-supervised/README.md). 기존 학습·GPU확보32844는 유지했다.

19:30 조회에서 작업32985는 대조군25/25, 기록 평균20/25 완료(합계45/50), 바깥9/10이다. 기록 평균은 fold4 내부 학습 중이므로 최종 성능을 비교하지 않았다. 조회 근거는 `control/gpu-watch/snapshot-20261031-192959.json`이며 이 진행률은 해당 시각의 값이다. 자동 감시는 유지한다.

'''
status.write_text(old.split('\n',1)[0]+'\n\n'+note+old.split('\n',1)[1],encoding='utf-8')
path=ROOT/'gdi-research-harness/control/gpu-watch/registry.json';reg=read(path)
assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
e=next(e for e in reg['experiments'] if e['id']=='stt-self-supervised-692')
e.update(review_state='reported',reported_digest=t['files']['runs/main-review.tar.gz'],reported_at=now,
    report_path=str(H/'analysis/comparison.json'),report_sha256=sha(H/'analysis/comparison.json'),
    result_note='MAE6.960038975864489 raw6.977044977556773;25SSL+25GDI,4629epochs,25SSL CPU probes+5GDI CPU64 replays verified. Source unchanged audits passed; local exact coefficient failures retained. No gain over STT6.944988 or best combination6.602250. Held32844 preserved;record CNN still running.')
path.write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(recorded=True,comparison_sha256=sha(H/'analysis/comparison.json'),at=now)))
