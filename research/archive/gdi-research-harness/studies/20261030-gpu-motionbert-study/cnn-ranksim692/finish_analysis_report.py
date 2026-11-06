from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;S=H.parent;ROOT=S.parents[1];O=H/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
j=read(O/'comparison.json');assert j['qualified_full_verification_passed'] and j['fits']==50 and j['final_cpu_replays']==10
assert j['max_model_replay_error']<1e-4
models=j['models'];now=datetime.datetime.now().astimezone().isoformat();lines=[]
u=Path(read(H/'checks/closeout-receipt.json')['archive']).parent/'performance-review-originals'
peers=[read(u/'control'/('worker-%d.json'%rank)) for rank in range(3)]
assert {p['job_id'] for p in peers}=={'33581','33582','33583'} and len({p['uuid'] for p in peers})==3
assert all(p['phase']=='assigned_folds_complete' and p['exit_code']==0 for p in peers)
tasks=[tuple(task) for p in peers for task in p['assignments']]
assert len(tasks)==len(set(tasks))==10 and set(tasks)=={(v,f) for v in ['mse','ranksim'] for f in range(5)}
curves=list((u/'runs/main').glob('*/fold*/*/curve.json'));assert len(curves)==50
j['epochs_verified']=sum(len(read(p)) for p in curves);j['parallel_jobs_verified']=[p['job_id'] for p in peers]
save(O/'comparison.json',j)
for variant in ['mse','ranksim']:
 source_summary=read(u/'runs/main'/variant/'summary.json')
 assert source_summary['fits']==25 and sha(u/'runs/main'/variant/'oof_predictions.npz')==source_summary['oof_sha256']
 assert abs(source_summary['metrics'][0]['mae']-models[variant]['mae'])<1e-10
for key,label in [('existing_best','기존 최선 결합'),('multiwindow_CNN','기존 여러 구간 CNN'),('multiwindow_STT','기존 여러 구간 STT'),('mse','사람 비중 MSE 대조군'),('ranksim','MSE＋RankSim')]:
 m=models[key];hi=m['subgroups']['person_mean90'];lines.append(f"| {label} | {m['mae']:.6f} | {hi['mae']:.6f} | {hi['bias']:+.6f} | {m['bands']['10']['accuracy']*100:.2f}% | {m['bands']['20']['accuracy']*100:.2f}% |")
pair=j['paired'];r=models['ranksim'];m=models['mse'];best=models['existing_best'];p=pair['ranksim_vs_best'];control=pair['ranksim_vs_mse']
f=lambda x:f'{x:+.6f}'
qualified=''
if not j['original_exact_local_audit_passed']:
 d=j['supplemental_diagnostics']
 qualified=f"""로컬 원본 감사의 보정계수 일치 검사(허용오차 1e-10)는 실패했고 원래 실패 기록을 보존했다. 원본 코드나 기준을 바꿔 통과 처리하지 않았다. 원래 서버 환경에서 기존 worker의 summarize가 모든 학습 감사를 마친 뒤 요약을 생성했음을 코드 지문과 완료 표식으로 확인했다. 별도 보완 검사에서는 저장된 계수로 모든 epoch 지표를 재계산하고, 로컬 재적합식과 저장식의 실제 점수 차이를 원래 점수 허용오차 0.0001로 확인했다. 계수 최대 차이는 {d['coefficient_max_error']:.9g}, 학습 점수 차이는 {d['train_transformation_max_error']:.9g}, 내부 검증 점수 차이는 {d['validation_transformation_max_error']:.9g}다. 따라서 엄격한 로컬 계수 일치는 미통과이고, 나머지 근거를 합친 조건부 검증 결과로 구분한다.

"""
section='## 완료·회수와 성능 검산 — 2026-11-01\n\n'
section+='작업 33581·33582·33583의 정상 종료와 50개 학습·10개 바깥 예측·요약 완료를 확인했다. 모델·모든 epoch 예측·로그·코드·설정·라이선스를 D에 회수하고 서버/로컬 SHA256과 개별 파일 지문을 대조했다. 이번 사용자 추가 요청에 따라 기존 결과를 CPU에서 분석했으며, 새 학습이나 앙상블 비중 학습은 하지 않았다.\n\n'
section+='같은 692명·1,061기록·사람 5분할이며 각 사람에게 같은 평가 비중을 주었다. 고득점 집단은 모든 기록·좌우 정답의 사람 평균 GDI가 90 이상인 68명이다. 편향이 음수면 실제보다 낮게 예측한 것이다.\n\n'
section+='| 모델 | 전체 MAE | 고득점 68명 MAE | 고득점 편향 | 10점 정확도 | 20점 정확도 |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(lines)+'\n\n'
section+=f"RankSim의 전체 MAE 차이는 같은 MSE 대비 {f(control['delta'])}점(사람 짝지은 재표집 10,000회 95% 범위 {f(control['bootstrap95'][0])}~{f(control['bootstrap95'][1])}), 기존 최선 대비 {f(p['delta'])}점({f(p['bootstrap95'][0])}~{f(p['bootstrap95'][1])})이다. 고득점 MAE 차이는 기존 최선 대비 {f(p['high90_delta'])}점이며 95% 범위는 {f(p['high90_bootstrap95'][0])}~{f(p['high90_bootstrap95'][1])}점이다. 0을 포함하는 차이는 명확한 개선·악화로 단정하지 않는다.\n\n"
section+=f"보정 전 MSE의 전체/고득점 MAE는 {m['raw']['mae']:.6f}/{m['raw']['subgroups']['person_mean90']['mae']:.6f}점, RankSim은 {r['raw']['mae']:.6f}/{r['raw']['subgroups']['person_mean90']['mae']:.6f}점이다. 보정 후 결과를 주비교로 두고 원시 결과는 출력 분포와 보정 영향의 진단으로 구분했다.\n\n"
section+='MSE와 RankSim의 직접 비교는 같은 초기값·사람별 추출·기록 평균·학습자료 한정 OLS 조건에서 추가 순서 손실의 영향을 본다. 기존 최선은 CNN·STT·좌우 STT 결합과 내부 모델 평균을 사용하므로, 단독 RankSim과의 비교는 현재 시스템을 대체할 만한 성능인가에 대한 비교다. 전체 오차와 고득점 오차를 구분해 판단하며 한 지표의 변화만으로 둘 다 개선됐다고 표현하지 않는다.\n\n'
section+=qualified
section+=f"최종 모델 10개 CPU float64 재추론의 최대 차이는 {j['max_model_replay_error']:.9g}로 기존 0.0001 허용오차 이내다. 모든 학습의 사람 분리·학습자료 정규화·균등 추출·저장 계수 지표·내부 epoch 선택·전체 OOF의 행 연결을 검산했다. 보호된 최종 평가자료와 LJMU는 사용하지 않았으며, 반복 개발자료 결과로서 임상 또는 외부 일반화 검증은 아니다.\n\n"
section+='[성능과 불확실성](analysis/comparison.json) · [검산 진행·진단](analysis/progress.json) · [회수 근거](checks/closeout-receipt.json). SynthGait 전이 실험은 별도 학습이며, 미완료 결과를 이번 표에 섞지 않았다.\n\n'
path=H/'README.md';text=path.read_text(encoding='utf-8-sig');first,rest=text.split('\n',1)
if section.split('\n')[0] not in text:path.write_text(first+'\n\n'+section+rest.lstrip(),encoding='utf-8')
status=S/'STATUS.md';text=status.read_text(encoding='utf-8-sig');first,rest=text.split('\n',1);heading='## RankSim 완료·성능 검산 — 2026-11-01'
if heading not in text:
 note=f"RankSim 실험 50개 학습·10개 바깥 평가를 완료하고 자료를 D에 회수했다. 같은 사람 비중에서 MSE의 전체/고득점68명 MAE는 {m['mae']:.6f}/{m['subgroups']['person_mean90']['mae']:.6f}점, RankSim은 {r['mae']:.6f}/{r['subgroups']['person_mean90']['mae']:.6f}점, 기존 최선은 {best['mae']:.6f}/{best['subgroups']['person_mean90']['mae']:.6f}점이다. 10개 최종 모델 CPU64 재추론을 검산했다. "
 note+=('로컬 원본 엄격 계수 일치 검사는 실패했으며, 원환경 집계 감사·별도 저장 계수 지표·재적합식 점수 차이 검사를 구분해 보완했다. ' if not j['original_exact_local_audit_passed'] else '원본 전체 감사도 통과했다. ')
 note+='[상세 결과](cnn-ranksim692/README.md)에 불확실성과 원래 실패 기록을 함께 남겼다. 이번 사용자 요청의 일회성 분석만 수행했으며, 새 학습·앙상블은 시작하지 않았다. 미완료 SynthGait 학습과 회수 예약은 유지한다.\n\n'
 status.write_text(first+'\n\n'+heading+'\n\n'+note+rest.lstrip(),encoding='utf-8')
registry=ROOT/'control/gpu-watch/registry.json';reg=read(registry);assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
e=next(x for x in reg['experiments'] if x['id']=='cnn-ranksim-692')
assert e['retrieval_state']=='received_sha256_verified'
e.update(retrieval_reported=True,retrieval_reported_at=now,retrieval_reported_digest=e['retrieval_receipt_sha256'],
 review_state='reported',reported_digest=sha(O/'comparison.json'),performance_review_state='qualified_verified_exact_local_coefficient_failure_preserved' if qualified else 'verified',
 performance_analysis_path=str(O/'comparison.json'),performance_reported_at=now)
for a in reg.get('one_off_analysis_authorizations',[]):
 if a.get('id')=='ranksim-synth-completed-performance-20261101':a.update(state='ranksim_reported_synth_still_running',ranksim_report_sha256=e['reported_digest'])
save(registry,reg)
print(json.dumps(dict(report_sha256=e['reported_digest'],models=models,paired=pair,original_exact_local_audit_passed=j['original_exact_local_audit_passed'],max_model_replay_error=j['max_model_replay_error'])))
