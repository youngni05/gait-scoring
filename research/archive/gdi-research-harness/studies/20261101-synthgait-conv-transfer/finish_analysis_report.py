"""Publish only an actually finished CPU verification; never repair source results."""
from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;ROOT=H.parents[1];O=H/'analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
j=read(O/'comparison.json')
assert j['qualified_full_verification_passed'] and not j['full_model_verification_pending']
assert j['fits']==50 and j['external_pretraining']==25 and j['final_cpu_replays']==10
assert j['max_model_replay_error']<1e-4 and j['max_external_replay_error']<1e-4
receipt=read(H/'checks/closeout-receipt.json');u=Path(receipt['archive']).parent/'performance-review-originals'
assert receipt['server_local_sha256_matched'] and receipt['counts']['main_fit_complete']==50
assert receipt['states']['33610']=={'state':'FAILED','exit_code':'1:0'}
curves=list((u/'runs/main').glob('*/fold*/*/curve.json'));assert len(curves)==50
j['epochs_verified']=sum(len(read(p)) for p in curves)
j['external_epochs_verified']=sum(len(read(p)) for p in (u/'runs/pretrain/main').glob('seed*/curve.json'))
assert len(list((u/'runs/pretrain/main').glob('seed*/complete.json')))==25
j['source_summary_failed']=True;j['summary_failure_reason']='reference_metrics.json missing after training'
save(O/'comparison.json',j)
m=j['models'];lines=[]
for key,label in [('existing_best','기존 최선 결합'),('ranksim','RankSim'),('scratch','동일 조건 scratch 대조군'),('synth','SynthGait 합성곱 전이')]:
 q=m[key];hi=q['subgroups']['person_mean90']
 lines.append(f"| {label} | {q['mae']:.6f} | {hi['mae']:.6f} | {hi['bias']:+.6f} | {q['bands']['10']['accuracy']*100:.2f}% | {q['bands']['20']['accuracy']*100:.2f}% |")
pair=j['paired'];p=pair['synth_vs_scratch'];b=pair['synth_vs_best'];s=m['synth'];c=m['scratch']
def ci(p,key='bootstrap95'):return f"{p[key][0]:+.3f}~{p[key][1]:+.3f}"
qualification='원본 엄격 로컬 감사도 통과했다。'
if not j['original_exact_local_audit_passed']:
 d=j['supplemental_diagnostics']
 qualification=f"원본 엄격 로컬 보정계수 일치(1e-10)는 실패했고 그 기록을 보존했다. 원서버에서 원본 감사 50개를 같은 기준으로 별도 실행한 증거와, 저장된 보정식의 모든 epoch 지표·로컬 재적합식의 실제 점수 차이 검사를 구분해 확인했다. 기준을 완화하거나 원본을 고쳐 통과 처리하지 않았다. 로컬 계수 최대 차이는 {d['coefficient_max_error']:.9g}, 학습/내부 검증 점수 차이는 {d['train_transformation_max_error']:.9g}/{d['validation_transformation_max_error']:.9g}다. 아래 결과는 이 근거를 합친 조건부 검산으로 표시한다."
text=f"""## 성능 검산·비교 — 2026-11-01

**18개 합성 움직임의 걸음빈도로 합성곱을 사전학습한 이번 조건에서는 명확한 GDI 개선을 확인하지 못했다.** 같은 조건의 대조군보다 전체 MAE가 {abs(p['delta']):.3f}점 작았지만 95% 차이 범위({ci(p)})에 0을 포함한다. 현재 최선 결합보다 전체 오차가 {b['delta']:.3f}점 크다. 단독 대체보다는 다른 모델과 오차가 다른지 확인할 후보로 보존한다.

같은 692명·1,061기록·사람 5분할·사람 동일 비중이다. 고득점 집단은 사람 평균 정답 GDI가 90 이상인 68명이며, 음수 편향은 과소예측이다.

| 모델 | 전체 MAE | 고득점 MAE | 고득점 편향 | 10점 정확도 | 20점 정확도 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

전이의 고득점 MAE 차이는 대조군 대비 {p['high90_delta']:+.3f}점(95% 범위 {ci(p,'high90_bootstrap95')}), 최선 결합 대비 {b['high90_delta']:+.3f}점({ci(b,'high90_bootstrap95')})이다. 작은 점 추정의 감소를 확실한 개선으로 단정하지 않는다. 재표집은 사람을 짝지어 10,000회, seed 20261101로 수행했다. 같은 경계의 구간 정확도이며, 넓은 구간 효과와 학습 개선을 혼동하지 않는다.

보정 전 전체/고득점 MAE는 scratch {c['raw']['mae']:.3f}/{c['raw']['subgroups']['person_mean90']['mae']:.3f}점, synth {s['raw']['mae']:.3f}/{s['raw']['subgroups']['person_mean90']['mae']:.3f}점이다. 보정 후를 주비교로 두었다. 사람 평균 100 이상, 개별 관측 90/100 이상, 낮은 점수대와 분할별 결과는 [동일 계산 결과](analysis/comparison.json)에 인원·편향·차이와 함께 보존했다.

작업 33610의 실제 종료는 FAILED 1:0이다. 50개 GDI 학습과 10개 바깥 예측을 저장한 뒤, 과거 기준 점수를 읽는 `reference_metrics.json` 누락으로 요약 생성이 실패했다. 원본·실패 기록은 보존했다. 서버와 로컬 SHA256을 대조한 회수본에서 별도 전체 예측을 구성했으며, 실패한 원본 요약을 정상 완료했다고 바꾸지 않았다.

{qualification}

모든 {j['epochs_verified']}개 GDI epoch, 외부 {j['external_epochs_verified']}개 epoch, 사람 분리·학습자료만의 정규화/보정·동일 초기화·합성곱 12텐서만 이전·BN/출력/최적화 초기화·내부 epoch 선택을 확인했다. 최종 GDI 10개 CPU64 재추론 최대 차이는 {j['max_model_replay_error']:.9g}점, 외부 25개 모델의 걸음빈도 재추론 최대 차이는 {j['max_external_replay_error']:.9g}회/분으로 기존 0.0001 기준 이내다.

외부 25개 학습은 같은 18개 움직임의 여러 초기값이며 사람 수 확대가 아니다. 이번 결과로 SynthGait 전체 또는 모든 사전학습이 무효라고 결론내리지 않는다. 입력 길이·시간축·보조 과제의 제약도 남는다. 보호 최종 평가자료나 LJMU는 사용하지 않았으며, 반복 개발자료 결과로서 성인·노인·임상 일반화 검증은 아니다.

다음에는 승인된 CPU 앙상블 비교에서 RankSim과 SynthGait의 추가 가치를 확인한다. 비중은 바깥 사람을 제외한 내부 예측에서만 정하고, 그 결과를 본 뒤 GPU 후속 실험을 정한다. 아직 새 GPU 학습을 제출하지 않았다.

"""
path=H/'README.md';old=path.read_text(encoding='utf-8-sig');first,rest=old.split('\n',1)
if '## 성능 검산·비교 — 2026-11-01' not in old:path.write_text(first+'\n\n'+text+rest.lstrip(),encoding='utf-8')
now=datetime.datetime.now().astimezone().isoformat();regpath=ROOT/'control/gpu-watch/registry.json';reg=read(regpath)
assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
row=next(x for x in reg['experiments'] if x['id']=='synthgait-conv-transfer-692')
row.update(review_state='reported',reported_digest=sha(O/'comparison.json'),performance_review_state='verified_summary_failure_preserved' if j['original_exact_local_audit_passed'] else 'qualified_verified_exact_local_coefficient_failure_preserved',performance_analysis_path=str(O/'comparison.json'),performance_reported_at=now)
reg['iterative_research'].update(phase='nested_cpu_ensemble',next_evidence='SynthGait full verification and matched comparison finished; test complementary errors with inner-only convex weights',synth_result_digest=row['reported_digest'])
for item in reg.get('post_training_analyses',[]):
 if item.get('id') in ['ranksim-best-ensemble-692','synth-best-ensemble-692']:item['state']='ready_for_protocol_and_execution'
save(regpath,reg)
pr=read(O/'process-receipt.json');pr.update(active=False,completed_at=now);save(O/'process-receipt.json',pr)
print(json.dumps(dict(report_sha256=row['reported_digest'],overall={k:q['mae'] for k,q in m.items()},paired=pair,full_verified=True),ensure_ascii=True))
