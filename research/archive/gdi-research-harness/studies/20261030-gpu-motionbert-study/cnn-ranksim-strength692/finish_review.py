"""Record a completed full audit and predeclared ensemble comparison."""
from pathlib import Path
import json,hashlib,datetime
H=Path(__file__).resolve().parent;S=H.parent;ROOT=S.parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
j=read(H/'analysis/comparison.json');e=read(H/'ensemble/analysis/comparison.json');v=read(H/'ensemble/analysis/verification.json')
assert j['qualified_full_verification_passed'] and j['fits']==50 and j['final_cpu_replays']==10 and j['max_model_replay_error']<1e-4
assert e['passed'] and v['passed'] and v['comparison_sha256']==sha(H/'ensemble/analysis/comparison.json')
assert read(H/'analysis/independent-strength-selection.json')['passed']
receipt=read(H/'checks/full-receipt.json');u=Path(receipt['archive']).parent/'performance-review-originals'
assert receipt['source_succeeded'] and receipt['full_fit_counts_present']
epochs=sum(len(read(p)) for p in (u/'runs/main').glob('*/fold*/*/curve.json'))
lines=['## 완료·회수·성능 검산 — 2026-11-02','',
 f'세 작업 33677·33678·33679의 정상 종료, 새 학습 50개와 바깥 평가 10개를 확인했다. 모델·{epochs}개 epoch의 예측·로그·코드·설정을 D에 회수하고 실제 서버/로컬 압축 및 개별 파일 SHA256을 대조했다. 같은 692명·1,061기록·고정 사람 분할·사람 동일 비중이다. 고득점은 사람 평균 GDI 90 이상인 68명이다. 음수 편향은 과소예측이다.','',
 '강도를 내부 검증에서 선택한 CNN 자체와, 같은 예측을 기존 최선에 추가한 CPU 앙상블을 구분했다. 강도와 결합 비중은 평가 사람을 제외한 내부 자료에서만 골랐다.','',
 '| CNN 자체 또는 기준선 | 전체 MAE | 고득점 MAE | 고득점 편향 |','|---|---:|---:|---:|']
for key,label in [('control_mse','MSE 대조군(0)'),('rank01','RankSim 0.1'),('rank03','RankSim 0.3'),('control_ranksim','기존 RankSim 1'),('inner_selected_strength','내부 선택 강도'),('existing_best','기존 최선 결합'),('existing_rank_ensemble','기존 RankSim 추가 앙상블')]:
 q=j['models'][key];hi=q['subgroups']['person_mean90'];lines.append(f"| {label} | {q['mae']:.6f} | {hi['mae']:.6f} | {hi['bias']:+.6f} |")
lines+=['','| 기존 최선에 추가한 후보 | 전체 MAE | 고득점 MAE | 고득점 편향 |','|---|---:|---:|---:|']
for key,label in [('rank01','강도0.1 추가'),('rank03','강도0.3 추가'),('rank1','기존 강도1 추가 재현'),('joint_inner_selected','내부에서 강도·비중 함께 선택')]:
 q=e['models'][key];hi=q['high90'];lines.append(f"| {label} | {q['mae']:.6f} | {hi['mae']:.6f} | {hi['bias']:+.6f} |")
lines+=['','앙상블의 차이와 불확실성은 다음과 같다. 음수 차이가 오차 감소다.']
for name,data in [('기존 최선 대비',e['paired']),('기존 RankSim 추가 앙상블 대비',e['paired_vs_original_rank_ensemble'])]:
 for key in ['rank01','rank03','joint_inner_selected']:
  p=data[key];lines+=['',f"{name} {key}: 전체 {p['delta']:+.6f}점, 사람 짝지은 10,000회 재표집 95% 범위 {p['bootstrap95'][0]:+.6f}~{p['bootstrap95'][1]:+.6f}, 개선 분할 {p['improved_folds']}/5. 고득점 MAE {p['high90_delta']:+.6f}점(95% {p['high90_bootstrap95'][0]:+.6f}~{p['high90_bootstrap95'][1]:+.6f}). 0을 포함하는 작은 차이를 확실한 개선으로 해석하지 않는다."]
lines+=['',f"최종 모델 10개 CPU float64 재추론 최대 차이는 {j['max_model_replay_error']:.9g}점으로 기존 0.0001 이내다. 모든 epoch의 표본·학습자료 정규화와 보정·사람 분리·행 연결·내부 epoch 선택을 확인했다. 네 강도의 내부 선택과 전체 예측을 별도로 재구성했다."]
if not j['original_exact_local_audit_passed']:
 q=j['supplemental_diagnostics'];lines+=['',f"원본 엄격 로컬 계수 일치(1e-10)는 실패했고 원래 실패 기록을 보존했다. 원서버의 변경 없는 summarize가 원본 감사 50개를 수행하고 정상 요약·종료한 근거와, 별도 저장계수 지표·재적합식 실제 점수 차이 검사를 구분했다. 학습/내부 검증 점수 최대 차이는 {q['train_transformation_max_error']:.9g}/{q['validation_transformation_max_error']:.9g}점이다. 기준을 완화해 원본 실패를 통과로 바꾸지 않았다."]
lines+=['','내부 선택에는 단일 모델, 바깥 평가의 기존 기준선에는 내부 모델 평균을 썼다는 구조 차이가 남는다. 기존 결합 계수는 바깥 학습 자료의 메타학습 결과다. 같은 개발자료를 반복 탐색한 결과이며, 보호 최종자료·LJMU를 사용하지 않았고 외부·임상 일반화 검증은 아니다.','',
 '[CNN 검산·비교](analysis/comparison.json) · [독립 강도 선택](analysis/independent-strength-selection.json) · [앙상블 비교](ensemble/analysis/comparison.json) · [회수](checks/full-receipt.json).','']
p=H/'README.md';text=p.read_text(encoding='utf-8');first,rest=text.split('\n',1)
if lines[0] not in text:p.write_text(first+'\n\n'+'\n'.join(lines)+'\n'+rest.lstrip(),encoding='utf-8')
combined=dict(full_audit_passed_qualified=True,standalone_sha256=sha(H/'analysis/comparison.json'),ensemble_sha256=sha(H/'ensemble/analysis/comparison.json'),epochs=epochs,standalone=j['models'],ensemble=e['models'],paired_vs_best=e['paired'],paired_vs_previous_rank_ensemble=e['paired_vs_original_rank_ensemble'],original_exact_local_audit_passed=j['original_exact_local_audit_passed'])
save(H/'analysis/combined-comparison.json',combined)
regpath=ROOT/'control/gpu-watch/registry.json';reg=read(regpath);assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a'
row=next(x for x in reg['experiments'] if x['id']=='cnn-ranksim-strength-692')
row.update(review_state='reported',reported_digest=sha(H/'analysis/combined-comparison.json'),performance_analysis_sha256=sha(H/'analysis/comparison.json'),performance_analysis_path=str(H/'analysis/comparison.json'),performance_review_state='qualified_verified_exact_local_coefficient_failure_preserved' if not j['original_exact_local_audit_passed'] else 'verified',performance_reported_at=datetime.datetime.now().astimezone().isoformat())
reg['iterative_research'].update(phase='rank_strength_reported_next_planning',result_digest=row['reported_digest'])
save(regpath,reg)
print(json.dumps(dict(report_sha256=row['reported_digest'],standalone=j['models'],ensembles=e['models'],paired=e['paired'],paired_vs_rank=e['paired_vs_original_rank_ensemble']),ensure_ascii=True))
