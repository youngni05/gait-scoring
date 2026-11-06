"""Publish only a finished SHA-verified seed comparison; preserve failed attempts."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 v=read(H/'analysis/full-verification.json');e=read(H/'ensemble/analysis/comparison.json');proof=read(H/'ensemble/analysis/verification.json')
 assert v['qualified_full_verification_passed'] and v['fits']==50 and v['final_cpu_replays']==10 and v['max_model_replay_error']<1e-4
 assert e['passed'] and proof['passed'] and proof['comparison_sha256']==sha(H/'ensemble/analysis/comparison.json')
 assert e['protocol_sha256']==sha(H/'protocol.json')=='56a8a17907bfb06df28aa9bab8301bd47724e48f685aefbd59bf345c6635ea69'
 assert read(H/'analysis/progress.json')['phase']=='complete' and read(H/'ensemble/analysis/progress.json')['phase']=='complete'
 assert read(H/'analysis/process-receipt.json')['active'] is False and read(H/'ensemble/analysis/process-receipt.json')['active'] is False
 models=e['models'];p=e['primary']
 lines=['## 새 seed 회수·검산·비교 — 2026-11-03','',
 '두 조건의 새 학습50개·바깥 평가10개와 모든 epoch를 검산했다. 모델·예측·로그는 대용량 자료 폴더에 보존했고 압축·개별 파일 지문을 대조했다. 같은692명·1,061기록·고정 사람5분할·사람 동일 비중으로 비교했다. 고득점은 사람 평균GDI90 이상68명이며 음수 편향은 과소예측이다.','',
 '| CNN 자체 | 전체 MAE | 고득점 MAE | 고득점 편향 |','|---|---:|---:|---:|']
 labels=[('old_mse','기존 seed MSE'),('new_mse','새 seed MSE'),('mean_old_new_mse','두 seed MSE 평균'),('old_rank1','기존 seed RankSim1'),('new_rank1','새 seed RankSim1'),('mean_old_new_rank1','두 seed RankSim1 평균')]
 for k,label in labels:
  m=models['standalone_'+k];q=m['high90'];lines.append(f"| {label} | {m['mae']:.6f} | {q['mae']:.6f} | {q['bias']:+.6f} |")
 lines+=['','| 기존 최선과 추가 앙상블 | 전체 MAE | 고득점 MAE | 고득점 편향 |','|---|---:|---:|---:|']
 for k,label in [('existing_best','기존 최선 결합'),('existing_rank_ensemble','기존 RankSim 추가'),('add_new_rank1','새 seed RankSim 추가'),('add_mean_old_new_rank1','두 seed RankSim 평균 추가'),('add_mean_old_new_mse','두 seed MSE 평균 추가')]:
  m=models[k];q=m['high90'];lines.append(f"| {label} | {m['mae']:.6f} | {q['mae']:.6f} | {q['bias']:+.6f} |")
 lines+=['','두 seed는1:1로 고정해 평균했다. 최선에 추가하는 비중만0~1의0.05간격에서 바깥 평가 사람을 뺀 내부 예측으로 선택했다. 내부j 예측의 모든 모델은f,j 사람을 모두 학습에서 제외했다. 바깥 결과로 seed나 여섯 후보 중 하나를 다시 고르지 않았다.','',
 '| 두 seed RankSim 평균 추가의 비교 대상 | 전체 MAE 차이 | 사람 짝지은95% 범위 | 개선 분할 | 고득점 MAE 차이 | 고득점 편향 차이 |','|---|---:|---:|---:|---:|---:|']
 for k,label in [('existing_best','기존 최선'),('existing_rank_ensemble','기존 RankSim 추가')]:
  x=e['comparisons']['add_mean_old_new_rank1_vs_'+k]
  lines.append(f"| {label} | {x['delta']:+.6f} | {x['bootstrap95'][0]:+.6f}~{x['bootstrap95'][1]:+.6f} | {x['improved_folds']}/5 | {x['high90_delta']:+.6f} | {x['high90_bias_delta']:+.6f} |")
 lines+=['','오차 차이는 음수일 때 감소다. 고득점 편향 차이는 원래 음수 편향이0에 가까워지는지 함께 본다. 사람 짝지은10,000회 재표집(seed20261101)을 사용했다. 전체 차이95%상한<0·4/5분할 이상 개선·고득점MAE/절대편향 비악화의 사전 기준을 모두 만족한 경우만 양쪽 개선으로 분류한다. 이번 주 비교의 양쪽 개선 기준 통과 여부는 '+str(e['joint_improvement_gate_passed'])+'다.','',
 '| 같은 두 seed RankSim 추가: 고득점과 낮은 점수대 | 인원 | MAE | 편향 |','|---|---:|---:|---:|']
 for k,label in [('person_mean100','사람 평균100 이상'),('person_mean60below','사람 평균60 미만'),('observation_ge90','관측90 이상'),('observation_ge100','관측100 이상')]:
  q=models['add_mean_old_new_rank1']['subgroups'][k];lines.append(f"| {label} | {q['people']} | {q['mae']:.6f} | {q['bias']:+.6f} |")
 lines+=['',f"최종10개 모델 CPU float64 재추론의 최대 점수 차이는{v['max_model_replay_error']:.9g}로 기존0.0001 이내다. 엄격 로컬 계수1e-10 실패는 보존했다. 변경 없는 원서버 summarize의50개 감사 코드 경로 근거와, 별도 저장 계수 지표·재적합식 실제 점수 차이 검사를 구분했다. 저장 기록 오류로 중단된18개 성공 검산은 동일 archive·6,030개 파일 SHA·원래 실행기와 성공 반환 기록을 대조해 재사용했고 나머지를 완료했다.",'',
 f"재적합식과 저장 계수식의 학습/검증 점수 최대 차이는{v['diagnostics']['train_transformation_max_error']:.9g}/{v['diagnostics']['validation_transformation_max_error']:.9g}다. 원본 학습·결과·허용오차는 바꾸지 않았다.",'',
 '내부 단일 모델과 바깥 기준선 모델 평균, 내부100epoch와 바깥 선택epoch의 차이가 남는다. 원래 CNN/STT 결합 비중도 바깥 학습 자료 안의 메타학습 결과다. 두 seed는 초기값 전체의 안정성을 보장하지 않는다. 같은 개발자료를 반복한 결과이며 보호 최종자료·LJMU를 사용한 선택이나 외부·임상 검증이 아니다.','',
 '[전체 모델 검산](analysis/full-verification.json) · [seed·앙상블 비교](ensemble/analysis/comparison.json) · [선택·파일 검증](ensemble/analysis/verification.json) · [회수](checks/full-receipt.json).','']
 text=(H/'README.md').read_text(encoding='utf-8-sig');first,rest=text.split('\n',1)
 if lines[0] not in text:(H/'README.md').write_text(first+'\n\n'+'\n'.join(lines)+'\n'+rest.lstrip(),encoding='utf-8')
 combined=dict(qualified_full_verification_passed=True,verification_sha256=sha(H/'analysis/full-verification.json'),comparison_sha256=sha(H/'ensemble/analysis/comparison.json'),models=models,comparisons=e['comparisons'],primary=e['primary'],joint_improvement_gate_passed=e['joint_improvement_gate_passed'],original_exact_local_audit_passed=False)
 save(H/'analysis/combined-comparison.json',combined)
 regpath=R/'control/gpu-watch/registry.json';reg=read(regpath)
 assert reg['thread_id']=='01a18b12-4a31-7283-a35d-17f28017111a' and reg['automation_id']=='gdi-2'
 row=next(x for x in reg['experiments'] if x['id']=='cnn-ranksim-seed-692')
 row.update(review_state='reported',performance_review_state='qualified_verified_original_exact_failure_preserved',reported_digest=sha(H/'analysis/combined-comparison.json'),performance_analysis_sha256=sha(H/'ensemble/analysis/comparison.json'),performance_analysis_path=str(H/'ensemble/analysis/comparison.json'),performance_reported_at=datetime.datetime.now().astimezone().isoformat())
 reg['iterative_research'].update(phase='rank_seed_reported_next_planning',result_digest=row['reported_digest']);save(regpath,reg)
 print(json.dumps(dict(reported_digest=row['reported_digest'],primary=e['primary'],joint_improvement_gate_passed=e['joint_improvement_gate_passed']),ensure_ascii=False))
if __name__=='__main__':main()

