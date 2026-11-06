from pathlib import Path
import json,hashlib,datetime
H=Path('D:/Projects/D_P/studies/20261103-cnn-stable-scale');B=Path('D:/gait-research-data/D_P/cnn-stable-scale-20261103');OWNER='01a19c6f-49f7-7300-97b0-a303b71ffe91'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
r=read(H/'results.json');v=read(H/'verification.json');iv=read(H/'inner-verification.json');recovery=read(H/'recovery.json');assert v['passed'] and iv['passed'] and iv['models']==40 and v['original_models_replayed']==10 and sha(H/'results.json')==v['results_sha256']
a=r['metrics']['baseline'];b=r['metrics']['stable_scale'];lines=['# 같은 CNN의 크기 보정 비교 결과',
'같은 692명의 같은 보행 구간에서 CNN 구조와 학습 조건을 유지하고, 프레임마다 달라지는 크기 보정값을 영상 전체의 중앙값으로 안정화했다. 평균오차는 조금 낮아졌지만 불확실성을 고려하면 개선을 입증하지 못했다. 낮은 점수는 높게, 높은 점수는 낮게 예측하는 경향도 남았다.',
'| 비교 항목 | 기존 입력 | 안정화 크기 보정 |\n| --- | --- | --- |\n'+f"| 사람별 평균 절대오차 | {a['mae']:.3f}점 | {b['mae']:.3f}점 |\n| 오차가 큰 상위 10%의 경계 | {a['p90']:.3f}점 | {b['p90']:.3f}점 |\n| 평균오차 10점 초과 사람 | {a['people_over10']}명 | {b['people_over10']}명 |\n| 평균오차 20점 초과 사람 | {a['people_over20']}명 | {b['people_over20']}명 |\n| 가장 큰 사람별 평균오차 | {a['maximum']:.3f}점 | {b['maximum']:.3f}점 |",
f"안정화 입력−기존 입력의 오차 차이는 {r['delta_mae']:.3f}점, 사람 단위 짝지은 95% 구간은 [{r['delta_ci95'][0]:.3f}, {r['delta_ci95'][1]:.3f}]점이다. 음수이면 개선이다. 구간에 0이 포함되어 이번 개발자료에서 확실한 개선으로 판단하지 않았다. 370명은 개선됐고 322명은 악화됐다. 5개 평가 분할 중 3개는 개선, 2개는 악화됐다.",
'## 점수대별 오차와 편향',
'점수대는 각 사람의 실제 GDI 평균으로 나눴다. 편향은 예측−정답이며, 양수이면 실제보다 높은 점수를 준 것이다.',
'| 실제 GDI | 사람 수 | 기존 오차 | 안정화 오차 | 기존 편향 | 안정화 편향 |\n| --- | --- | --- | --- | --- | --- |']
for key,label in [('below60','60 미만'),('60to75','60 이상 75 미만'),('75to90','75 이상 90 미만'),('90plus','90 이상')]:
    x=a['bands'][key];y=b['bands'][key];lines[-1]+=f"\n| {label} | {x['people']} | {x['mae']:.2f} | {y['mae']:.2f} | {x['bias']:+.2f} | {y['bias']:+.2f} |"
lines.extend(['중간 점수대의 개선이 전체 평균을 낮췄다. 60 미만과 90 이상에서는 평균오차가 증가했다. 따라서 크기 보정 안정화가 점수 양끝의 오차를 해결했다고 볼 수 없다.',
f"학습 사람으로만 맞춘 선형 보정을 적용하기 전 오차는 기존 {a['raw_mae']:.3f}점, 안정화 {b['raw_mae']:.3f}점이었다. 위 주분석은 사전에 정한 선형 보정 후 결과다. 어느 값을 보고할지 평가 결과를 보고 바꾸지 않았다.",
'## 기존 최선 결과와 해석 범위',
'이전 최선 앙상블의 평균오차 약 6.52점보다 이번 안정화 단일 CNN의 6.77점이 낮지 않다. 이전 앙상블과 이번 단일 CNN은 모델 구성도 달라, 이번 실험의 주비교는 새로 동일 조건으로 학습한 기존 입력 6.87점과 안정화 입력 6.77점이다.',
'이번 결과는 반복해서 사용한 뇌성마비 중심 개발자료에서 한 초기값으로 진행한 비교다. 다른 초기값의 안정성, 독립 최종 평가자료, 고령자의 GDI 정확도 또는 실제 관절 좌표 정확도를 입증하지 않는다. 크기 보정 하나의 효과가 불확실하다는 결과이며, 모든 좌표 오차가 중요하지 않다는 뜻은 아니다.',
'## 학습과 검산',
'692명·1,061개 기록·11,261개 구간에 대해 사람 분리 5분할을 사용했다. 각 평가 분할의 나머지 사람 안에서 4개 내부 분할로 epoch를 선택하고 새로 학습했다. 두 입력 조건의 구조·초기값·표본 순서·손실·최적화·정답 정규화·보정 방식은 동일하다. 총 50개 학습과 최종 평가 모델 10개가 완료됐다.',
f"원본 코드·모델·결과·로그 {recovery['files']}개를 회수하고 서버·로컬 SHA256을 대조했다. 최종 평가 모델 10개와 내부 모델 40개 모두 CPU에서 재추론했다. 최종 보정 예측의 서버 대비 최대 차이는 {max(x['calibrated_replay_max_error'] for x in r['audits']):.8f}점, 내부 모델 지표의 최대 차이는 {iv['max_metric_difference']:.8f}점으로 사전 기준 0.001점 이내였다. 사람 분리, 내부 epoch 선택, 학습 사람만의 정답 정규화와 선형 보정을 독립 확인했다.",
'검산 시작 때 Python 3.13과 기존 Python 3.12용 PyTorch의 불일치를 발견해 호환 환경으로 실행했다. 이어 검산 코드가 정답을 먼저 float32로 반올림하는 오류를 확인했다. 원본 학습은 원래 정답 자료형에서 범위를 계산한 뒤 float32로 저장하므로, 이 계산 순서를 정확히 대조하도록 검산 코드만 수정했다. 원본 모델·학습·허용오차는 변경하지 않았다. 최초 검산 코드는 review_results.initial.py로 보존했다.',
'95% 구간은 고정된 학습 결과에서 사람을 10,000번 다시 뽑아 계산한 조건부 불확실성이다. 초기값이나 사람 분할을 바꿔 재학습했을 때의 불확실성은 포함하지 않는다.',
'원본 보관: `D:/gait-research-data/D_P/cnn-stable-scale-20261103/received/final/original`. 사람별 비교 예측은 같은 대용량 폴더의 `paired-predictions.npz`, 상세 지표는 이 실험 폴더의 `results.json`에 있다.'])
(H/'RESULTS.md').write_text('\n\n'.join(lines)+'\n',encoding='utf8')
save(H/'review-amendment.json',dict(reason='Fix checker premature float32 conversion; compare exact original scale_from_train calculation order',initial_checker_sha256=sha(H/'review_results.initial.py'),corrected_checker_sha256=sha(H/'review_results.py'),source_training_script_sha256=sha(Path(recovery['received_root'])/'run.py'),original_threshold_gdi=.001,unchanged_threshold_gdi=.001,source_model_training_untouched=True,normalization_checks='exact equality using original y dtype then float32 storage',python='C:/Users/jun/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'))
reg=read(H/'watch/registry.json');assert reg['thread_id']==OWNER and reg['automation_id']=='gdi-cnn';e=next(e for e in reg['experiments'] if e['id']=='cnn-stable-scale-20261103');assert e['jobs']==['34737','34738']
reg.update(last_checked_at=datetime.datetime.now().astimezone().isoformat(),last_observed_state='completed_verified_pending_report',last_snapshot=str(H/'watch/snapshot-20261103-230938.json'),last_progress=dict(completed_fits=50,expected_fits=50,completed_outer_predictions=10,expected_outer_predictions=10));e.update(review_state='verified_pending_report',results_sha256=sha(H/'results.json'),report=str(H/'RESULTS.md'));save(H/'watch/registry.json',reg)
status=read(H/'status.json');status.update(status='completed_verified_pending_report',checked_at=reg['last_checked_at'],completed_fits=50,completed_outer_predictions=10,remaining=['report results and stop this monitor'],results=str(H/'results.json'),verification=str(H/'verification.json'),inner_verification=str(H/'inner-verification.json'),recovery=str(H/'recovery.json'));save(H/'status.json',status)
print(json.dumps(dict(status=status['status'],results_sha256=sha(H/'results.json'),report_sha256=sha(H/'RESULTS.md'))))
