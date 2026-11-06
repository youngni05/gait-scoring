from pathlib import Path
import json,hashlib,datetime,sys
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');B=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');sys.stdout.reconfigure(encoding='utf8')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    c=read(H/'comparison.json')['results'];v=read(H/'verification.json');s=read(H/'signal-verification.json');a=read(H/'data-audit.json');assert v['status']==s['status']==read(H/'final-audit.json')['status']=='passed'
    blocks=['# FRAILPOL 짧은 보행의 TUG와 노쇠 예측 결과',
    '짧은 보행 특징을 추가하면 TUG 시간 예측은 세 모델 모두 개선됐다. TabICLv2의 평균오차는 1.281초에서 1.044초로 약 18.5% 줄었다. 공식 노쇠 3단계 판정과 보행 느림을 뺀 4개 항목 예측에서는 추가 특징의 개선을 확인하지 못했다. 특히 실제 노쇠군 43명 중 노쇠로 찾아낸 사람은 현재 판정 방식에서 모델별 0–1명으로, 노쇠 판정 모델로 사용하기에는 부족했다.',
    '지역사회 고령자의 짧은 직선 보행에서 얻은 정보가 속도 하나보다 기능 상태를 잘 예측하는지 비교했다. 전체 TUG 기록에서 자동으로 고른 한쪽 발의 한 보행 주기와 같은 시간의 몸통 신호를 사용했다. 아래 결과는 학습에 쓰지 않은 사람의 예측만 모아 계산했다.',
    f"668명 중 {a['measurable']}명에서 특징을 추출했다. 사용 구간은 {a['support_seconds']['min']:.2f}–{a['support_seconds']['max']:.2f}초, 중앙값 {a['support_seconds']['median']:.2f}초였다. TUG 정답이 있는 594명, 공식 노쇠 정답이 있는 596명, 보행 느림을 제외한 4항목 정답이 일관되게 있는 590명을 각각 평가했다.",
    '## TUG 소요 시간 예측',
    '평균 절대오차는 예측 시간과 실제 시간의 차이를 초 단위로 평균한 값으로, 작을수록 좋다. 오차 변화는 특징 추가에서 속도만 사용한 결과를 뺀 값이다. 음수이면 개선이다. 95% 구간은 같은 사람의 두 예측을 짝지어 10,000번 재표집해 계산했다.',
    table(['모델','속도만 MAE 초','특징 추가 MAE 초','오차 변화와 95% 구간','특징 추가 R²'],[[k,f"{m['speed']['mae']:.3f}",f"{m['features']['mae']:.3f}",f"{m['paired_difference']['feature_minus_speed']:.3f} [{m['paired_difference']['ci95'][0]:.3f}, {m['paired_difference']['ci95'][1]:.3f}]",f"{m['features']['r2']:.3f}"] for k,m in c['tug']['models'].items()]),
    f"학습 사람들의 TUG 중앙값만 반복해서 출력하는 기준의 오차는 {c['tug']['baseline']['mae']:.3f}초였다. TUG는 로그값으로 학습하고 초로 되돌렸으며, 높은 TUG 정답이나 예측을 잘라내지 않았다.",
    '## 공식 노쇠 3단계 예측',
    '정상·전노쇠·노쇠를 구분했다. 균형 정확도는 각 집단을 맞힌 비율의 평균이므로, 정상군만 맞혀 전체 정확도를 높이는 문제를 줄여 보여준다. 항상 한 종류만 출력하면 약 33.3%다. 노쇠 재현율은 실제 노쇠군 43명 중 노쇠로 찾아낸 비율이다.',
    table(['모델','속도만 균형 정확도','특징 추가 균형 정확도','차이와 95% 구간','특징 추가 노쇠 재현율'],[[k,f"{m['speed']['balanced_accuracy']*100:.1f}%",f"{m['features']['balanced_accuracy']*100:.1f}%",f"{m['paired_difference']['balanced_accuracy_feature_minus_speed']*100:.1f}%p [{m['paired_difference']['balanced_accuracy_ci95'][0]*100:.1f}, {m['paired_difference']['balanced_accuracy_ci95'][1]*100:.1f}]",f"{m['features']['class_recall'][2]*100:.1f}%"] for k,m in c['frailty']['models'].items()]),
    table(['모델','속도만 확률 오차','특징 추가 확률 오차','변화와 95% 구간','특징 추가 macro F1'],[[k,f"{m['speed']['logloss']:.3f}",f"{m['features']['logloss']:.3f}",f"{m['paired_difference']['feature_minus_speed']:.3f} [{m['paired_difference']['ci95'][0]:.3f}, {m['paired_difference']['ci95'][1]:.3f}]",f"{m['features']['macro_f1']:.3f}"] for k,m in c['frailty']['models'].items()]),
    '확률 오차는 log loss로, 틀린 답을 높은 확률로 내놓을수록 커지며 작을수록 좋다. 모델 설정 선택에는 학습 사람 내부의 확률 오차를 사용했다. 공식 노쇠 판정에는 TUG 기반 느림 항목이 포함되므로, 이 결과만으로 보행 외의 신체 상태까지 측정했다고 말할 수 없다.',
    '세 집단 중 확률이 가장 높은 집단을 고르는 현재 판정은 드문 노쇠군을 거의 선택하지 않았다. 특징을 추가한 모델의 노쇠 대 나머지 확률 순위 AUC는 약 0.80–0.82로, 순서에 관한 정보 자체가 없다는 뜻은 아니다. 다만 노쇠군을 더 많이 찾도록 가중치나 판정 기준을 바꾼 실험은 이번에 실행하지 않았다. 평가 결과를 본 뒤 유리한 기준으로 바꾸지 않았다.',
    '## 보행 느림을 제외한 4개 항목 예측',
    '체중 감소·쇠약 관련 응답·활동 부족·근력 저하의 제공된 0/1 항목을 더한 0–4점이다. 공식 노쇠 진단을 새로 정의한 값이 아니라, 속도와 직접 겹치는 항목을 뺀 보조 분석이다. 원본 항목 누락 또는 합계 불일치 6명은 이 분석에서만 제외했다.',
    table(['모델','속도만 MAE 점','특징 추가 MAE 점','오차 변화와 95% 구간'],[[k,f"{m['speed']['mae']:.3f}",f"{m['features']['mae']:.3f}",f"{m['paired_difference']['feature_minus_speed']:.3f} [{m['paired_difference']['ci95'][0]:.3f}, {m['paired_difference']['ci95'][1]:.3f}]"] for k,m in c['non_slowness']['models'].items()]),
    f"학습 사람의 중앙값만 출력하는 기준은 {c['non_slowness']['baseline']['mae']:.3f}점이었다. 세 모델 모두 이 단순 기준보다 평균 절대오차가 컸다. 이 정답은 0점 347명, 1점 181명, 2점 48명, 3점 13명, 4점 1명으로 높은 점수 사례가 적다.",
    '## 상태가 나쁜 사람과 분석에서 빠진 사람',
    'TUG가 20초 이상인 평가 대상은 9명뿐이다. 아래 편향이 음수이면 실제보다 빠르게 검사할 것으로 예측했다는 뜻이다.',
    table(['모델','조건','TUG 20초 이상 MAE 초','평균 편향 초'],[[k,cond,f"{c['tug']['subgroups']['tug_20plus'][k+'/'+cond]['mae']:.3f}",f"{c['tug']['subgroups']['tug_20plus'][k+'/'+cond]['bias']:.3f}"] for k in c['tug']['models'] for cond in ['speed','features']]),
    table(['공식 집단','전체 인원','분석 가능 인원'],[[['정상','전노쇠','노쇠'][int(k)],v['total'],v['measurable']] for k,v in a['by_frailty'].items()]),
    '72명의 측정 불가 사유는 직선 주기 검출 실패 41명, 센서 시간 불일치 12명, 비현실적인 복원 궤적 10명, 각속도 단위 확인 실패 8명, 누락 또는 중복 패킷 1명이었다. 실제 TUG 96.4초인 가장 느린 참가자도 직선 주기 검출에 실패해 빠졌다. 그러므로 가장 심한 보행 저하까지 성능을 입증한 결과가 아니다.',
    '## 입력과 비교 절차',
    '속도 조건은 같은 짧은 구간에서 복원한 발목 센서의 이동 거리÷주기 시간 1개 값이다. 확장 조건은 이를 포함한 19개 값으로, 주기 시간·발을 들어 이동하는 시간 비율·이동 거리·센서 높이·좌우 방향 이동·기울기 변화·발목 및 몸통 신호의 흔들림 크기를 사용했다. 관절 중심의 실제 각도나 장기간 보행 변동성을 측정한 것으로 해석하지 않는다.',
    '원본 668명의 ID를 공식 노쇠 비율이 비슷한 5개 묶음으로 고정하고, 한 묶음을 평가할 때 나머지 사람만 학습에 사용했다. 품질 및 정답 누락 필터를 적용해도 사람의 묶음은 바꾸지 않았다. LightGBM은 학습 사람 내부에서 3분할로 16개 설정을 비교했다. TabICLv2와 TabPFN v2는 고정된 사전학습 가중치와 8개 앙상블을 사용했다. 평가 대상의 정답으로 모델·설정·확률 기준을 고르지 않았다.',
    'TUG 전체 시간·전체 걸음 수·기록 길이·연령·성별·검사 항목·질환·기관 정보는 입력에 넣지 않았다. 원본 안내의 각속도 단위와 실제 자세 변화 사이의 불일치를 확인해, 원신호를 라디안/초로 해석하고 gaitmap 입력인 도/초로 변환했다. 외부 Barth 보행 틀을 사용했으며 이 자료의 평가 사람으로 보행 틀을 학습하지 않았다.',
    '## 해석 범위와 검증',
    '속도 자체가 발목 IMU에서 추정한 값이며, 이 자료에서 정밀 장비의 속도 정답으로 검증하지 못했다. 특징 추가 효과에는 상태 정보뿐 아니라 속도 추정 오차를 보완한 효과가 포함될 수 있다. 전체 TUG에서 구간을 찾아낸 실험이므로, 별도로 촬영한 짧은 영상이나 다른 날 검사로의 성능도 아직 확인되지 않았다. 이번 모델은 기존 뇌성마비 GDI 모델과 별도의 센서 기반 예측 모델이다.',
    f"최종 평가 모델 {v['models']}개를 모두 저장 상태 또는 고정 가중치와 학습 자료로 다시 구성해 재추론했다. 최대 예측 차이는 {v['max_abs_replay_difference']:.9g}였다. 30개 사람 분할과 30개 내부 설정 선택을 검사했고, 596명의 원신호 특징 재계산은 모두 일치했다. {s['perturbation_people']}명에서 선택 구간 밖 신호를 임의로 바꾸어도 특징은 변하지 않았다. 원자료와 가중치 SHA256도 대조했다.",
    '95% 구간은 고정된 교차검증 예측에서 사람을 다시 뽑은 조건부 불확실성이다. 분할을 바꾸어 재학습했을 때의 불확실성이나 독립 외부 검증을 대신하지 않는다. 65세 이상과 파일럿 대상 제외 분석, 점수 구간별 결과 및 혼동행렬은 comparison.json에 함께 기록했다.',
    '원자료와 모델·사람별 예측은 `D:/gait-research-data/D_P/frailpol-short-walk-20261103`에, 비교 기준·검증 기록과 이 보고서는 현재 study 폴더에 보관했다.',
    '자료: [FRAILPOL 원 논문](https://www.nature.com/articles/s41597-026-06854-8), [공개 자료 목록](https://api.figshare.com/v2/articles/29315771), [저자 분석 코드](https://github.com/aabbasi-polsl/FRAILPOL_classification). 자료 파일의 제공 라이선스는 CC BY 4.0이다.']
    (H/'RESULTS.md').write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    files=['protocol.json','data-audit.json','preparation-receipt.json','feature-receipt.json','weights-manifest.json','run-receipt.json','comparison.json','verification.json','signal-verification.json','final-audit.json','requirements-signals.txt','requirements-models.txt','RESULTS.md','tug-predictions.png']
    manifest={str(H/f):sha(H/f) for f in files};manifest.update({str(p):sha(p) for p in H.glob('*.py')})
    status=dict(status='completed_and_verified',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=90,measurable_people=596,report=str(H/'RESULTS.md'),hashes=manifest)
    (H/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False),encoding='utf8');print('report saved; verified models',v['models'])
if __name__=='__main__':main()
