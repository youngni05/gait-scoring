from pathlib import Path
import json,hashlib,datetime
H=Path('D:/Projects/D_P/studies/20261103-frailpol-objective-threshold');B=Path('D:/gait-research-data/D_P/frailpol-objective-threshold-20261103')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    c=read(H/'comparison.json');v=read(H/'verification.json');assert v['status']=='passed' and v['cases']==80
    t=c['tug']['metrics'];f=c['frailty']['metrics'];blocks=['# FRAILPOL의 노쇠 판정 기준과 TUG 학습 목표 비교',
    '노쇠 판정 기준을 조정하면 기존 확률의 정보를 활용해 노쇠군을 더 찾을 수 있었다. 오탐률 10% 목표의 TabICLv2는 속도만 사용하면 노쇠 43명 중 15명, 특징 추가 시 20명을 찾았다. 실제 오탐은 각각 553명 중 53명과 51명이었다. 다만 특징 추가의 검출률 차이 11.6%p의 95% 구간은 -2.3~25.6%p로 0을 포함했다. 새로운 분별 능력이 생긴 것과 판정 기준을 바꾸어 더 많은 사람을 표시한 효과를 구분해야 한다.',
    'TUG의 초 단위 직접 학습은 TabICLv2의 전체 평균오차를 1.044초에서 1.041초로 거의 바꾸지 않았고, 전체 개선은 확인되지 않았다. TUG 20초 이상인 9명에서는 오차가 6.754초에서 5.447초로 줄었지만 모두 과소예측이 남았다. LightGBM의 느린 사람 가중 학습은 해당 집단의 오차를 줄이는 대신 전체 평균오차를 1.113초에서 1.203초로 늘렸다. 노쇠 집단 균형 가중 학습도 같은 오탐 목표에서 검출의 추가 이점을 확인하지 못했다.',
    '짧은 보행 입력과 사람 분할을 유지하고, 노쇠를 판정하는 기준 및 TUG 학습 목표만 바꾸었다. 노쇠 596명과 TUG 594명을 이전과 같은 다섯 묶음으로 평가했다. 노쇠군은 43명이며, TUG 20초 이상은 9명이다. 기준과 설정은 각 평가 사람을 제외한 학습 사람 안에서만 정했다. 기존 실험을 반복 활용하는 후속 개발 비교이며 독립 외부 검증은 아니다.',
    '## 노쇠군을 찾는 비율과 오탐',
    '기존 모델의 세 집단 확률 중 노쇠 확률에 판정 기준을 적용했다. 정상·전노쇠인 553명 중 잘못 노쇠로 판정하는 비율을 학습 내부에서 10% 또는 20% 이하로 맞추었다. 평가 사람에서 실제 오탐률이 그 값 이하라는 보장은 없다. 두 조건의 실제 오탐률이 다르면 검출률 차이를 동일 오탐률에서의 우월성으로 해석하지 않는다.',
    '각 표의 검출은 노쇠 43명 중 찾아낸 인원이고, 오탐은 나머지 553명 중 잘못 노쇠로 판정한 인원이다. 양성 중 실제 노쇠 비율은 노쇠로 분류된 사람이 실제로 노쇠인 비율이다. 이는 확정 진단의 성능 기준이 아닌 연구용 비교다.']
    for budget in ['argmax','0.1','0.2']:
      blocks += [dict(argmax='### 기존 세 집단 중 최대 확률 판정',**{'0.1':'### 학습 내부 오탐률 10% 목표','0.2':'### 학습 내부 오탐률 20% 목표'})[budget],table(['모델','학습','입력','노쇠 검출','오탐','양성 중 실제 노쇠'],[[key.split('/')[0],{'original':'기존','weighted':'집단 균형 가중'}[key.split('/')[1]],{'speed':'속도만','features':'특징 추가'}[key.split('/')[2]],f"{m[budget]['tp']}/43 ({m[budget]['recall']*100:.1f}%)",f"{m[budget]['fp']}/553 ({m[budget]['fpr']*100:.1f}%)",f"{m[budget]['ppv']*100:.1f}%" if m[budget]['ppv'] is not None else '해당 없음'] for key,m in f.items()])]
    blocks+=['### 특징 추가의 변화와 불확실성',
    '같은 모델·학습 방식에서 속도만 사용하는 경우와 특징을 추가한 경우를 비교했다. 다음은 오탐률 10% 목표에서의 차이다. 95% 구간은 고정된 평가 예측을 사람 단위로 재표집한 조건부 구간이다.',
    table(['모델과 학습','검출률 변화 %p와 95% 구간','오탐률 변화 %p와 95% 구간'],[[key.replace('/features',''),f"{m['0.1']['recall']['difference']*100:.1f} [{m['0.1']['recall']['ci95'][0]*100:.1f}, {m['0.1']['recall']['ci95'][1]*100:.1f}]",f"{m['0.1']['fpr']['difference']*100:.1f} [{m['0.1']['fpr']['ci95'][0]*100:.1f}, {m['0.1']['fpr']['ci95'][1]*100:.1f}]"] for key,m in c['frailty']['paired'].items()]),
    '집단 균형 가중 학습은 LightGBM에 적용했다. 학습 집단에서 드문 종류일수록 더 큰 가중치를 주고, 학습 내부 3분할에서 집단별 확률 오차를 균형 있게 반영해 설정을 선택했다. 가중 학습에서 출력한 확률은 실제 노쇠 유병 확률로 해석하지 않는다. TabICLv2와 TabPFN v2는 기존 고정 가중치의 예측에 판정 기준만 조정했다.',
    '## TUG 시간 학습 목표 변경',
    '로그값으로 학습한 기존 결과와 실제 초를 직접 학습한 결과를 비교했다. LightGBM에서는 학습 TUG 12초 이상 20초 미만에 2배, 20초 이상에 4배 가중치를 주는 초 단위 학습도 추가했다. 평가 사람의 실제 TUG를 입력하거나 예측 단계의 분기 기준으로 사용하지 않았다. 느린 집단의 평균 편향이 음수이면 실제보다 빠를 것으로 예측했다는 뜻이다.',
    table(['모델','학습 목표','입력','전체 MAE 초','20초 이상 MAE 초','20초 이상 편향 초'],[[key.split('/')[0],{'log':'기존 로그','raw':'초 직접','weighted':'초 직접·느린 사람 가중'}[key.split('/')[1]],{'speed':'속도만','features':'특징 추가'}[key.split('/')[2]],f"{m['all']['mae']:.3f}",f"{m['20plus']['mae']:.3f}",f"{m['20plus']['bias']:.3f}"] for key,m in t.items()]),
    '### 같은 입력에서 기존 로그 학습 대비 변화',
    '음수이면 새 방식의 전체 평균오차가 줄었다는 뜻이다. 이 표는 특징 추가 효과와 학습 목표 변경 효과를 구분하기 위한 것이다.',
    table(['모델과 조건','평균오차 변화 초와 95% 구간'],[[key,f"{m['vs_same_input_log']['difference']:.3f} [{m['vs_same_input_log']['ci95'][0]:.3f}, {m['vs_same_input_log']['ci95'][1]:.3f}]"] for key,m in c['tug']['paired'].items() if 'vs_same_input_log' in m]),
    '## 검증과 남은 범위',
    f"80개 평가 조건의 사람 분리·입력·정답·설정 선택·판정 기준을 독립 검산했다. 새 LightGBM 최종 모델과 선택된 내부 모델을 저장 후 다시 불러왔고, 새 기초 모델 예측은 고정 가중치와 학습 자료로 다시 구성해 대조했다. 최대 재추론 차이는 {v['max_replay_difference']:.9g}였다. 원래 모델의 최종 예측은 이전에 검증된 결과를 해시 대조해 재사용했다.",
    '72명의 측정 불가 사례, 가장 느린 TUG 96.4초 사례의 추출 실패, 센서 추정 속도의 정확도 미검증은 이번 변경으로 해결되지 않았다. 입력 길이·특징 구성·센서 구간 선택은 바꾸지 않았다. 보행 느림 항목을 제외한 4항목 점수도 이번 후속 실험에서 다시 학습하지 않았다. 공식 노쇠 판정에는 TUG 기반 느림 항목이 포함되므로 보행 외 상태의 독립 측정으로 해석하지 않는다.',
    '큰 자료와 모델 및 사람별 예측은 `D:/gait-research-data/D_P/frailpol-objective-threshold-20261103`에 보관했다. 비교 기준과 검산 수치는 같은 study 폴더의 protocol.json, comparison.json, verification.json에 기록했다.']
    (H/'RESULTS.md').write_text('\n\n'.join(blocks)+'\n',encoding='utf8')
    status=dict(status='completed_and_verified',cases=80,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),hashes={str(p):sha(p) for p in H.glob('*') if p.is_file() and p.name not in ['status.json','progress.json']})
    save=lambda p,v:p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');save(H/'status.json',status);save(H/'progress.json',dict(status='completed_and_verified',complete=80,expected=80));print('Report and verified completion saved')
if __name__=='__main__':main()
