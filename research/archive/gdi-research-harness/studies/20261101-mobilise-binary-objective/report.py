from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==50
    assert sha(HERE/'analyze.py')==v['analysis_sha256'] and sha(HERE/'verify.py')==v['verify_sha256']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 보행 기반 저기능 직접 분류와 확률 보정 비교','',
      '질문은 네 점수 구간을 함께 맞추는 대신 저기능을 직접 학습하면 놓침을 줄일 수 있는가다. SPPB≤3점/TUG>20초를 그대로 사용하고 측정 정답을 바꾸지 않았다. 같은2,174명/1,939명,30–60초5개,기존4개+회전 빈도25열,사람5분할·초기값5개를 유지했다.','',
      'OriginalRF는 이전4구간RF의 온도 보정·경고 기준을 그대로 재현한다. MultiRF/MultiHGB는 기존4구간 모델의 저기능 확률에 공통 이진 Platt 보정만 새로 적용한다. BinaryRF/BinaryHGB는 저기능 여부를 직접 학습하고 같은 보정을 사용한다. 이 비교로 모델 목표 변경과 확률 보정 변경을 구분한다. RF는128나무·최소잎20, HGB는200단계·학습률0.05·최소잎20·L2=1을 고정하고 잎7/15만 내부 검증에서 고른다. 모든 모델은 사람 동일 학습 비중이며 저기능 가중치를 추가하지 않았다.','',
      '아래 Brier는 이진 확률 오차이므로 이전4구간 Brier와 수치 자체를 직접 비교하면 안 된다. 놓침·오경고는 학습 내부 오경고10% 기준에서 정한 문턱을 바깥 평가에 적용한 결과다. 값은 초기값5개의 사람 동일 비중 지표 평균이다.','',
      '| 목표 | 방법 | 이진 Brier ↓ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 짝 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: |']
    for t in ['sppb','tug']:
        for name in METHODS:
            row=m[m.target.eq(t)&m.method.eq(name)].mean(numeric_only=True)
            match=mt[mt.target.eq(t)&mt.method.eq(name)&mt.matching.eq('speed_cohort')].pair_order_accuracy.mean()
            lines.append(f'| {t.upper()} | {name} | {row.brier:.5f} | {100*row["miss"]:.2f}% | {100*row.fpr:.2f}% | {100*match:.2f}% |')
    lines+=['','내부 모델 선택·확률 보정·문턱 계산에는 바깥 평가 정답을 사용하지 않았다. 같은 내부 예측을 설정 선택과 보정에 재사용하므로 내부 지표는 선택용이다. 여러 후속 실험에 같은 평가 사람을 재사용한 탐색이며 새 임상 영상의 검증은 아니다.','',
      '속도 짝은 같은 평가 분할·같은 질환의 다른 사람을0.05m/s 이내로 연결했다. 저기능 쪽 확률이 높으면1, 동률은0.5다. 비교 대상 반복 사용이 가능하므로 이 구분 비율을 독립 표본 정확도나 모집단 AUROC로 해석하지 않는다.','',
      f'검산:50작업, 모델·보정기 재현{v["model_and_calibrator_replays"]}개, 지정 재학습{v["refits"]}개, Brier·로그손실·놓침·오경고 독립 계산 최대 차이{v["independent_metric_error"]:.3g}. 분할·입력·정답·학습 정규화·보정·경고 문턱·속도 짝도 확인했다.',
      f'작은 결과: {HERE}',f'모델·대량 예측: {STORE}','',
      '[초기값별 결과](metrics-by-seed.csv) · [사람 재표집 비교](paired-bootstrap.json) · [속도 짝](matched-by-seed.csv) · [검산](main-verification.json) · [조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('REPORT_READY',flush=True)
