from study import *

if __name__=='__main__':
    v=json.loads((HERE/'main-verification.json').read_text());assert v['passed'] and v['tasks']==50
    assert sha(HERE/'analyze.py')==v['analysis_sha256'] and sha(HERE/'verify.py')==v['verify_sha256']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    lines=['# 회전 빈도 기반 HGB와 RF 비교','',
      '질문은 구간별 기존4개+회전 빈도를 유지하면서 모델을 HGB로 바꾸면 RF의 저기능 놓침을 줄일 수 있는가다. 같은30–60초5개, SPPB2,174명/TUG1,939명, 바깥5분할·안쪽3분할·초기값5개를 유지했다. RF5개 특징이 주 비교군이고 길이를 포함한RF6개가 보조 비교군이다.','',
      'HGB는 CPU에서 학습했다. 학습률0.05·200단계·최소 잎20·L2=1·자동 조기종료 끄기를 고정하고 잎 수7/15만 학습 사람 내부 로그손실로 선택했다. 정규화·확률 보정·경고 문턱은 학습 사람 안에서만 계산했다. 설정 선택에 사용한 내부 예측을 보정에도 재사용하므로 내부 성능은 최종 성능으로 보고하지 않는다.','',
      '| 목표 | 방법 | Brier ↓ | 구간 적중 ↑ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 짝 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    labels={'TurnRF_cal':'회전 RF','FullRF_cal':'길이+회전 RF','HGB_cal':'회전 HGB'}
    for t in ['sppb','tug']:
        for name,label in labels.items():
            q=m[m.target.eq(t)&m.method.eq(name)].mean(numeric_only=True)
            pair=mt[mt.target.eq(t)&mt.method.eq(name)&mt.matching.eq('speed_cohort')].pair_order_accuracy.mean()
            lines.append(f'| {t.upper()} | {label} | {q.brier:.4f} | {q.accuracy*100:.2f}% | {q.fpr10_miss*100:.2f}% | {q.fpr10_fpr*100:.2f}% | {pair*100:.2f}% |')
    lines+=['','Brier는 확률 오차다. 놓침·오경고는 내부 오경고10% 목표로 고정한 별도 경고 규칙의 실제 평가 결과다. 속도 짝은 같은 질환·평가 분할에서 다른 사람을0.05m/s 이내로 연결해 저기능 쪽 확률이 높은지 계산한다. 동률은0.5이며 모집단 AUROC나 구간 적중률이 아니다.','',
      '이 표는 초기값5개의 사람 동일 비중 지표 평균이다. 같은 개발자료를 반복 사용한 탐색 결과로, 입력 선택 자체가 이전 바깥 평가 결과에 영향을 받았다. 새 영상·외부 임상 유효성을 증명하지 않는다. paired-bootstrap.json의 조건부 사람 재표집 구간과 초기값별 결과를 함께 확인한다.','',
      f'검산50작업, 모델 예측 재현{v["model_replays"]}개, 선택 재학습{v["refits"]}개, 독립 지표 오차{v["independent_metric_error"]:.3g}.',
      f'작은 결과: {HERE}',f'모델·대량 예측: {STORE}','',
      '[초기값별 지표](metrics-by-seed.csv) · [비교 구간](paired-bootstrap.json) · [속도 짝](matched-by-seed.csv) · [검산](main-verification.json) · [조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('REPORT_READY',flush=True)
