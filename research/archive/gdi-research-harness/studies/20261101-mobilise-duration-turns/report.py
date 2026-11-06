"""Verified length/turn feature effects with fixed bout representation."""
from study import *

def run():
    verification=json.loads((HERE/'main-verification.json').read_text());assert verification['passed']
    assert verification['analysis_sha256']==sha(HERE/'analyze.py')
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    contrasts=[('length','Bout4','Length5'),('turn','Bout4','Turn5'),('turn_after_length','Length5','Bout6'),('length_after_turn','Turn5','Bout6'),('turn_vs_length','Length5','Turn5'),('both','Bout4','Bout6')]
    effects=[];pair_effects=[]
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        sub=m[m.target.eq(target)&m.band.eq(band)&m.k.eq(k)]
        for metric in ['brier','accuracy','extreme_error','fpr10_miss','fpr10_fpr']:
            wide=sub.pivot(index='seed',columns='method',values=metric)
            for label,a,b in contrasts+[('interaction',None,None)]:
                delta=wide[b+'_cal']-wide[a+'_cal'] if a else wide.Bout6_cal-wide.Length5_cal-wide.Turn5_cal+wide.Bout4_cal
                effects.append(dict(target=target,band=band,k=k,contrast=label,metric=metric,difference=float(delta.mean()),min=float(delta.min()),max=float(delta.max()),improved_seeds=int((delta>0).sum() if metric=='accuracy' else (delta<0).sum())))
        for matching in ['speed','speed_cohort']:
            q=mt[mt.target.eq(target)&mt.band.eq(band)&mt.k.eq(k)&mt.matching.eq(matching)]
            wide=q.pivot(index='seed',columns='method',values='pair_order_accuracy')
            for label,a,b in contrasts:
                delta=wide[b+'_cal']-wide[a+'_cal']
                pair_effects.append(dict(target=target,band=band,k=k,contrast=label,matching=matching,difference=float(delta.mean()),min=float(delta.min()),max=float(delta.max()),improved_seeds=int((delta>0).sum())))
    e=pd.DataFrame(effects);e.to_csv(HERE/'feature-effects.csv',index=False)
    pe=pd.DataFrame(pair_effects);pe.to_csv(HERE/'matched-effects.csv',index=False)
    losses=pd.read_csv(STORE/'primary-person-losses.csv.gz');rng=np.random.default_rng(20261101);interaction=[]
    for target in ['sppb','tug']:
        for metric in ['brier','fpr10_miss','fpr10_fpr']:
            q=losses[losses.target.eq(target)&losses.method.isin([n+'_cal' for n in METHODS])]
            wide=q.groupby(['id','method'])[metric].mean().unstack().dropna()
            diff=(wide.Bout6_cal-wide.Length5_cal-wide.Turn5_cal+wide.Bout4_cal).to_numpy()
            expected=e[e.target.eq(target)&e.band.eq('long')&e.k.eq(5)&e.metric.eq(metric)&e.contrast.eq('interaction')].difference.iloc[0]
            assert abs(float(diff.mean())-expected)<1e-12
            samples=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
            interaction.append(dict(target=target,metric=metric,difference=float(diff.mean()),ci95=np.quantile(samples,[.025,.975]).tolist()))
    save(HERE/'interaction-bootstrap.json',{'scope':'conditional person-cluster resampling of fixed held-out predictions, five seeds averaged per person, no retraining or multiplicity correction','rows':interaction})
    names={'Bout4_cal':'기존 4개','Length5_cal':'길이만 추가','Turn5_cal':'회전 빈도만 추가','Bout6_cal':'둘 다 추가'}
    lines=['# Mobilise-D: 구간 길이와 회전 빈도 분리 비교','',
      '보행 구간별 입력을 유지하고 기존4개·길이만 추가·회전 빈도만 추가·둘 다 추가를 같은 RF로 비교했다. 입력은 속도·보행 빈도·주기·보폭에 필요에 따라 구간 지속 시간과 초당 회전 수를 더한다. 회전 각도나 회전 중 자세 품질을 새로 측정한 실험은 아니다.','',
      'SPPB 2,174명·7,356방문, TUG 1,939명·5,688방문, 동일 바깥5분할·안쪽3분할·초기값5개를 유지했다. 같은 사람의 반복 방문은 같은 평가 분할에 둔다. 구간은[4,10)/[30,60)초 각각1·3·5개이며, 대표 조건은 결과 확인 전 고정한30–60초5개다.','',
      '모든 RF는128나무·최소 잎20개·max_features1, 사람 동일 학습 비중이다. 입력 정규화·온도 보정·경고 문턱은 학습 사람 안에서만 계산했다. 기존4개 및 둘 다 추가 모델은 검산된 결과를 재사용하고 두5개 특징 조합을 새로 학습했다. 구간은 속도와 해당 모델에 보이는 특징만으로 정렬한다. 제외한 특징이 정렬을 통해 입력에 섞이지 않는다.','',
      'SPPB 구간은0–3/4–6/7–9/10–12점, TUG는≤10/(10,15]/(15,20]/>20초다. 아래 값은 사람 동일 비중으로 평가한 초기값5개의 지표 평균이며 앙상블이 아니다.','',
      '## 대표 조건 결과','',
      'Brier는 낮을수록 확률이 정답에 가깝다. 놓침·오경고는 내부 오경고10% 목표로 정한 별도 경고 규칙을 바깥 평가에 적용한 실제 결과다. 실제 오경고가10%로 고정되지는 않는다.','',
      '| 목표 | 입력 | Brier ↓ | 구간 적중 ↑ | 저기능 놓침 ↓ | 오경고 ↓ | 같은 질환·속도 짝 구분 ↑ |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for target in ['sppb','tug']:
        for name,label in names.items():
            row=m[m.target.eq(target)&m.band.eq('long')&m.k.eq(5)&m.method.eq(name)].mean(numeric_only=True)
            match=mt[mt.target.eq(target)&mt.band.eq('long')&mt.k.eq(5)&mt.method.eq(name)&mt.matching.eq('speed_cohort')].pair_order_accuracy.mean()
            lines.append(f'| {target.upper()} | {label} | {row.brier:.4f} | {row.accuracy*100:.2f}% | {row.fpr10_miss*100:.2f}% | {row.fpr10_fpr*100:.2f}% | {match*100:.2f}% |')
    lines+=['','## 초기값 5개와 효과 차이','','아래는 추가 후−추가 전 차이로, 오차·놓침·오경고는 음수가 개선이다.','',
      '| 목표 | 비교 | Brier 차이 | 놓침 차이 | 오경고 차이 | 놓침 개선 초기값 | 같은 질환·속도 짝 구분 차이 | 구분 개선 초기값 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    labels={'length':'기존→길이','turn':'기존→회전','turn_after_length':'길이→둘 다','length_after_turn':'회전→둘 다','both':'기존→둘 다'}
    for target in ['sppb','tug']:
        for contrast,label in labels.items():
            q=e[e.target.eq(target)&e.band.eq('long')&e.k.eq(5)&e.contrast.eq(contrast)].set_index('metric')
            pp=pe[pe.target.eq(target)&pe.band.eq('long')&pe.k.eq(5)&pe.contrast.eq(contrast)&pe.matching.eq('speed_cohort')].iloc[0]
            lines.append(f'| {target.upper()} | {label} | {q.loc["brier","difference"]:+.4f} | {q.loc["fpr10_miss","difference"]*100:+.2f}%p | {q.loc["fpr10_fpr","difference"]*100:+.2f}%p | {int(q.loc["fpr10_miss","improved_seeds"])}/5 | {pp.difference*100:+.2f}%p | {int(pp.improved_seeds)}/5 |')
    lines+=['','## 불확실성과 한계','',
      'paired-bootstrap.json은 네 가지 단순 효과와 두5개 조합 차이·전체 차이의 사람 재표집95%구간을 담는다. 같은 사람의 초기값별 오차를 평균한 뒤2,000번 재표집했으며 재학습·다중비교 보정은 포함하지 않는다. 초기값 방향 일치와 사람 표본의 불확실성은 다르다.','',
      '속도 짝은 평가자료 안에서 같은 질환·같은 평가 분할·서로 다른 사람의 방문을 속도 차이0.05m/s 이내로 연결했다. 저기능 쪽에 더 높은 저기능 확률을 주면1, 동률은0.5로 계산한다. 이는 전체 구간 적중률이나 모집단 AUROC가 아니다. 비교 대상의 반복 사용을 허용한 기술적 분석이다.','',
      '여러 실험에서 같은 평가 사람을 사용한 탐색 결과다. 새 영상·외부 임상 점수에서의 유효성을 증명하지 않는다. 구간 길이·회전 빈도와 정답의 연관은 활동 환경이나 질환 구성과도 관련될 수 있어 생리적 인과효과로 해석하지 않는다. 회전 빈도는 회전 수를 구간 길이로 나눈 값이므로 두 변수를 통계적으로 완전히 독립시킨 실험은 아니다.','',
      f'검산: 전체{verification["tasks"]}작업, 저장 모델 예측 재현{verification["model_replays"]}개, 지정 재학습{verification["refits"]}개. 독립 지표 계산 최대 차이{verification["independent_metric_error"]:.3g}.',
      f'작은 결과: {HERE}',f'모델·대량 예측: {STORE}','',
      '[여섯 조건 특징 효과](feature-effects.csv) · [질환·속도 짝 효과](matched-effects.csv) · [사람 재표집](paired-bootstrap.json) · [검산](main-verification.json) · [사전 조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[12:40]),flush=True)

if __name__=='__main__':run()
