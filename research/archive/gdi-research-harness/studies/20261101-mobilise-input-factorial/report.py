from study import *

def run():
    verification=json.loads((HERE/'main-verification.json').read_text());assert verification['passed']
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv')
    contrasts=[('features_median','Median4','Median6'),('preserve_core','Median4','Bout4'),('preserve_expanded','Median6','Bout6'),('features_bouts','Bout4','Bout6')]
    effects=[]
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        sub=m[m.target.eq(target)&m.band.eq(band)&m.k.eq(k)]
        for metric in ['brier','accuracy','extreme_error','fpr10_miss','fpr10_fpr']:
            wide=sub.pivot(index='seed',columns='method',values=metric)
            for label,a,b in contrasts+[('interaction',None,None)]:
                delta=wide[b+'_cal']-wide[a+'_cal'] if a else wide.Bout6_cal-wide.Bout4_cal-wide.Median6_cal+wide.Median4_cal
                effects.append(dict(target=target,band=band,k=k,contrast=label,metric=metric,difference=float(delta.mean()),
                    min=float(delta.min()),max=float(delta.max()),improved_seeds=int((delta>0).sum() if metric=='accuracy' else (delta<0).sum())))
    e=pd.DataFrame(effects);e.to_csv(HERE/'factorial-effects.csv',index=False)
    # Person-cluster uncertainty for the interaction, not a causal decomposition.
    losses=pd.read_csv(STORE/'primary-person-losses.csv.gz');rng=np.random.default_rng(20261101);interaction=[]
    for target in ['sppb','tug']:
        for metric in ['brier','fpr10_miss','fpr10_fpr']:
            wide=losses[losses.target.eq(target)&losses.method.isin([n+'_cal' for n in METHODS])].groupby(['id','method'])[metric].mean().unstack().dropna()
            diff=(wide.Bout6_cal-wide.Bout4_cal-wide.Median6_cal+wide.Median4_cal).to_numpy()
            expected=e[e.target.eq(target)&e.band.eq('long')&e.k.eq(5)&e.metric.eq(metric)&e.contrast.eq('interaction')].difference.iloc[0]
            assert abs(float(diff.mean())-expected)<1e-12
            boot=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
            interaction.append(dict(target=target,metric=metric,difference=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    save(HERE/'interaction-bootstrap.json',{'scope':'fixed-prediction person-cluster bootstrap; no retraining; metric interaction not causal attribution','rows':interaction})
    names={'Median4_cal':'4개·중앙값','Median6_cal':'6개·중앙값','Bout4_cal':'4개·구간별','Bout6_cal':'6개·구간별'}
    lines=['# Mobilise-D RF 입력 4조합 비교','',
      '**결론: 특징 추가와 구간별 정보 보존이 모두 개선에 기여했고, 두 변경을 함께 적용한6개·구간별 RF가 대표 조건의 전체 확률 오차와 저기능 놓침이 가장 작았다.**','',
      'SPPB 저기능 놓침은 특징 추가만으로2.68%p, 정보 보존만으로2.40%p, 둘을 함께 적용하면10.26%p 감소했다. 단독 변경의 놓침 차이는 사람 재표집95%구간이0을 포함했다. 두 변경을 결합한 추가 효과(상호작용)는−5.18%p,95%구간−9.60~−1.08%p였다. 초기값5개에서의 방향 일치와 사람 표본 불확실성을 구분한다.','',
      'TUG 놓침은 특징 추가만으로6.05%p, 정보 보존만으로9.47%p, 둘을 함께 적용하면16.86%p 감소했다. 두 단독 비교의95%구간은0을 포함하지 않았다. 확률 오차에서는 특징 추가의 감소가 더 컸다. 어느 변경이 더 중요한지는 평가 지표에 따라 다르다.','',
      '속도가 비슷한 사람의 구분은 두 변경 모두 좋아졌다. 다만 SPPB에서 같은 질환까지 맞추면 특징 추가의 개선은 중앙값 입력1/5·구간별 입력2/5 초기값에 그쳤고 평균은 악화했다. 정보 보존 효과는 같은 질환에서도5/5 유지됐다. TUG는 같은 질환에서도 네 단순 비교 모두5/5 개선됐다.','',
      '두 목표 모두 특징 추가는 여섯 보행 조건 전체에서 평균 확률 오차와 놓침을 줄였다. 정보 보존은 구간3·5개의 네 조건 모두에서 줄였고, 구간1개의 두 조건은 예측이 정확히 같았다.','',
      '보행 속도·빈도·주기·보폭의 기존4개와 구간 길이·회전 빈도를 추가한6개를, 중앙값 입력과 구간별 입력으로 비교했다. 모델은 모두128나무·최소 잎20개·max_features1 RF다. 사람 동일 학습 비중과 기존 확률 보정·내부 오경고10% 경고 문턱을 유지했다.','',
      'SPPB 2,174명·7,356방문, TUG 1,939명·5,688방문, 기존 바깥5/안쪽3 사람 분할·초기값5개·보행 구간을 유지했다. 점수 구간도 SPPB 0–3/4–6/7–9/10–12, TUG ≤10/(10,15]/(15,20]/>20초로 고정했다.','',
      '기존 두 조합을 재사용하고 빠진 두 조합을 학습했다. 구간1개에서는 중앙값과 구간별 입력이 같으므로 같은 모델을 재사용하고 예측 일치를 검산했다. 구간별 입력은 두 특징 수 모두 같은 속도순 정렬을 사용했다.','',
      '## 대표 조건: 30–60초 5개','',
      'Brier는 낮을수록 확률이 정답에 가깝다. 놓침·오경고는 내부 오경고10% 목표로 선택한 별도 경고 규칙의 실제 평가 결과다. 표는 초기값5개의 사람 동일 비중 지표 평균이며 앙상블이 아니다.','',
      '| 목표 | 입력 | Brier | 구간 적중 | 저기능 놓침 | 오경고 | 속도 짝 순서 적중 | 같은 질환·속도 짝 적중 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for target in ['sppb','tug']:
        for method,label in names.items():
            row=m[m.target.eq(target)&m.band.eq('long')&m.k.eq(5)&m.method.eq(method)].mean(numeric_only=True)
            match=mt[mt.target.eq(target)&mt.band.eq('long')&mt.k.eq(5)&mt.method.eq(method)].groupby('matching').pair_order_accuracy.mean()
            lines.append(f'| {target.upper()} | {label} | {row.brier:.4f} | {row.accuracy*100:.2f}% | {row.fpr10_miss*100:.2f}% | {row.fpr10_fpr*100:.2f}% | {match["speed"]*100:.2f}% | {match["speed_cohort"]*100:.2f}% |')
    lines+=['','## 효과 분리','','음수는 오차·놓침·오경고 감소다. 상호작용은 (6개 구간별−4개 구간별)−(6개 중앙값−4개 중앙값)이다.','',
      '| 목표 | 비교 | Brier 차이 | 놓침 차이 | 오경고 차이 | 놓침 개선 초기값 |','| --- | --- | ---: | ---: | ---: | ---: |']
    labels={'features_median':'중앙값에서 4→6개','preserve_core':'4개에서 중앙값→구간별','preserve_expanded':'6개에서 중앙값→구간별','features_bouts':'구간별에서 4→6개','interaction':'상호작용'}
    for target in ['sppb','tug']:
        for contrast in labels:
            q=e[e.target.eq(target)&e.band.eq('long')&e.k.eq(5)&e.contrast.eq(contrast)].set_index('metric')
            lines.append(f'| {target.upper()} | {labels[contrast]} | {q.loc["brier","difference"]:+.4f} | {q.loc["fpr10_miss","difference"]*100:+.2f}%p | {q.loc["fpr10_fpr","difference"]*100:+.2f}%p | {int(q.loc["fpr10_miss","improved_seeds"])}/5 |')
    lines+=['','## 해석 범위와 검산','',
      '속도 짝은 평가 사람 중 같은 분할·다른 사람·속도 차이0.05m/s 이하로 연결했다. 저기능 쪽에 더 높은 확률을 준 비율을 계산하며 동률은0.5로 센다. 이는 구간 적중률이나 모집단 AUROC가 아니다. 비교 대상은 반복 사용될 수 있어 속도 짝 결과는 기술적 분석이다.','',
      '같은 평가 사람을 여러 연구에서 재사용했다. 이 결과는 현재 RF·특징·자료에서의 비교이며 생리적 원인의 인과 분리나 영상 실사용 검증은 아니다. 새 특징 두 개의 개별 효과는 이번4조합으로 분리하지 않았다.','',
      f'검산: {verification["tasks"]}작업, 모델 재현{verification["model_replays"]}개, 지정 재학습{verification["refits"]}개. 지표 독립 계산 최대 차이{verification["independent_metric_error"]:.3g}.',
      '최초 파일럿의 Windows 초기화 함수 참조 오류를 수정했다. 전체 실행 중 프로세스 풀 정체를 확인해 해당 실행만 종료하고, 동일한 학습 함수를4개 스레드로 재개했다. 완료 결과를 재사용했으며 모델·분할·초기값·허용오차는 변경하지 않았다.','',
      f'작은 결과: {HERE}',f'모델·대량 예측: {STORE}','',
      '[여섯 조건 효과](factorial-effects.csv) · [사람 재표집 비교](paired-bootstrap.json) · [상호작용 구간](interaction-bootstrap.json) · [검산](main-verification.json) · [사전 조건](protocol.json)']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[10:38]),flush=True)

if __name__=='__main__':run()
