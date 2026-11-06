from common import *

def main():
    audit=json.loads((HERE/'preparation.json').read_text());ver=json.loads((HERE/'verification.json').read_text());assert ver['passed']
    m=pd.read_csv(HERE/'metrics.csv');comparisons=json.loads((HERE/'comparisons.json').read_text());q=pd.read_csv(HERE/'quality-audit.csv')
    checked=0;ranges=[]
    for name in audit['cohorts']:
        p=pd.read_csv(HERE/name/'predictions.csv');d=load_data(name);p['stable']=p.id.map(dict(zip(d['ids'],d['stable'])))
        for r in m[m.dataset==name].itertuples():
            g=p[(p.model==r.model)&(p.view==r.view)]
            if 'stable' in r.group:g=g[g.stable]
            if 'flagged' in r.group:g=g[~g.stable]
            if 'tail' in r.group:g=g[g.target>=20]
            assert g.id.nunique()==r.n;err=g.prediction-g.target
            for x,y in [(abs(err).mean(),r.mae),(np.maximum(-err,0).mean(),r.under),(np.maximum(err,0).mean(),r.over)]:assert abs(x-y)<1e-10
            checked+=1
        for (model,view),g in p.groupby(['model','view']):
            ranges.append({'dataset':name,'model':model,'view':view,'min':float(g.prediction.min()),'max':float(g.prediction.max()),'nonpositive_rows':int((g.prediction<=0).sum())})
    save('prediction-ranges.json',ranges)
    negative_rows=sum(r['nonpositive_rows'] for r in ranges)
    def row(name,model,view,group):return m[(m.dataset==name)&(m.model==model)&(m.view==view)&(m.group==group)].iloc[0]
    names={'STS_common6':'5회 일어서기','TUG_common6':'TUG'}
    lines=['# 보행 특징의 계산 안정성과 추가 예측 정보 분리 — 2026-10-31','',
        '**특징 계산의 불안정성은 확인했지만, 계산 상태를 통제한 뒤 패턴값이 더 제공하는 예측 정보는 작고 불확실했다. 여러 검출 설정의 중앙값을 쓰는 보정도 뚜렷한 개선을 보이지 않았다. 안정 집단의 긴 사례에서도 큰 오차가 남아, 이번 품질 문제만으로 오차 전부를 설명하기 어렵다. 보행에 필요한 정보가 전혀 없다는 것을 입증한 결과는 아니다.**','',
        '실제 5회 일어서기를 주정답, TUG를 보조 정답으로 비교했다. 품질 변수의 정보와 실제 패턴값의 추가 정보를 나누고, 특징 계산 설정의 변화에 덜 민감한 추정이 도움이 되는지 확인했다.','',
        '## 같은 사람·같은 관찰시간에서 네 입력 비교','',
        '| 입력 | 포함한 정보 | 확인하려는 질문 |','|---|---|---|',
        '| A 속도 | 기존 속도 평균·표준편차·인접 변화량 평균 | 기존 기준을 재현하는가 |',
        '| B 품질 추가 | A + 관절 급변 비율·검출 설정 간 일관성·특징 결측 표시 | 특징값 없이 계산 상태만으로 설명되는 차이가 있는가 |',
        '| C 패턴 추가 | B + 기존 보행 패턴 14개 | 같은 품질 정보를 알고도 패턴값이 도움이 되는가 |',
        '| D 안정화 후보 | B + 9가지 고정 검출 설정의 패턴 중앙값 14개 | 특정 검출 설정에 대한 민감도를 줄이면 나아지는가 |','',
        'B·C·D에는 똑같은 품질 변수 31개가 들어간다. 원래 패턴과 안정화 패턴의 결측 표시를 모두 포함해 결측 여부 자체를 통제한다. 결측치 중앙값과 정규화는 각 학습 집단에서만 계산하며 자동 결측 표시를 추가로 중복 생성하지 않았다. 품질 변수에도 실제 움직임 특성이 반영될 수 있으므로 B를 순수 추적 오류 정보로 해석하지 않는다.','',
        '42.59초 TUG 사례(sub-283)를 제외한 기존 일어서기 179명·TUG 195명, 동일한 방문·파일·시작점·6초 구간을 유지했다. 기존 사람 분할 5가지와 내부 3분할을 그대로 썼다. RF는 기존 128개 나무·잎 크기 2/5·학습 난수 3개, Ridge는 규제값 1/10/100을 사용했다. 설정은 학습 사람 안에서만 선택했다. GPU·새 패키지·새 사람 자료는 사용하지 않았다.','',
        '## 전체 사람 결과','',
        '평균 절대오차(MAE), 초. 반복마다 개인별 오차를 계산한 뒤 평균했다. 예측값을 먼저 평균해서 오차를 줄이지 않았다.','',
        '| 정답 | 모델 | 사람 수 | A 속도 | B 품질 | C 패턴 | D 안정화 후보 |','|---|---|---:|---:|---:|---:|---:|']
    for name in names:
        for model in ['RF','Ridge']:
            rr=[row(name,model,v,'all') for v in VIEWS]
            lines.append(f"| {names[name]} | {model} | {rr[0].n} | "+' | '.join(f'{r.mae:.3f}' for r in rr)+' |')
    lines+=['','## 패턴값의 추가 효과와 안정화 효과','',
        '음수는 오차 감소다. 구간은 사람별 반복 평균 오차를 짝지어 10,000회 재추출한 조건부 95% 구간이다. 사람을 새로 모집하거나 전체 학습을 다시 반복한 불확실성까지 포함하지 않는다. 여러 비교를 한 탐색 결과이며 가장 좋은 조건만 골라 결론을 내리지 않는다.','',
        '| 정답·모델 | 비교 | 전체 MAE 차이 | 조건부 95% 구간 | 낮은 오차를 보인 반복 |','|---|---|---:|---|---:|']
    for c in comparisons:
        if c['group']!='all' or (c['new'],c['base']) not in [('quality','speed'),('pattern','quality'),('robust','quality'),('robust','pattern')]:continue
        tag={'speed':'A','quality':'B','pattern':'C','robust':'D'};lo,hi=c['ci95']
        lines.append(f"| {names[c['dataset']]}·{c['model']} | {tag[c['new']]}−{tag[c['base']]} | {c['mae_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {c['run_wins']}/{c['runs']} |")
    lines+=['','## 계산이 안정적인 집단에서도 같은 비교','',
        '다음 조건을 결과 확인 전에 고정했다. 각도 급변 주변이 전체의 5% 이하, 기존 검출에서 다리마다 완전한 주기 후보 2개 이상, 9개 설정 중 7개 이상에서 주기 계산 가능, 설정 간 주기 추정 범위 0.15초 이하, 기존·안정화 패턴 14개가 모두 계산 가능해야 한다.','',
        '검출 설정은 3/5/7시점 평활화 × 6/8/10도 봉우리 돌출 크기다. 다른 특징 정의는 이전과 같다. 이는 계산 안정성 조건이지 정상 보행·좋은 건강 상태·정확한 관절 추적의 정답이 아니다. 전체 사람으로 학습한 같은 모델의 평가 결과를 안정 집단과 나머지 집단으로 나눴다. 안정 집단만 재학습한 실험은 아니다.','',
        '| 정답 | 집단 | 사람 수 | 실제 검사시간 평균 | 20초 이상 사람 수 |','|---|---|---:|---:|---:|']
    for name,z in q.groupby('dataset'):
        for stable,g in z.groupby('stable'):
            lines.append(f"| {names[name]} | {'안정' if stable else '조건 미충족'} | {len(g)} | {g.target.mean():.3f} | {(g.target>=20).sum()} |")
    lines+=['','안정 집단은 약 1/3만 남으며 실제 검사시간이 더 긴 사람의 비율도 높다. 전체 집단과 안정 집단의 오차 차이를 품질 개선 효과로 해석하지 않는다. 아래 표의 각 행 안에서 A/B/C/D를 비교한다.','',
        '| 정답 | 집단 | 사람 수 | A MAE | B MAE | C MAE | D MAE |','|---|---|---:|---:|---:|---:|---:|']
    for name in names:
        for group,label in [('stable','안정'),('flagged','조건 미충족'),('tail','전체 중 20초 이상'),('stable_tail','안정 중 20초 이상')]:
            rr=[row(name,'RF',v,group) for v in VIEWS]
            lines.append(f"| {names[name]} | {label} | {rr[0].n} | "+' | '.join(f'{r.mae:.3f}' for r in rr)+' |')
    lines+=['','## 긴 검사 시간의 과소예측','',
        '과소예측은 max(실제−예측, 0)을 모든 해당 사람·반복에서 평균한 값이다. 과대예측으로 상쇄하지 않는다. 20초는 분석용 구분이며 임상 판정 기준이 아니다.','',
        '| 정답·모델 | 사람 수 | A | B | C | D |','|---|---:|---:|---:|---:|---:|']
    for name in names:
        for model in ['RF','Ridge']:
            rr=[row(name,model,v,'tail') for v in VIEWS]
            lines.append(f"| {names[name]}·{model} | {rr[0].n} | "+' | '.join(f'{r.under:.3f}' for r in rr)+' |')
    fit=pd.read_csv(HERE/'training-fit-diagnostic.csv');fit_summary=[]
    lines+=['','## 학습 적합과 새 사람 예측의 차이','',
            '이 진단은 학습 시작 뒤 추가한 보조 분석이며 모델이나 설정 선택에 사용하지 않았다. 학습 오차를 평가 성능으로 제시하지 않는다. 사람마다 학습에 포함됐던 모델들의 오차를 먼저 평균했다. 낮은 학습 오차는 자료를 외운 결과일 수도 있고, 높은 학습 오차는 현재 모델의 규제나 표현 제약일 수도 있다.','',
            '| 정답·모델 | 집단 | 입력 | 학습 사람 MAE | 평가 사람 MAE |','|---|---|---|---:|---:|']
    for name in names:
        for model in ['RF','Ridge']:
            for group in ['all','stable_tail']:
                for view in ['quality','pattern','robust']:
                    g=fit[(fit.dataset==name)&(fit.model==model)&(fit.view==view)]
                    if group=='stable_tail':g=g[g.stable&(g.target>=20)]
                    r=row(name,model,view,group)
                    assert len(g)==r.n
                    fit_summary.append({'dataset':name,'model':model,'group':group,'view':view,'n':len(g),'training_mae':float(g.training_mae.mean()),'heldout_mae':float(r.mae)})
                    lines.append(f"| {names[name]}·{model} | {group} | {view} | {g.training_mae.mean():.3f} | {r.mae:.3f} |")
    pd.DataFrame(fit_summary).to_csv(HERE/'fit-versus-heldout.csv',index=False)
    lines+=['','일어서기 RF에서 B→C 변경 시 학습 MAE는 3.108→2.465초로 줄었지만, 평가 MAE는 4.930→4.842초만 줄었다. 안정 집단의 긴 사례에서는 학습 MAE 5.938→5.160초와 달리 평가 MAE는 8.764→9.686초로 악화했다. 현재 특징의 추가 적합 능력이 새 사람의 예측 개선으로 이어지지 않는 양상이다. 과적합·희소한 사례·현재 표현 및 모델의 한계가 남으며, 어떤 원인이 얼마만큼인지 분리한 인과 추정은 아니다.']
    lines+=['','## 해석 가능한 범위','',
        '- B−A는 계산 상태의 추가 정보, C−B는 그 상태를 통제한 기존 패턴값의 추가 정보, D−C는 이번 안정화 후보의 효과다. 이는 모델 조건을 맞춘 예측 비교이며 측정 오류의 인과 효과를 추정한 것은 아니다.',
        '- 안정화 후보가 좋아져도 실제 관절·발 접촉 측정이 정확해졌다고 입증한 것은 아니다. 같은 OpenCap 시계열에서 설정만 바꾼 결과이며 원본 RGB 수동 판독·독립 모션캡처 정답이 없다.',
        '- 안정 집단에서도 오차가 남는다면 계산 불안정성만으로 오차 전부를 설명하기 어렵다는 근거가 된다. 그러나 다른 특징·모델·더 많은 사람으로도 예측할 수 없다는 정보의 한계를 증명하지는 않는다.',
        '- 보행과 다른 동작의 차이, 실제 일어서기 정답의 누락, 적은 긴 TUG 사례, 정답 계측의 변동, 현재 특징·모델의 한계가 함께 남는다. 긴 TUG 8명 중 6명의 일어서기 정답이 없는 상태는 바뀌지 않았다.',
        '- 단일 카메라 성능·일반 고령자 집단·SPPB 점수의 유효성을 검증한 실험이 아니다. 준비·기립·착석·회전 구간은 입력하지 않았다.','',
        f'출력을 제한하지 않은 보조 Ridge에서 0 이하 예측이 총 {negative_rows}개 평가 행에 발생했다. 사후 잘라내지 않고 오차에 포함했다. 모든 RF 예측은 양수였다. 상세 범위는 `prediction-ranges.json`에 기록했다.','',
        '다음 판단에는 원본 영상이나 독립적인 접촉 시점 정답으로 특징 계산이 맞는지 확인한 자료가 필요하다. 자료를 늘릴 때는 걷는 모습은 비슷하지만 일어서기 결과가 다른 사람과 긴 검사 사례의 실제 정답을 확보하는 방향이 중요하다. 이 자료 확보는 이번 실험에서 수행하지 않았다.','',
        '## 검산과 코드','',
        f"기존 자료와 임상 정답·6초 입력·사람 분할을 대조하고 원래 패턴 특징값의 완전 일치를 확인했다. 최종 모델 {ver['final_models_replayed']}개를 재추론했으며 최대 차이는 {ver['max_prediction_delta']:.3g}초다. 기존 속도 모델 예측과의 최대 차이는 {ver['prior_speed_baseline_max_delta']:.3g}초였다. 학습 사람의 결측 처리·표준화와 내부 선택 {ver['inner_fit_selections_checked']}회분의 저장 검증 결과를 확인했다. 내부 모델 전체를 새로 재학습한 검산은 아니다.",
        '',f'집계표 {checked}행을 예측 원자료에서 다시 계산했다. `prepare.py`가 입력·품질 조건·분할을 고정하고, `common.py::extract`가 같은 창 안에서 9개 설정의 특징을 만든다. `run.py`가 네 입력을 학습하며, `verify.py`가 자료와 저장 모델을 대조한다.',
        '', '파일: `protocol.json`, `quality-audit.csv`, `metrics.csv`, `comparisons.json`, `person-results.csv`, `run-results.csv`, `verification.json`, `prediction-ranges.json`, `consistency-audit.png`, `comparison.png`.','']
    (HERE/'STATUS.md').write_text('\n'.join(lines),encoding='utf8')
    save('summary-verification.json',{'metric_rows_recalculated':checked,'passed':True})
    import sklearn,scipy,platform
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    files={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact-manifest.json'}
    save('artifact-manifest.json',files);print('REPORT_VERIFIED',checked,'rows',len(files),'files',flush=True)

if __name__=='__main__':main()
