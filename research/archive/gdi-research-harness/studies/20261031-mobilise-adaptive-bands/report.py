"""Create a concise, reproducible report from verified complete comparisons."""
from run import *
NAMES={'RFreg':'회귀 RF → 확률','RF':'분류 RF','OrdinalRF':'순서형 RF'}
SCHEMES={'fixed4':'기존 4구간','adjusted4':'조정 4구간','merged3':'통합 3구간'}
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def pct(v):return f'{100*v:.2f}%'
def rulelabel(rule):
    if rule=='mode':return '최댓값'
    if rule=='absolute':return '거리 최소'
    if rule=='squared':return '거리 제곱 최소'
    _,d,miss,fp=rule.split('-');return f'혼합(거리 {d}, 놓침 {miss}, 오경고 {fp})'
def bandname(target,cuts):
    if target=='sppb':return '0–3 / 4–9 / 10–12' if len(cuts)==2 else f'0–3 / 4–{int(cuts[1])} / {int(cuts[1])+1}–9 / 10–12'
    return '≤10 / (10,20] / >20' if len(cuts)==2 else f'≤10 / (10,{cuts[1]:g}] / ({cuts[1]:g},20] / >20'
def run():
    v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed'
    m=pd.read_csv(HERE/'metrics-by-seed.csv');p=pd.read_csv(HERE/'probabilities-by-seed.csv');o=pd.read_csv(HERE/'warnings-by-seed.csv');s=pd.read_csv(HERE/'inner-selections.csv')
    primary=m[m.band.eq('long')&m.n.eq(5)];pp=p[p.band.eq('long')&p.n.eq(5)];op=o[o.band.eq('long')&o.n.eq(5)];sp=s[s.band.eq('long')&s.n.eq(5)]
    parts=['# Mobilise-D: 점수 경계와 판정 규칙 비교',
           '**결론: 구간을 바꾸면 일부 적중률과 양끝 오판은 개선되지만, 저기능 놓침까지 함께 줄이는 안정적인 개선은 얻지 못했다.** 대표 조건에서 조정 4구간의 구간별 평균 적중률 개선은 SPPB 0.33–1.28%p, TUG 0–0.45%p였다. 기존 경계가 모델별 25번 선택 중 80–100% 유지됐다. TUG 순서형 RF는 25번 모두 기존 15초 가운데 경계와 최댓값 판정을 유지했다.',
           '3구간으로 합치면 SPPB 적중률은 56.23–56.97%, TUG는 57.19–58.09%로 올랐다. 양끝 오판과 오경고도 줄었다. 그러나 낮은 SPPB의 놓침은 94.91–100%, 긴 TUG의 놓침은 73.14–81.31%가 됐다. 대표 조건에서 세 모델 모두 초기값 5개 전체에서 적중률·양끝 오판·오경고는 개선되고 저기능 놓침은 악화됐다. 가운데 범위로 더 많이 분류하면서 생긴 상충관계다.',
           '예를 들어 SPPB 분류 RF는 기존 4 → 조정 4구간에서 적중률 47.09 → 47.80%, 구간별 평균 31.47 → 32.75%, 양끝 오판 1.83 → 1.74%로 좋아졌지만 저기능 놓침은 84.42 → 85.61%로 늘었다. TUG 순서형 RF는 기존 4 → 통합 3구간에서 적중률 53.03 → 58.09%, 양끝 오판 4.99 → 2.26%, 오경고 8.79 → 3.97%로 좋아졌지만 저기능 놓침이 55.63 → 73.14%로 늘었다.',
           '구간별 구분 능력도 부족했다. 대표 조건의 기존 4구간에서 TUG 순서형 RF의 실제 15–20초 구간 적중률은 0.48%였다. 3구간의 가운데 10–20초를 맞히는 비율은 45.99%로 높아지지만 더 넓고 다른 정답이다. SPPB 회귀 RF의 3구간은 저기능 구간 적중률이 0%였다. 전체 적중률만으로 점수화가 충분히 좋아졌다고 볼 수 없다.',
           '저기능 확률을 별도 경고로 사용해 내부 오경고 10% 기준을 적용하면 놓침은 대표 조건에서 여전히 약 49–54%였다. 경계를 조정하거나 가운데 구간을 합쳐도 이 값은 거의 바뀌지 않았다. 이번 입력 4개·공통 확률 모델·고정된 양끝 경계·후보 범위 안에서는 구간 재설계만으로 저기능 구분 문제를 해결하지 못했다. 확정 기능 등급의 채택 근거로 삼기에는 부족하다.',
           '보행으로 점수 구간을 더 잘 구분하기 위해 기존 4구간·가운데 경계를 조정한 4구간·가운데 두 구간을 합친 3구간을 비교했다. 사람·방문·보행 구간·초기값 5개를 유지하고, 경계와 판정 규칙 선택에는 바깥 평가 사람의 정답을 사용하지 않았다. 대표 조건은 사전에 정한 **30–60초 보행 구간 5개**다.',
           '## 유지한 조건과 실제 변경한 부분',
           'SPPB 2,174명·7,356방문, TUG 1,939명·5,688방문이다. SPPB 저기능 기록이 있는 사람은 195명, TUG는 290명이다. 같은 사람의 방문을 같은 바깥 분할에 묶고 바깥 5분할·안쪽 3분할과 초기값 20261030·20262030·20263030·20264030·20265030을 유지했다. [4,10), [30,60)초 구간을 각각 1·3·5개 사용하는 여섯 조건과 draw0의 구간 선택을 그대로 사용했다. 입력은 보행 빈도·주기 시간·속도·보폭의 중앙값 4개다.',
           'SPPB는 0–3점과 10–12점, TUG는 10초 이하와 20초 초과를 고정했다. 양끝 오판과 저기능 놓침이 분할마다 같은 뜻을 갖게 하기 위한 조건이다. SPPB 가운데 경계는 4.5·5.5·6.5·7.5·8.5점, TUG는 11–19초의 정수 경계에서 선택했다. SPPB의 소수 경계는 예를 들어 6.5일 때 정수 점수 4–6과 7–9를 나눈다. 3구간은 SPPB 0–3 / 4–9 / 10–12점, TUG ≤10 / (10,20] / >20초다.',
           '회귀 RF는 이전 초기값 5개의 학습 모델·안쪽 예측을 재사용하고 각 분할에 맞는 확률 변환기를 학습했다. 분류 RF·순서형 RF는 모든 후보 경계에 공통으로 쓸 세부 확률을 학습한 뒤 합산한다. 같은 모델의 확률을 분할만 달리 묶어 비교하므로 경계 변경 효과를 볼 수 있다. 후보마다 별개의 숲을 재학습하는 실험은 아니다. RF 두 모델은 128개 트리·최소 잎 표본 20개를 유지한다.',
           '**이번 기존 4구간 비교군과 이전 실험의 분류 RF·순서형 RF는 학습 출력이 다르다.** 이번에는 세부 확률을 합산한 공통 모델을 사용한다. 따라서 세 분할의 주 비교는 이번 공통 모델 안에서만 한다. 이전에 네 구간을 직접 학습한 결과는 별도 대조 표에 보존했다. 회귀 RF의 기존 4구간 확률은 이전 결과와 재현 일치를 검산했다.',
           '## 선택 방법',
           '먼저 각 후보의 학습자료 안쪽 평가 확률을 만들고 온도를 보정했다. 한 사람의 모든 방문을 같은 내부 분할에 두었다. 각 구간이 모든 내부 학습 분할에서 독립 참여자 40명 이상, 바깥 학습자료에서 사람 가중 비중 3% 이상일 때만 경계 후보로 인정했다. 평가 사람을 보고 후보를 제외하지 않았다.',
           '최댓값 구간, 기대 구간 거리 최소, 기대 거리 제곱 최소, 거리와 저기능 놓침·오경고 비용을 조합한 총 11개 규칙을 사전에 고정했다. 내부 구간별 평균 적중률을 가장 높이되 **양끝 오판·저기능 놓침·오경고가 모두 나빠지지 않는 규칙**을 골랐다. 기존 4구간은 자체 최댓값 규칙을 기준으로, 조정 4구간은 기존 4구간에서 선택된 규칙을 기준으로 제한했다. 3구간은 자체 최댓값 규칙을 기준으로 제한했다. 조건을 만족하는 개선안이 없으면 비교 기준으로 돌아간다.',
           '경계·규칙·보정값·경고 문턱은 학습 내부에서만 선택했다. 안쪽 지표는 선택에 재사용되므로 최종 성능으로 보고하지 않는다. 여기의 성능은 바깥 평가 예측이다. 안쪽에서 악화 방지 조건을 만족해도 바깥 평가에서 악화될 수 있으며 그 결과도 그대로 보고한다.',
           '## 대표 조건: 선택한 규칙의 전체 성능',
           '모든 수치는 초기값 5개의 평균이다. 적중률은 사람별 같은 가중치로 계산했다. 구간별 평균 적중률은 실제 각 구간의 적중률을 같은 비중으로 평균한다. 양끝 오판은 두 끝 구간을 서로 바꾼 오류다. 놓침은 실제 저기능 중 저기능으로 선택하지 않은 비율, 오경고는 실제 비저기능 중 저기능으로 선택한 비율이다. 분할마다 가운데 정답의 정의가 달라지므로 적중률 상승만으로 예측 능력이 개선됐다고 판단하지 않는다.']
    for target in ['sppb','tug']:
        rows=[]
        for family,scheme in itertools.product(FAMILIES,SCHEMES):
            g=primary[primary.target.eq(target)&primary.family.eq(family)&primary.scheme.eq(scheme)&primary.decision.eq('selected')]
            base=primary[primary.target.eq(target)&primary.family.eq(family)&primary.scheme.eq(scheme)&primary.decision.eq('prior_class')]
            rows.append([NAMES[family],SCHEMES[scheme],pct(g.accuracy.mean()),pct(base.accuracy.mean()),pct(g.balanced_accuracy.mean()),pct(g.extreme.mean()),pct(g['miss'].mean()),pct(g.fpr.mean())])
        parts.extend(['### '+target.upper(),table(['모델','분할','적중률 ↑','빈도 기준선','구간별 평균 ↑','양끝 오판 ↓','저기능 놓침 ↓','오경고 ↓'],rows)])
    parts.extend(['## 구간별 적중률과 판정 규칙 효과','조정 4구간의 가운데 경계는 각 바깥 학습 분할에서 독립적으로 선택된다. 가운데 구간의 적중률은 동일한 고정 점수 범위 하나의 결과가 아니다. 양끝 구간은 모든 비교에서 동일하다.'])
    for target in ['sppb','tug']:
        rows=[]
        for family,scheme in itertools.product(FAMILIES,SCHEMES):
            for decision in ['mode','selected']:
                g=primary[primary.target.eq(target)&primary.family.eq(family)&primary.scheme.eq(scheme)&primary.decision.eq(decision)];k=int(g.classes.iloc[0])
                rows.append([NAMES[family],SCHEMES[scheme],'최댓값' if decision=='mode' else '내부 선택',' / '.join(pct(g[f'recall{j}'].mean()) for j in range(k)),pct(g.low_to_best.mean()),f'{g.prediction_middle_width.mean():.2f}'])
        parts.extend(['### '+target.upper(),table(['모델','분할','판정','각 구간 적중률: 점수·시간 오름차순','저기능→최상 구간 ↓','선택한 중간 구간 평균 폭'],rows)])
    parts.extend(['## 경계·규칙 선택 빈도','대표 조건은 바깥 5분할×초기값 5개로 모델당 25번 선택한다. 이는 25개의 독립 표본 집단을 뜻하지 않는다.'])
    rows=[]
    for (target,family,scheme),g in sp.groupby(['target','family','scheme']):
        bands='; '.join(f'{bandname(target,json.loads(c))}: {n}회' for c,n in g.cuts.value_counts().items())
        rules='; '.join(f'{rulelabel(c)}: {n}회' for c,n in g.rule.value_counts().items());rows.append([target,NAMES[family],SCHEMES[scheme],bands,rules])
    parts.append(table(['목표','모델','분할','선택한 경계','선택한 규칙'],rows))
    parts.extend(['## 동일한 저기능 경고 정책','구간 하나를 선택하는 규칙과 별도로 저기능 확률의 경고 문턱을 내부에서 골랐다. FPR10은 내부 오경고 10% 이내, sens80은 내부 저기능 감지율 80% 이상 조건이다. 실제 평가 비율은 별도로 측정했다.'])
    for target in ['sppb','tug']:
        rows=[]
        for family,scheme,policy in itertools.product(FAMILIES,SCHEMES,['fpr10','sens80']):
            g=op[op.target.eq(target)&op.family.eq(family)&op.scheme.eq(scheme)&op.policy.eq(policy)]
            rows.append([NAMES[family],SCHEMES[scheme],policy,pct(g['miss'].mean()),pct(g.fpr.mean())])
        parts.extend(['### '+target.upper(),table(['모델','분할','정책','놓침 ↓','오경고 ↓'],rows)])
    parts.extend(['## 여섯 보행 조건과 초기값 안정성','숫자는 선택한 규칙의 구간별 평균 적중률이다. 서로 다른 정답 분할의 난이도 차이를 포함한다. 저기능 놓침·오경고·양끝 오판은 같은 의미로 비교할 수 있다.'])
    rows=[]
    for target,band,n,family in itertools.product(['sppb','tug'],['short','long'],[1,3,5],FAMILIES):
        g=m[m.target.eq(target)&m.band.eq(band)&m.n.eq(n)&m.family.eq(family)&m.decision.eq('selected')]
        wide=g.pivot(index='seed',columns='scheme',values='balanced_accuracy');rows.append([target,band,n,NAMES[family]]+[pct(wide[c].mean()) for c in SCHEMES]+[f'{int((wide.adjusted4>wide.fixed4).sum())}/5'])
    parts.append(table(['목표','길이','구간 수','모델','기존 4','조정 4','통합 3','조정 4 적중률 개선 초기값'],rows))
    paired=json.loads((HERE/'paired-bootstrap.json').read_text())['comparisons'];rows=[]
    for r in paired:
        if r['reference']!='fixed4_selected' or r['new'] not in ['adjusted4_selected','merged3_selected'] or r['metric'] not in ['extreme','miss','fpr']:continue
        rows.append([r['target'],NAMES[r['family']],r['new'],r['metric'],f"{100*r['difference']:+.2f}",f"[{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]",f"{r['lower_error_seeds']}/5"])
    parts.extend(['### 대표 조건에서 기존 4구간 대비 오류 차이','새 분할−기존 4구간 차이이며 단위는 %p다. 음수가 개선이다. 사람별로 초기값 5개의 차이를 평균한 뒤 사람을 2,000회 재표집했다. 고정한 모델에 조건을 둔 95% 구간이며 재학습·다중 비교·새 집단 불확실성을 전부 포함하지 않는다.',table(['목표','모델','분할','오류','차이 %p','조건부 95% 구간','개선 초기값'],rows)])
    parts.extend(['## 확률·검산·저장','구간 수와 정답이 달라지면 Brier를 직접 비교할 수 없다. probabilities-by-seed.csv에 해당 분할의 빈도 기준선 대비 Brier 개선과 고정된 저기능의 Brier를 별도로 기록했다. 이전 직접 4구간 학습 모델은 historical-four-class-models.csv에서 확인한다.',
                  f"검산 상태 **{v['status']}**. {v['fold_seed_ledgers']}개 결과 기록, {v['final_model_replays']}개 저장 모델을 재실행했고 사전에 정한 대표 셀의 내부·최종 학습 {v['selected_refits']}회를 다시 학습했다. 최대 예측 재현 차이는 {v['max_prediction_error']:.3g}, 독립 지표 재계산 차이는 {v['independent_metric_max_error']:.3g}다. 절대 허용오차 {v['absolute_tolerance']:.0e}를 유지했다.",
                  f'- 작은 산출물: `{HERE}`\n- 모델·대량 예측·내부 배열·개인별 오차: `{STORE}`',
                  '실행 순서: run.py prepare → run.py pilot → run.py main → analyze.py → verify.py → report.py. protocol.json에 고정 설정과 원본 지문을 보존했다. 검산 개발 중 경고 메타데이터의 비교 방식과 이전 실험의 동명 분석 모듈을 불러온 경로 오류를 수정하고 재실행했다. 기록은 verification-development-note.json에 있다. 학습 설정·선택 기준·수치 허용오차는 변경하지 않았다.',
                  '## 한계','센서 보행 특징의 연구용 예측이다. 이 구간을 검증된 임상 위험 등급으로 취급하지 않는다. 보행 영상에서의 측정 오차를 검증하지 않았으며, 기존 개발용 사람 분할을 반복 사용했으므로 독립 외부 평가가 아니다. 세 구간은 가운데 점수의 세밀함을 포기한 결과이며 높은 적중률만으로 더 좋은 점수 추정이라고 결론 내리지 않는다.'])
    (HERE/'README.md').write_text('\n\n'.join(parts)+'\n',encoding='utf8');print('REPORT_DONE',flush=True)
if __name__=='__main__':run()
