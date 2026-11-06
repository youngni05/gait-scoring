"""Check summaries directly against prediction rows and write the experiment record."""
from common import *

def main():
    metrics=pd.read_csv(HERE/'metrics.csv');persons=pd.read_csv(HERE/'person-results.csv')
    prep=json.loads((HERE/'preparation.json').read_text());verification=json.loads((HERE/'verification.json').read_text())
    assert verification['passed'];comparisons=json.loads((HERE/'comparisons.json').read_text())
    checked=0
    for r in metrics.itertuples():
        p=pd.read_csv(HERE/r.dataset/'predictions.csv');p=p[(p.model==r.model)&(p.view==r.view)]
        if r.group=='ge20':p=p[p.target>=20]
        if r.group=='lt20':p=p[p.target<20]
        err=p.prediction.to_numpy()-p.target.to_numpy()
        assert p.id.nunique()==r.n
        for actual,expected in [(abs(err).mean(),r.mae),(np.maximum(-err,0).mean(),r.under),(np.maximum(err,0).mean(),r.over)]:
            assert abs(actual-expected)<1e-10
        checked+=1
    ranges=[]
    for name in prep['cohorts']:
        pred=pd.read_csv(HERE/name/'predictions.csv')
        for (model,view),g in pred.groupby(['model','view']):
            ranges.append({'dataset':name,'model':model,'view':view,'minimum':float(g.prediction.min()),
                           'maximum':float(g.prediction.max()),'nonpositive_prediction_rows':int((g.prediction<=0).sum())})
    save('prediction-ranges.json',ranges)
    import sklearn, scipy, platform
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                             'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__})
    def row(name,view,group='all',family='RF'):
        return metrics[(metrics.dataset==name)&(metrics.view==view)&(metrics.group==group)&(metrics.model==family)].iloc[0]
    def comp(name,family='RF',group='all'):
        return next(c for c in comparisons if c['dataset']==name and c['model']==family and c['group']==group)
    names=['STS_common6','STS_common3','STS_expanded3','TUG_common6','TUG_common3','TUG_expanded3']
    labels=['일어서기·공통 6초','일어서기·공통 3초','일어서기·확대 3초','TUG·공통 6초','TUG·공통 3초','TUG·확대 3초']
    lines=['# 단순 보행 패턴으로 실제 5회 일어서기·TUG 시간 예측 — 2026-10-31','',
        f"**주분석인 179명·6초 보행에서 일어서기 MAE는 속도만 {row('STS_common6','speed').mae:.3f}초에서 패턴 추가 {row('STS_common6','pattern').mae:.3f}초로 소폭 줄었다. 그러나 차이의 조건부 95% 구간은 0을 포함해 뚜렷한 개선으로 확정하기 어렵다. TUG의 전체 오차와 긴 사람의 과소예측은 주모델 RF에서 개선되지 않았다.**",'',
        '실행과 저장 모델 재추론 검산을 완료했다. 아래 결과는 기존 개발 자료에서의 탐색 비교이며, 단일 카메라 측정 정확도나 임상 검사 대체를 검증한 결과가 아니다.','',
        '## 질문과 비교 조건','',
        '걷는 속도 외에 무릎 움직임의 반복 주기·좌우 차이·관절 가동범위·몸통 흔들림을 추가하면, 실제 5회 일어서기 검사 시간을 더 정확히 예측할 수 있는지 비교했다. TUG를 별도 보조 정답으로 평가했다. 의자 동작·회전·준비시간·다른 임상 점수·진단·나이는 입력하지 않았다.','',
        '- 주분석은 6초 보행이 있는 일어서기 정답 보유자 179명이다. 같은 179명의 같은 시작점 3초와 276명의 확대 3초를 보조 분석했다. TUG는 각각 195명·195명·297명이며, 일어서기 정답 보유자만으로 제한하지 않았다.',
        '- sub-283(42.59초 TUG)은 모든 학습·내부 검증·평가에서 제외했다. 기존 사람별 방문·보행 파일·구간 시작점을 유지했다.',
        '- 기존 속도 안정성 실험의 사람 분할 5가지와 각 학습 집단 내부 3분할을 해당 사람 집합으로 제한했다. 입력 비교 사이의 학습·평가 사람과 관찰시간은 동일하다. 공통 3초·6초의 분할도 동일하다.',
        '- 주모델 RF는 기존과 같은 128개 나무, 잎 최소 사람 수 2/5를 학습 집단 내부에서 선택했다. 학습 난수 3개를 모두 평가했다. Ridge는 같은 입력에서 보조 비교했으며 규제값 1/10/100을 내부 선택했다. 좋은 분할이나 난수만 고르지 않았다.',
        '- 속도 입력은 평균·표준편차·인접 변화량 평균 3개다. 패턴 입력은 같은 속도 3개에 사전 고정한 패턴 14개를 추가했다. 결측치는 각 학습 집단의 중앙값과 결측 표시로 처리하며 평가 사람의 값이나 정답은 사용하지 않았다.',
        '- 20초 이상 집단은 오차 분석용 구분이며 임상 판정 기준이 아니다. 5회 일어서기 초를 SPPB 의자 점수로 환산하지 않았다.','',
        '## 전체 오차','',
        '평균 절대오차(MAE), 단위 초. 반복마다 사람별 오차를 먼저 계산한 뒤 평균했다. 예측을 먼저 평균해 성능을 좋게 만들지 않았다. 대표값은 각 학습 집단의 중앙값만 사용하는 기준선이다.','',
        '| 정답·관찰 조건 | 사람 수 | 대표값 | 속도 RF | 패턴 추가 RF | 차이(추가−속도) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        a,b=row(name,'speed'),row(name,'pattern');base=row(name,'none',family='Median')
        lines.append(f'| {label} | {a.n} | {base.mae:.3f} | {a.mae:.3f} | {b.mae:.3f} | {b.mae-a.mae:+.3f} |')
    lines+=['','각 행 안의 두 모델을 비교한다. 공통 집단과 확대 집단의 오차 차이는 관찰시간 효과가 아니다. 같은 사람 3초→6초 비교는 `duration-comparisons.json`에 별도로 저장했다.','',
            '## 긴 검사 시간의 오차','',
            '| 정답·관찰 조건 | 20초 이상 사람 수 | 속도 MAE | 추가 MAE | 속도 과소예측 | 추가 과소예측 | 속도 과대예측 | 추가 과대예측 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in zip(names,labels):
        a,b=row(name,'speed','ge20'),row(name,'pattern','ge20')
        lines.append(f'| {label} | {a.n} | {a.mae:.3f} | {b.mae:.3f} | {a.under:.3f} | {b.under:.3f} | {a.over:.3f} | {b.over:.3f} |')
    lines+=['','과소예측은 max(실제−예측, 0), 과대예측은 max(예측−실제, 0)을 해당 집단의 모든 사람·반복에서 평균한 값이다. 두 값의 합이 MAE이며 양방향 오차를 상쇄하지 않는다.','',
            '## 차이의 안정성과 해석','']
    for name in ['STS_common6','TUG_common6']:
        c=comp(name);ci=c['conditional_ci95'];a,b=row(name,'speed',family='Ridge'),row(name,'pattern',family='Ridge')
        lines.append(f"- {name}: RF 오차 차이 {c['mae_delta_pattern_minus_speed']:+.3f}초, 조건부 95% 구간 [{ci[0]:+.3f}, {ci[1]:+.3f}]. 반복 {c['paired_runs']}개 중 {c['paired_run_wins']}개에서 추가 입력이 낮은 오차를 보였다. 사람별로는 {c['n']}명 중 {c['people_improved']}명이 개선됐다. Ridge 전체 MAE는 {a.mae:.3f}→{b.mae:.3f}초다.")
        a,b=row(name,'speed','ge20','Ridge'),row(name,'pattern','ge20','Ridge');c=comp(name,'Ridge','ge20')
        ci=c['conditional_ci95']
        lines.append(f"  Ridge의 20초 이상 집단에서는 MAE {a.mae:.3f}→{b.mae:.3f}초, 과소예측 {a.under:.3f}→{b.under:.3f}초였다. {c['n']}명 중 {c['people_improved']}명이 개선됐으며 MAE 차이의 조건부 구간은 [{ci[0]:+.3f}, {ci[1]:+.3f}]초다.")
    lines+=['','긴 TUG에서 Ridge는 패턴을 추가하면 나아졌으나, 추가 후에도 MAE 9.911초로 속도만 사용하는 RF의 9.683초보다 낮지 않았다. 따라서 Ridge의 약점을 일부 보완하는 결과와 기존 유력 기준 모델을 넘어서는 개선을 구분해야 한다.']
    lines+=['','구간은 사람별 반복 평균 오차를 짝지어 10,000회 재추출한 값이다. 같은 사람을 여러 번 평가했으므로 반복 15개를 독립 실험 15개로 보지 않는다. 구간은 저장된 모델에 조건부이며 재학습·새 사람 모집·여러 분석 중 선택의 불확실성을 모두 포함하지 않는다. Ridge나 일부 집단에서 좋은 결과가 나왔다고 주분석을 바꾸지 않았다.','',
            '일어서기 주분석에서는 대표값만 사용한 MAE가 5.282초이고 패턴 RF는 4.930초다. 대표값 대비 차이의 조건부 95% 구간도 [-0.865, +0.159]초로 0을 포함한다. 따라서 보행만으로 일어서기 시간을 유용하게 예측한다는 주장을 이번 결과로 확정할 수 없다. 같은 179명의 패턴 RF를 3초에서 6초로 늘린 오차 차이는 -0.320초이나 구간 [-0.628, +0.001]초로 이 비교 역시 불확실성이 남는다.','',
            '## 입력 품질과 정답 누락의 한계','',
            '- 무릎 각도를 5시점 평균한 뒤 굽힘 봉우리 사이를 주기 후보로 정의했다. 최소 봉우리 간격 0.4초·돌출 크기 8도, 완전한 후보 주기 0.6~2.5초를 요구했다. 양쪽 다리에서 적어도 한 후보가 있어야 주기 특징을 사용하고, 변동계수에는 다리마다 두 후보 이상을 요구했다.',
            '- 17개 관절 각도 중 1초 환산 1,000도 이상의 급변 주변 ±3시점은 제외했다. 몸통·골반 범위에는 해당 창의 80% 이상이 안정적이어야 한다. 이런 기준은 계산 규칙이지 실제 관절 추정의 임상 정확도 보증이 아니다.',
            '- 6초 일어서기 집단 179명 중 165명에게 양쪽 주기 후보가 있었고, 변동계수는 150명에게 계산됐다. 같은 사람의 3초에서는 각각 138명·17명이다. 결측 사람을 제거해 모델 비교 대상을 바꾸지 않았다.',
            '- 원본 각도 파형 6사례를 그림으로 점검했다. 급격한 흔들림 때문에 후보가 모두 제외되는 경우도 있고, 작은 추가 봉우리가 후보로 남기도 한다. 수동 발 접촉 정답이나 원본 RGB 검증이 없으므로 정확한 보행 주기·대칭성 측정이라고 주장하지 않는다. `quality-examples.png`는 이러한 한계를 보여주는 예시이며 전수 수동 검증이 아니다.',
            '- 패턴 추가 모델에는 특징값과 함께 결측 표시도 들어간다. 따라서 두 모델의 차이는 이 묶음의 추가 효과이며, 정확한 보행 역학 정보만의 효과를 분리해 입증한 것은 아니다.',
            '- sub-283 제외 후 297명 중 일어서기 정답이 없는 사람은 21명이다. 긴 TUG 8명 중 6명도 이 집단에 포함된다. 누락 이유는 확인되지 않았으며 0점이나 수행 불가로 바꾸지 않았다. 따라서 일어서기 결과의 적용 범위는 실제 기록이 있는 사람으로 제한된다.',
            '- 자료는 여러 카메라에서 복원한 OpenCap 관절 시계열이다. 일반 고령자 집단·한 대의 카메라·새 기관에 대한 독립 검증이 아니다. 기존 개발 사람을 반복 분석한 결과다.','',
            'Ridge는 출력 범위를 제한하지 않아 일부 반복에서 음수 예측도 발생했다. 일어서기 공통 6초 속도 모델 1건, TUG 확대 3초 속도 모델 5건·패턴 모델 1건이다. 이 예측을 사후 잘라내지 않고 오차에 포함했다. 모든 RF 예측은 양수였다. 상세 범위는 `prediction-ranges.json`에 있다.','',
            '## 재현과 검산','',
            f"원본 파일 {verification['raw_files_checked']}개의 SHA-256과 임상 기록 연결을 확인했다. 원본 좌표를 다른 읽기·속도 계산 경로로 재구성해 {verification['windows_reconstructed']}개 입력 창을 대조했다. 모든 사람 분리와 공통 시간 비교를 검사했고, 저장한 최종 모델 {verification['final_models_replayed']}개를 재추론했다. 최대 재예측 차이는 {verification['max_prediction_delta']:.3g}초였다. 내부 모델 선택 {verification['inner_fit_selections_checked']}회에 해당하는 저장 검증 예측·오차도 다시 계산했다. 내부 모델 전체를 별도로 재학습했다는 뜻은 아니다.",
            '',f'집계표 {checked}행을 예측 원자료에서 다시 계산해 일치함을 확인했다. GPU나 새 패키지 없이 기존 로컬 CPU 환경을 사용했다.',
            '', '최초 재검산에서 검산 측 평균 계산이 float32 정밀도로 누적돼 StandardScaler의 float64 누적값과 약 1e-7 차이가 났다. 검산의 기준 계산을 float64로 바로잡고 평균·분산을 다시 검사했다. 학습 데이터·특징·모델·예측·허용 기준을 바꿔 통과시킨 것은 아니다.',
            '', '핵심 코드 흐름은 `prepare.py`에서 보행 창·실제 정답·분할을 고정하고, `common.py::patterns`에서 해당 창 안의 특징을 계산한 뒤, `run.py`에서 같은 사람 분할로 두 입력을 비교하는 순서다. `verify.py`는 원본 입력과 저장 모델을 재검산한다.',
            '', '산출물: `protocol.json`, `preparation.json`, `metrics.csv`, `comparisons.json`, `duration-comparisons.json`, `person-results.csv`, 각 조건의 `predictions.csv`·`selections.json`·모델, `input-verification.json`, `verification.json`, `results.png`.','']
    (HERE/'STATUS.md').write_text('\n'.join(lines),encoding='utf8')
    save('summary-verification.json',{'metric_rows_recalculated':checked,'passed':True})
    manifest={p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='artifact-manifest.json'}
    save('artifact-manifest.json',manifest)
    print('REPORT_VERIFIED',checked,'rows;',len(manifest),'files',flush=True)

if __name__=='__main__':main()
