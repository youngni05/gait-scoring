"""Build the research result page from verified numerical artifacts."""
from common import *
os.environ['MPLCONFIGDIR'] = str(HERE/'mplconfig')
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = Path('C:/Windows/Fonts/malgun.ttf')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams['axes.unicode_minus'] = False

TITLES = {'mobilise_sppb':'SPPB 총점·주간 센서','mobilise_tug':'TUG·주간 센서','mobilise_nonwalk':'SPPB 균형＋일어서기·주간 센서',
          'opencap_tug':'TUG·OpenCap 3초','opencap_patterns_tug':'TUG·OpenCap 주기 후보 6초','older44_sppb':'SPPB·Vicon 제공처 보행 측정값'}
GROUPS = {'speed':'속도','stride_length':'보폭','cadence':'걸음 빈도','cycle_duration':'보행 주기',
          'pelvis_angles':'골반 각도','hip_angles':'고관절 각도','knee_angles':'무릎 각도','ankle_angles':'발목·거골하 관절각','lumbar_angles':'몸통 각도',
          'cycle_rhythm':'주기·변동·좌우 차이','leg_ROM_asymmetry':'다리 관절 움직임 범위·좌우 차이','hip_knee_coordination':'고관절·무릎 협응','pelvis_lumbar_ROM':'골반·몸통 움직임 범위',
          'step_stride_time':'걸음·보행 주기 시간','gait_events':'보행 사건 시점','support_time':'한발·두발 지지시간','step_geometry':'걸음길이·보폭·너비','limp_index':'제공처 Limp Index'}
MEASURES = {'ws_30_p90_w':'30초 초과 보행 속도의 90백분위 주간 요약', 'ws_10_p90_w':'10초 초과 보행 속도의 90백분위 주간 요약',
    'ws_1030_avg_w':'10~30초 보행 속도 대표값','ws_30_avg_w':'30초 초과 보행 속도 대표값','ws_30_var_w':'30초 초과 보행 속도 변동계수',
    'strlen_1030_avg_w':'10~30초 보폭 대표값','strlen_30_avg_w':'30초 초과 보폭 대표값','strlen_30_var_w':'30초 초과 보폭 변동계수',
    'cadence_all_avg_w':'전체 보행 걸음 빈도 대표값','cadence_all_var_w':'전체 보행 걸음 빈도 변동계수',
    'cadence_30_avg_w':'30초 초과 보행 걸음 빈도 대표값','cadence_30_p90_w':'30초 초과 걸음 빈도 90백분위',
    'strdur_all_avg_w':'전체 보행 주기 대표값','strdur_all_var_w':'전체 보행 주기 변동계수','strdur_30_avg_w':'30초 초과 보행 주기 대표값'}


def display(variant):
    if '|single:' in variant:
        model,feature = variant.split('|single:',1)
        return MEASURES.get(feature,feature)+' ('+model+')'
    if '|family:' in variant:
        model,group = variant.split('|family:',1)
        return '＋'.join(GROUPS.get(g,g) for g in group.split('+'))+' ('+model+')'
    return variant


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    summaries = {}
    for name in DATASETS:
        verified = json.loads((HERE/f'{name}-verification.json').read_text())
        assert verified['passed']
        summaries[name] = json.loads((HERE/f'{name}-summary.json').read_text(encoding='utf-8'))
    sections = ['# SPPB·TUG 예측에 유용한 보행 측정값 비교',
        '**현재 자료에서 가장 일관된 소수 조합은 속도 상위값과 보행 주기 변동계수다.** 큰 주간 센서 자료의 SPPB·TUG 두 값 탐색에서 이 조합이 각각 15개 평가 묶음 모두의 학습 자료에서 선택됐다. 단일값보다 오차가 줄었고, 더 많은 입력을 사용하면 추가 이득도 있었다. 짧은 OpenCap 자료에서는 속도 요약에 관절 정보를 추가한 이득이 뚜렷하지 않았으며, 44명 정밀 보행 자료에서는 최우수 지표 선택이 불안정했다.',
        '2026-10-31 사용자 요청에 따라 실제 확보 자료에서 비교했다. 결론의 범위는 아래 후보·자료·모델 안이다. 새 사람의 예측을 평가하되, 이미 여러 탐색에 사용한 개발자료이므로 독립 임상 검증으로 해석하지 않는다.',
        '## 큰 센서 자료의 결과',
        'SPPB 2,232명과 TUG 1,980명은 정답별로 첫 적격 방문 하나를 사용했다. 표의 단일값·조합 선택은 매번 학습 사람 안에서 수행했다. 평가 사람의 정답으로 가장 좋은 조합을 고른 값이 아니다. MAE는 실제 점수·시간과 예측의 차이를 절댓값으로 평균한 것으로 작을수록 좋다.',
        '여기서 보폭(stride length)은 같은 발이 다시 닿을 때까지 한 보행 주기에 이동한 거리다. 보행 주기는 그 한 주기에 걸린 시간이며, 걸음 빈도는 1분당 걸음 수다. 이 값들은 서로 관련되므로 개별 중요도와 조합의 추가 효과를 구분한다.']
    rows = []
    for name in ['mobilise_sppb','mobilise_tug','mobilise_nonwalk']:
        s = summaries[name]
        lookup = {r['variant']:r for r in s['selection_policy_metrics']}
        rows.append([TITLES[name],s['people'],f"{s['baseline_MAE']:.3f}",f"{lookup['selected_single']['MAE']:.3f}",f"{lookup['selected_family']['MAE']:.3f}"])
    sections.append(table(['정답·입력','독립 사람','학습 집단 중앙값','학습에서 고른 단일값','학습에서 고른 묶음 조합'],rows))
    sections += ['SPPB 총점과 균형＋일어서기는 점 단위, TUG는 초 단위다. 균형＋일어서기 정답은 제공된 두 구성점수의 합(0~8)이며 보행 구성점수는 제외했다. 구성점수 하나가 없는 1명은 이 보조 정답에서만 제외했고 뒤 방문으로 대체하지 않았다.',
        '단일값 비교에서는 SPPB·TUG 모두 `ws_30_p90_w`가 15개 바깥 평가 묶음 모두의 학습 자료에서 선택됐다. 이는 30초를 넘는 보행들의 속도 상위 수준을 나타내는 90백분위 주간 요약값이다. 한 번의 30초 영상에서 측정한 평균 속도나 최대 속도와 같지 않다. 주간 요약에 대한 이 결과를 3초 영상으로 바로 옮기지 않는다.',
        '## 조합의 추가 효과와 선택 안정성']
    rows = []
    for name in ['mobilise_sppb','mobilise_tug','mobilise_nonwalk']:
        s = summaries[name]
        c = s['policy_contrasts']['selected_family_minus_selected_single']
        selected = next(r for r in s['selection_choices'] if r['policy']=='selected_family')['choices'][0]
        rows.append([TITLES[name],display(selected['variant']),f"{selected['outer_cells']}/15",f"{c['difference']:+.3f}",f"[{c['conditional95'][0]:+.3f}, {c['conditional95'][1]:+.3f}]"])
    sections.append(table(['정답','학습에서 가장 자주 고른 묶음','선택 횟수','조합−단일값 MAE','조건부 95% 구간'],rows))
    sections += ['음수는 조합의 오차 감소다. 구간은 같은 사람의 반복 평균 오차를 짝지어 2,000회 다시 뽑아 계산했다. 이미 학습한 모델과 관측 사람에 조건부인 구간이다. 전체 재학습 변동·새 기관 불확실성·여러 탐색 비교의 불확실성을 모두 포함하지 않는다. 15개 평가는 같은 사람을 다시 나눈 것이며 15개 독립 실험의 승리 횟수가 아니다.',
        '속도·보폭·걸음 빈도·보행 주기는 서로 관련된다. 특정 묶음이 자주 선택됐다는 사실만으로 각 값이 독립적인 원인이라고 해석하지 않는다. 전체 입력에서 각 묶음을 하나씩 뺀 비교도 별도 표에 남겼다.',
        '## 적은 수의 측정값으로 만든 조합']
    sparse_summaries = {}
    sparse_rows = []
    for name in ['mobilise_sppb','mobilise_tug']:
        verified = json.loads((HERE/f'{name}-sparse-rf-verification.json').read_text())
        assert verified['passed']
        s = json.loads((HERE/f'{name}-sparse-rf-summary.json').read_text(encoding='utf-8'))
        sparse_summaries[name] = s
        metrics = {r['variant']:r for r in s['metrics']}
        pair = s['choices']['2'][0]
        text = '＋'.join(MEASURES.get(x,x) for x in pair['features'].split(' + '))
        sparse_rows.append([TITLES[name],text,f"{pair['outer_cells']}/15",f"{metrics['RF_forward_1']['MAE']:.3f}",f"{metrics['RF_forward_2']['MAE']:.3f}",f"{metrics['RF_selected_up_to3']['MAE']:.3f}"])
    sections.append(table(['정답','가장 자주 고른 두 값','선택 횟수','1개 MAE','2개 MAE','학습에서 고른 최대 3개 MAE'],sparse_rows))
    thirds = []
    for name,s in sparse_summaries.items():
        choice = s['choices']['3'][0]
        description = '＋'.join(MEASURES.get(x,x) for x in choice['features'].split(' + '))
        thirds.append([TITLES[name],description,f"{choice['outer_cells']}/15"])
    sections.append(table(['정답','세 값 탐색에서 가장 자주 선택된 조합','선택 횟수'],thirds))
    sections += ['이 표는 첫 비교 뒤 추가한 RF 순차 선택 탐색이다. 1개 값은 학습 자료의 내부 오차로 고르고, 거기에 하나씩 추가해 최대 3개를 비교했다. 새 평가 정답을 이용해 값을 선택하지 않았지만 같은 개발자료의 후속 탐색이며 독립 확인으로 해석하지 않는다. 가능한 모든 두 값·세 값 조합을 전수 조사한 것도 아니다. 전체 입력은 첫 비교와 동일하다.',
        '보행 주기 변동계수는 제공자의 일상 보행 집계값이다. 영상의 발 접촉 간격에서 직접 구한 걸음마다의 변동과 동일한 값으로 간주하지 않는다. 세 번째 값의 선택 빈도와 조건부 오차 차이는 `*-sparse-rf-summary.json`에 남겼다.',
        '## 관절과 정밀 보행 자료의 보조 비교']
    rows = []
    for name in ['opencap_tug','opencap_patterns_tug','older44_sppb']:
        s = summaries[name]
        lookup = {r['variant']:r for r in s['selection_policy_metrics']}
        single = s['top_fixed_single_exploratory'][0]
        family = s['top_fixed_families_exploratory'][0]
        rows.append([TITLES[name],s['people'],f"{s['baseline_MAE']:.3f}",display(single['variant'])+f" / {single['MAE']:.3f}",display(family['variant'])+f" / {family['MAE']:.3f}",f"{lookup['selected_any']['MAE']:.3f}"])
    sections.append(table(['자료','사람','중앙값 기준','고정 단일 후보 중 탐색상 최저 MAE','고정 묶음 중 탐색상 최저 MAE','학습에서 선택하는 절차 MAE'],rows))
    references = json.loads((HERE/'opencap-speed-reference-comparisons.json').read_text())
    direct = []
    for name in ['opencap_tug','opencap_patterns_tug']:
        d,meta = load(name)
        ref = references[name]
        full = ref['comparisons']['RF|family:'+'+'.join(meta['groups'])]
        delta = full['minus_RF_speed']
        direct.append([TITLES[name],f"{ref['RF_speed_MAE']:.3f}",f"{full['MAE']:.3f}",f"{delta['difference']:+.3f}",f"[{delta['conditional95'][0]:+.3f}, {delta['conditional95'][1]:+.3f}]"])
    sections += ['### 속도 요약을 맞춘 관절 추가 비교',
        '관절의 추가 효과는 평균 속도 하나와 전체 입력을 비교하는 것만으로 판단하지 않는다. 속도의 여러 요약값을 이미 포함한 같은 RF 기준에 관절·패턴을 더한 결과를 짝지어 비교했다. 3초 기준은 골반 수평 속도와 수직 속도의 평균·표준편차·인접 변화량 6개다. 6초 기준은 수평 속도의 같은 요약 3개다.',
        table(['자료','RF 속도 요약 MAE','RF 전체 관절·패턴 MAE','전체−속도 MAE','조건부 95% 구간'],direct),
        '현재 두 자료에서는 많은 관절·주기 대리값을 추가해 속도 요약보다 개선됐다고 확인할 근거를 얻지 못했다. 생체역학적으로 관절 정보가 가치 없다는 뜻은 아니다. 입력 추정 오차, 관찰 시간, 표본 수, 사용한 모델·특징 정의의 영향을 분리하지 못했다. 이 비교는 저장 예측의 사후 탐색 대비이며 원래 학습·후보 선택을 바꾸지 않았다.']
    older = summaries['older44_sppb']
    older_policy = {r['variant']:r for r in older['selection_policy_metrics']}
    older_delta = older['paired_against_median']['selected_any']
    sections.append(f"44명 자료에서 전체 교차검증 결과를 보고 고른 고정 속도 후보의 MAE는 {older['top_fixed_single_exploratory'][0]['MAE']:.3f}점이었다. 그러나 매 학습 집단 안에서 단일값을 다시 고르는 절차는 {older_policy['selected_single']['MAE']:.3f}점으로 중앙값 기준 {older['baseline_MAE']:.3f}점보다 나빴다. 모든 후보에서 고르는 절차의 기준선 대비 차이도 {older_delta['difference']:+.3f}점, 조건부 95% 구간 [{older_delta['conditional95'][0]:+.3f}, {older_delta['conditional95'][1]:+.3f}]로 0을 포함했다. 이 작은 자료에서는 하나의 측정값·조합을 안정적인 최우수로 확정할 근거가 부족하다.")
    sections += ['위 고정 후보 순위는 전체 교차검증 결과를 보고 정리한 탐색 순위다. 마지막 열은 후보 선택까지 학습 안에서 다시 수행한 절차의 오차이므로, 두 열의 차이를 유지해서 읽어야 한다.',
        '- OpenCap 3초 자료는 골반·고관절·무릎·발목·몸통 각도와 수평·수직 속도의 평균, 표준편차, 인접 시점 변화량을 비교했다. 관절 각도는 여러 RGB 카메라의 복원 결과이며 단일 RGB에서의 측정 정확도를 검증한 것이 아니다.',
        '- OpenCap 6초 자료의 보행 주기는 무릎 굽힘 봉우리 사이의 대리값이다. 발 접촉 정답으로 확인한 값이 아니다. 이전 실험의 42.59초 TUG 사례 제외를 그대로 유지한 195명이며 298명 자료와 직접 성능 차이를 계산하지 않는다.',
        '- Vicon 44명은 제공처가 계산한 좌우 26개 보행 측정값을 사용했다. 매 보행의 좌우 평균과 절대 차이를 만들고 5회 보행을 평균했다. 보행 후 잘라낸 길이 자체는 입력에서 제외했다. 실제 정답은 5~12점으로 0~4점에 대한 근거가 없다. 고령자 자료이지만 표본이 작고 완전한 5회 보행의 결과이며 짧은 RGB 영상의 검증이 아니다.',
        '## 무엇을 어떻게 비교했는가',
        '각 자료에서 사람별 기록 하나를 만들었다. 한 사람의 반복 구간을 독립 정답으로 늘리지 않았다. 사람을 다섯 묶음으로 나눠 하나를 평가하고 나머지로 학습하는 과정을, 난수 17·41·89로 세 번 반복했다. 질환군이 있는 자료는 가능한 경우 질환 구성으로 층화했다. 모델 설정·결측치 대체·정규화와 조합 선택은 평가 사람을 제외한 학습 자료 안에서만 수행했다.',
        '학습 사람 안에서는 다시 세 묶음으로 나누어 평균 절대오차를 계산했다. 각 측정값 하나씩, 측정값 묶음의 조합, 최대 3개 값의 순차 선택을 비교했다. 묶음이 5개 이하인 경우 모든 묶음 조합을 비교했다. 6개 이상인 자료에서는 각 묶음·두 묶음 조합·한 묶음씩 제외·전체를 비교했으며 모든 가능한 부분집합을 훑은 것은 아니다.',
        'Ridge는 여러 입력을 선형으로 결합하는 회귀 모델이고, RF는 랜덤포레스트로 여러 나무가 나눈 예측을 평균한다. Ridge 규제는 10, RF는 나무 128개·전체 특징 사용·최소 잎 인원은 500명 이상 자료에서 10명, 그보다 작은 자료에서 2명으로 고정했다. 최대 3개 순차 선택은 최초 비교에서 Ridge로 수행했다. 이 제한과 후속 RF 소수 조합 탐색은 구분해 기록한다.',
        '결측치는 각 학습 집단의 중앙값으로 대체하고 결측 여부를 함께 입력했다. 측정값 대신 결측 여부만 쓰는 대조도 계산했다. 실제 SPPB 구성점수·TUG·나이·성별·질환명은 보행 예측 입력에 넣지 않았다. 질환군만의 중앙값 예측은 해석용 대조로 별도 계산했다. 이 대조를 넘는다고 질환·연령 등 모든 교란 요인이 제거됐다는 뜻은 아니다.',
        '속도 상위값은 SPPB 34명, TUG 12명에서 결측이었다. 이 사람들도 제외하지 않고 학습 집단 대체값과 결측 표시를 사용했다. 따라서 모든 사람의 실제 속도 상위값이 측정된 결과는 아니다. 결측 여부만 입력한 RF의 MAE는 SPPB 2.199점·TUG 5.793초로, 실제 측정값을 사용하는 결과의 개선을 결측 표시만으로 설명하지는 못했다.',
        '큰 센서 자료는 COPD·MS·PD·PFF 질환군을 포함한다. 속도 상위값과 정답의 관련성 크기는 질환군 안에서도 서로 달랐다. 모든 고령자·질환군에 동일한 순위가 적용된다고 일반화하지 않는다. TUG가 실제 20초 이상인 277명에서 조합 선택 절차의 MAE는 10.673초, 평균 과소예측량은 9.300초였다. 이 20초 구분은 기존 분석을 잇는 연구용 기술 구분이며 새 진단 경계가 아니다. SPPB 0~2점에서도 평균 오차가 3점 이상으로 컸다. 전체 평균 개선만으로 낮은 기능 집단의 충분한 예측 성능을 주장하지 않는다.',
        '## 확인 범위와 재현',
        '각 자료의 source SHA-256, 사람 분리, 내부 분리, 60개 선택 판단, 모든 예측의 사람 번호·정답·평가 누락과 중복, 지표 재계산을 확인했다. 대표 내부·바깥 모델을 다시 학습해 저장 예측과 대조했다. 모든 나무를 다시 학습한 것은 아니다. 실제 임상 기록의 진실성이나 임상 활용 적합성을 검증한 결과도 아니다.',
        '코드는 이 폴더의 `prepare_comparison.py` → `run_comparison.py` → `analyze_comparison.py` → `verify_comparison.py` 순서다. 중요한 경로는 `run_comparison.py`의 각 바깥 분할 안의 `evaluate`와 정책 선택이다. 입력 후보를 학습 사람의 내부 오차로 고른 뒤, 그 선택을 고정하고 평가 사람을 예측한다. 새 패키지와 GPU는 사용하지 않았다.',
        '큰 입력·예측·분할 기록은 `D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gait-measurement-discovery`에, 작은 결과 표·검산 JSON은 이 C 작업본에 저장했다. 실행 조건은 `comparison-protocol.json`, 입력 연결은 `comparison-preparation.json`, 자료별 지표·상관·선택 빈도·질환별 결과는 자료 이름으로 시작하는 CSV/JSON에서 확인할 수 있다.',
        '## 자료 출처와 해석 범위',
        '[Mobilise-D CVS 공개본](https://zenodo.org/records/21060991), [Mobilise-D 집계 알고리즘 설명](https://mobgap.readthedocs.io/en/latest/modules/generated/aggregation/mobgap.aggregation.MobilisedAggregator.html), [OlderAdults44 공식 자료](https://data.mendeley.com/datasets/xgw6bg3g8h/1), [OlderAdults44 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7701341/). OpenCap 실제 입력 파일과 임상표의 경로·지문은 준비 기록에 남겼다. 자료별 측정 방식·대상자·정답 분포가 달라 하나의 통합 순위로 합치지 않는다.']
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    plot_titles = ['SPPB 총점 · 주간 센서','TUG · 주간 센서','SPPB 균형＋일어서기\n주간 센서','TUG · OpenCap 3초','TUG · OpenCap 6초 대리값','SPPB · Vicon 44명']
    for ax,name,title in zip(axes.ravel(),DATASETS,plot_titles):
        s = summaries[name]
        choices = ['Median|baseline:overall','selected_single','selected_small','selected_family','selected_any']
        index = {r['variant']:r for r in s['selection_policy_metrics']}
        small_label = '3개 이하 선택(선형 조합)'
        if name in sparse_summaries:
            small = next(r for r in sparse_summaries[name]['metrics'] if r['variant']=='RF_selected_up_to3')
            index['selected_small'] = small
            small_label = '3개 이하 선택(RF)'
        values = [index[v]['MAE'] for v in choices]
        low = [index[v]['MAE']-index[v]['MAE_min'] for v in choices]
        high = [index[v]['MAE_max']-index[v]['MAE'] for v in choices]
        ax.barh(range(5),values,color=['#9ba9b5','#62a5cf','#67bcb1','#3976a8','#234c70'])
        ax.errorbar(values,range(5),xerr=np.array([low,high]),fmt='none',color='#263238',capsize=3)
        ax.set_yticks(range(5),['중앙값 기준','단일 측정값 선택',small_label,'측정값 묶음 선택','전체 후보 중 선택'])
        ax.invert_yaxis()
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('평균 절대오차 (초)' if name.endswith('tug') else '평균 절대오차 (점)')
        ax.grid(axis='x',alpha=.2)
    fig.suptitle('사람 단위 예측 오차 · 막대: 3회 평균 · 선: 3회 최소~최대 (신뢰구간 아님)',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(HERE/'measurement-comparison.png',dpi=160)
    plt.close(fig)
    sections.insert(8,f"![사람 분리 예측 오차 비교]({(HERE/'measurement-comparison.png').as_posix()})")
    (HERE/'STATUS.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print('VERIFIED_COMPARISON_REPORT_WRITTEN',flush=True)


if __name__ == '__main__':
    main()
