from study import *
import platform

def main():
    verified=json.loads((HERE/'results-verification.json').read_text());assert verified['passed']
    summary=json.loads((HERE/'summary.json').read_text());cases=pd.read_csv(HERE/'case-results.csv');diagnostics=pd.read_csv(HERE/'diagnostic-subsets.csv')
    labels={'Baseline':'기존 정지시간 추가','Pooled':'왕복 통합 이동시간 정보 추가','Directional':'갈 때·돌아올 때 분리'}
    def metric(cohort,view,band):return next(r for r in summary[cohort]['metrics'] if r['view']==view and r['band']==band)
    lines=['# 보행 방향별 속도와 고정 정지시간 실험 — 2026-10-31','',
      '**왕복을 나누는 주 가설은 긴 TUG 개선으로 이어지지 않았다.** 같은 176명·최대 20초에서 방향별 모델의 긴 8명 절대오차는 5.447→5.613초로 커졌다. 보조 비교인 통합 이동 속도 역수 추가는 전체 오차 1.087→1.014초, 긴 사람 절대오차 5.447→5.304초로 줄였지만, 긴 사람 개선은 작고 42.59초 포함 조건에서 유지되지 않았다. 준비시간을 제외하는 방향은 유지하되, 새 모델을 확정 채택할 근거로 보지는 않는다.','',
      '준비시간의 실제 계측 포함 여부를 확인할 수 없어, 이번에는 준비·기립·착석 특징을 넣지 않았다. 기존 정지시간 추가 모델에서 왕복 평균에 가려지는 느린 구간을 찾기 위해, 갈 때와 돌아올 때의 이동 속도를 각각 반영했다. 실제 공개 OpenCap 보행·회전 입력과 임상 TUG 시간으로 실행한 개발자료 실험이다.','',
      '42.59초 사례를 제외한 176명(20초 이상 8명)을 주 비교로 사용했다. 포함 177명(긴 9명)은 민감도 비교다. 사람 순서·정답·원본 파일·관찰 구간·기존 5가지 사람 분할과 내부 3묶음 분할을 유지했다. 모든 모델에서 관찰한 보행 정지 후보 시간을 계수 1로 더하고, Ridge가 나머지 시간을 예측한다. 이 나머지에는 관찰하지 않은 일어서기·앉기 등의 시간도 남으므로, 정답이 따로 있는 순수 이동시간으로 부르지 않는다.','',
      '입력 → 기존 속도 평균·변동·변화와 회전시간 → 추가 이동시간 정보 → 학습 사람에서만 표준화·Ridge 학습 → 예측한 나머지 시간에 관찰 정지시간 추가의 순서다. 새 이동시간 정보는 정지 후보를 뺀 이동 속도의 역수다. 예를 들어 0.5m/s이면 1m에 2초라는 뜻이며, 실제 전체 과제 수행시간을 직접 입력하는 것은 아니다.','',
      '기존 모델은 학습 입력 4개, 통합 모델은 5개, 방향별 모델은 6개다. 통합 모델은 이동 중 전체 평균 속도의 역수 하나를 더하고, 방향별 모델은 갈 때·돌아올 때 각각의 역수 두 개를 더한다. 두 새 모델을 함께 비교해 단순한 역수 변환 효과와 방향 분리 효과를 구분했다. 두 방향 모두 이동 관측 0.5초 이상이 있었으며 결측 대체나 추가 사람 제외는 없었다.','',
      '정지 후보는 기존과 동일하게 평활화한 수평 속도 0.10m/s 미만, 지속 0.5초 이상이며 0.2초 이하의 유효 구간 틈만 연결한다. 추적 급변·앉은 자세로 판정된 구간은 제외한다. RGB 원영상으로 확인한 임상 정지 정답은 아니다. 새 속도는 앞선 이동 속도 실험과 같은 원래 수평 속도로 계산했다.','',
      '정규화 강도는 0.1·1·10·100을 학습 사람 내부의 전체 TUG 절대오차로 선택했다. 평가 정답으로 모델 종류·강도를 선택하지 않았다. 음수 나머지만 0으로 처리하고 예측 상한은 두지 않았다. 이번 결과를 본 뒤 특징이나 임계값을 다시 조정하지 않았다.','',
      '## 주 비교: 42.59초 제외, 최대 20초 관찰','',
      '단위는 초이다. 과소예측량은 실제보다 짧게 예측한 부분만 평균한다. 절대오차는 과소예측량과 과대예측량의 합이다. 다섯 분할의 오차를 사람별로 먼저 평균했다.','',
      '| 입력 | 전체 절대오차 | 긴 8명 절대오차 | 긴 8명 과소예측 | 긴 8명 과대예측 |','|---|---:|---:|---:|---:|']
    for model in labels:
        allr=metric('exclude283',model+'_20','all');r=metric('exclude283',model+'_20','ge20')
        lines.append(f'| {labels[model]} | {allr["MAE"]:.3f} | {r["MAE"]:.3f} | {r["underprediction"]:.3f} | {r["overprediction"]:.3f} |')
    for model in ['Pooled','Directional']:
        c=next(c for c in summary['exclude283']['contrasts'] if c['a'][2]==model+'_20' and c['b'][2]=='Baseline_20' and c['band']=='ge20');v=c['MAE'];u=c['underprediction']
        lines+=['',f'{labels[model]}의 긴 사람 절대오차 변화는 {v["difference"]:+.3f}초, 조건부 사람 재표집 95% 범위는 [{v["conditional_bootstrap95"][0]:.3f}, {v["conditional_bootstrap95"][1]:.3f}]초다. 절대오차는 {v["people_better"]}/8명, {v["better_partitions"]}/5분할에서 개선됐다. 과소예측량 변화는 {u["difference"]:+.3f}초다. 이 범위는 고정된 예측에서 사람을 다시 뽑은 범위이며 새 집단에서 재학습한 검증이 아니다.']
    lines+=['','## 관찰시간과 극단 사례 포함 여부','',
      '10·20·30초는 관찰시간 상한이다. 각 사람에게 실제 기록이 있는 만큼만 사용했으며, 모두에게 정확히 같은 길이의 영상을 확보했다는 뜻은 아니다. 같은 사람·같은 상한에서는 세 모델의 관찰 구간이 정확히 일치한다. 원래 기록 전체에서 동작 경계를 찾았으므로 실시간 시작 탐지 실험도 아니다.','',
      '| 집단 | 최대 관찰초 | 입력 | 전체 절대오차 | 긴 사람 절대오차 | 긴 사람 과소예측 |','|---|---:|---|---:|---:|---:|']
    for cohort in summary:
        for b in [10,20,30]:
            for model in labels:
                a=metric(cohort,f'{model}_{b}','all');r=metric(cohort,f'{model}_{b}','ge20')
                lines.append(f'| {"제외176" if cohort=="exclude283" else "포함177"} | {b} | {labels[model]} | {a["MAE"]:.3f} | {r["MAE"]:.3f} | {r["underprediction"]:.3f} |')
    lines+=['','## 28.03초 사례와 다른 긴 사람','',
      '| 입력 | 28.03초 사례 평균 예측 | 분할 간 표준편차 | 예측 범위 | 절대오차 |','|---|---:|---:|---|---:|']
    for model in labels:
        r=cases[(cases.cohort=='exclude283')&(cases.id=='sub-397')&(cases.view==model+'_20')].iloc[0]
        lines.append(f'| {labels[model]} | {r.prediction:.3f} | {r.sd:.3f} | {r.minimum:.3f}–{r.maximum:.3f} | {r.MAE:.3f} |')
    lines+=['','28.03초 사람만 평가에서 뺀 나머지 긴 7명 결과도 확인했다. 이 진단은 모델을 새로 학습한 결과가 아니다.']
    for model in labels:
        r=diagnostics[(diagnostics.cohort=='exclude283')&(diagnostics.subset=='long_without397')&(diagnostics.view==model+'_20')].iloc[0]
        lines.append(f'- {labels[model]}: 절대오차 {r.MAE:.3f}초, 과소예측량 {r.underprediction:.3f}초.')
    people=pd.read_csv(HERE/'exclude283/person-results.csv');manifest=json.loads((HERE/'exclude283/manifest.json').read_text());lookup={m['id']:m for m in manifest}
    detail=[]
    for pid,g in people[(people.target>=20)&people.view.str.endswith('_20')].groupby('id'):
        m=lookup[pid];r={'id':pid,'target':float(g.target.iloc[0]),'outbound_speed':m['pace']['20']['outbound']['speed'],'return_speed':m['pace']['20']['return']['speed'],'turn_seconds':m['turn_observed_seconds'],'stop_seconds':m['budgets']['20']['delay_features']['10'][2]}
        for row in g.itertuples():r.update({row.view+'_'+k:float(getattr(row,k)) for k in ['prediction','MAE','underprediction']})
        detail.append(r)
    pd.DataFrame(detail).to_csv(HERE/'long-case-detail.csv',index=False)
    lines+=['','## 해석과 확인 범위','',
      '통합 모델의 전체 오차 감소는 5개 사람 분할 모두에서 관찰됐고 113/176명이 개선됐다. 반면 긴 8명 절대오차는 4명만 개선됐으며 차이의 조건부 95% 범위가 0을 포함한다. 28.03초 사례를 평가에서 빼면 긴 7명의 절대오차는 5.526→5.491초로 거의 같았다. 이 사례의 평균 예측은 23.138→24.036초로 정답에 가까워졌지만, 분할 간 표준편차는 0.349→0.714초로 커졌다. 정확도와 안정성이 함께 개선된 결과는 아니다.','',
      '42.59초를 포함한 177명에서는 통합 모델의 전체 오차는 1.196→1.153초로 줄었지만 긴 9명 절대오차는 6.322→6.424초, 과소예측량은 5.849→5.855초였다. 이 집단에서 같은 기존 8명만 평가하면 절대오차는 4.917→4.906초로 거의 같았다. 42.59초 본인의 예측은 25.029→24.016초로 더 낮아졌다. 8명과 9명의 묶음 평균 차이만으로 포함 효과를 판단하지 않았다.','',
      '따라서 현재 비교 기준은 기존 고정 정지시간 추가 모델로 유지하고, 통합 이동 속도 역수 추가는 전체 오차를 줄인 후속 검증 후보로 남긴다. 방향별 속도 분리는 이번 자료에서 긴 TUG 문제를 해결하지 못했다. 최대 30초도 20초보다 일관되게 좋지 않았다. 남은 큰 오차를 준비시간이나 특정 운동 기능 탓으로 확정할 자료는 없다.','',
      '준비시간을 제외한 판단은 계측 시작 시점을 확인하지 못한 변수에 의존하지 않으려는 선택이다. 보행만으로 TUG의 모든 구성 요소를 복원할 수 있다는 뜻은 아니다. 같은 사람·방문의 임상 TUG와 연결했지만 정확히 어떤 시행을 계측했는지와 개인별 시작 신호 시각은 공개 자료에서 확인하지 못했다. 특히 42.59초 기록은 이 불확실성을 유지한 채 포함·제외로 비교했다.','',
      '현재 자료는 모델과 특징을 여러 차례 비교한 개발자료이며, 긴 사람이 8명뿐이다. 사람을 학습·평가에서 분리했더라도 반복된 실험 선택의 영향을 없애지는 못한다. 이번 성능 차이는 후속 검증 후보를 판단하는 근거이고, 독립 자료의 성능·SPPB 예측·단일 카메라 임상 측정 정확도를 입증하지 않는다.','',
      f'원본 MOT에서 531개 관찰 구간을 재계산해 입력 차이 0, 기존 분할·정답·정지시간 일치를 확인했다. 최종 모델 {verified["models"]}개와 내부 학습 {verified["inner_fits"]}회를 별도 경로로 재학습하고 저장 예측·지표·조건부 재표집 범위를 검산했다. 기존 정지시간 추가 예측도 그대로 재현했다.','',
      '`study.py`의 `pace()`가 동일 구간에서 정지 후보를 제외하고 왕복 속도를 계산한다. `fit()`은 마지막 열의 정지시간을 학습 정답에서 뺀 뒤 나머지를 학습하고 다시 더한다. 기존 연구 폴더는 변경하지 않고 이 폴더에만 입력·모델·예측을 저장했다. 기존 Python·NumPy·scikit-learn으로 로컬 CPU에서 실행했으며 새 패키지와 GPU는 사용하지 않았다.','',
      '재현 순서: `study.py` → `check_inputs.py` → `train.py` → `assess.py` → `check_results.py` → `report.py` → `finalize.py`. 준비·학습은 기존 산출물 덮어쓰기를 막는다. 재실행하려면 별도 폴더를 사용한다. 주요 수치는 `summary.json`, 사람별 결과는 `long-case-detail.csv`, 사례 변동은 `case-results.csv`, 검산은 `results-verification.json`에 있다.']
    (HERE/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    import sklearn
    save('environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})
    sys.path.insert(0,str(HERE.parent.parent/'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(14,4.4))
    for ax,(band,key,title) in zip(axes,[('all','MAE','전체 176명 절대오차'),('ge20','MAE','긴 8명 절대오차'),('ge20','underprediction','긴 8명 과소예측량')]):
        for model in labels:ax.plot([10,20,30],[metric('exclude283',f'{model}_{b}',band)[key] for b in [10,20,30]],marker='o',label=labels[model])
        ax.set(title=title,xlabel='최대 관찰시간 (초)',ylabel='오차 (초)',xticks=[10,20,30]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('준비시간 제외 · 동일 사람 분할과 관찰 구간 · 42.59초 사례 제외')
    fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=160);plt.close(fig)
    print('REPORT_WRITTEN',flush=True)
if __name__=='__main__':main()
