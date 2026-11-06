from study import *
import platform,py_compile

def main():
    verification=json.loads((HERE/'results-verification.json').read_text());assert verification['passed']
    assert verification['models_replayed_and_independently_refit']==125
    summary=json.loads((HERE/'summary.json').read_text());people=pd.read_csv(HERE/'long-people.csv')
    def metric(view,band):return next(r for r in summary['metrics'] if r['view']==view and r['band']==band)
    names={'MatchedSpeedTurn':'시간을 맞춘 보행 속도 + 회전 시간','PlusSitTime':'3초 보행 속도 + 회전 시간 + 앉기 시간',
        'PlusSitMotion':'3초 보행 속도 + 회전 시간 + 앉기 움직임','PlusSitAll':'3초 보행 속도 + 회전 시간 + 앉기 시간·움직임',
        'ShortSpeedTurn':'3초 보행 속도 + 회전 시간 (짧은 참고 입력)'}
    table=['| 입력 | 전체 100명 절대오차 | 긴 3명 절대오차 | 긴 3명 과소예측 | 긴 3명 과대예측 |','|---|---:|---:|---:|---:|']
    for v in ['MatchedSpeedTurn','PlusSitTime','PlusSitMotion','PlusSitAll','ShortSpeedTurn']:
        m=metric(v,'ge20');a=metric(v,'all')
        table.append(f'| {names[v]} | {a["MAE"]:.3f} | {m["MAE"]:.3f} | {m["underprediction"]:.3f} | {m["overprediction"]:.3f} |')
    individual=['| 사람 | 실제 TUG | 과소예측: 단순 → 앉기 추가 | 절대오차: 단순 → 앉기 추가 |','|---|---:|---:|---:|']
    for pid,g in people.groupby('id'):
        a=g[g.view=='MatchedSpeedTurn'].iloc[0];b=g[g.view=='PlusSitAll'].iloc[0]
        individual.append(f'| {pid} | {a.target:.2f} | {a.underprediction:.2f} → {b.underprediction:.2f} | {a.MAE:.2f} → {b.MAE:.2f} |')
    text='''# 보행 속도·회전 시간 기준의 앉기 추가 비교 — 2026-10-31

**앉기 시간과 움직임을 추가하면 긴 3명의 평균 과소예측은 6.111→5.903초로 0.207초 줄었다. 그러나 한 사람 개선·두 사람 악화였고 전체 오차는 커졌다.** 현재 자료에서 앉기 추가의 안정적인 이득이 확인됐다고 판단하기 어렵다.

## 비교 질문과 동일 조건

입력은 실제 OpenCap 자료에서 추출한 보행 속도 요약·회전 구간 시간·앉기 요약이며, 정답은 기존과 같은 임상 `tft_tug` 초 단위 시간이다. 42.59초인 sub-283을 학습·내부 검증·평가에서 완전히 제외한 기존 100명을 그대로 사용했다. 20초 이상인 긴 사례는 sub-246, sub-267, sub-310 세 명이다. 20초는 분석상 묶음 기준이며 임상 판정선이 아니다.

이전의 사람 분할 5개 × 바깥 5묶음과 내부 3묶음을 파일 그대로 사용했다. 사람을 다시 배치하거나 기록·방문·정답을 바꾸지 않았다. Ridge와 정규화 후보 0.1/1/10/100을 유지하고, 표준화와 후보 선택은 각 학습 사람 내부에서 수행했다. 후보는 내부 평균 절대오차로 고르며 긴 사람의 평가 정답으로 조정하지 않았다. 음수만 0초로 처리하고 상한은 두지 않았다.

보행 속도는 앞선 단순 모델과 같은 세 값, 즉 평균 속도·속도의 표준편차·이웃 시점 속도 차이의 평균 절댓값이다. 회전 정보는 회전 구간의 시간값 하나다. 회전 시간은 먼 지점에서 돌아서는 접근·이탈을 포함하는 기존 구간의 길이이며 순수 회전 각도만의 시간이 아니다.

## 관찰시간을 어떻게 맞췄나

앉기 구간을 추가하면 모델이 더 오래 관찰하는 효과가 생길 수 있다. 이를 통제하기 위해 사람마다 앉기 길이 S초와 회전 길이 D초를 이용해 다음과 같이 비교했다.

| 비교 | 보행 입력 | 회전 입력 | 앉기 입력 | 총 관찰시간 |
|---|---|---|---|---|
| 단순 기준 | 3+S초 속도 요약 | D초 구간의 시간값 | 없음 | 3+D+S초 |
| 앉기 추가 | 3초 속도 요약 | 같은 D초 구간의 시간값 | S초 요약 | 3+D+S초 |

전체 100명에서 양쪽의 총 입력시간이 같으며 5.70~10.53초였다. 기준 모델의 보행 구간은 이전에 품질을 확인한 원래 직선 보행 구간 안에서만 늘렸다. 모두 같은 시작점·파일을 사용했고 회전과 앉기 경계도 유지했다. 새 사람을 제외하지 않았다.

앞선 준비·일어서기까지 포함한 6.73~13.23초와는 구분해야 한다. 이번 비교에서는 준비·일어서기 구간을 양쪽에서 함께 빼고 보행·회전·앉기에 사용하는 시간을 맞췄다. 따라서 이전 전체 동작 입력의 길이를 그대로 사용한 실험은 아니다. 전체 기록은 구간을 찾는 데 쓰였고, 여기의 초 수는 전체 촬영시간이 아닌 모델에 주는 구간 합계다.

앞선 3초 보행 속도+회전 시간 모델도 `ShortSpeedTurn`으로 재학습했다. 이 참고 입력은 3+D초로 더 짧으므로 주된 동일시간 비교로 사용하지 않는다.

## 결과

단위는 초다. 예측값을 먼저 평균한 뒤 오차를 내지 않고, 각 분할의 오차를 계산한 뒤 사람별·집단별로 평균했다. 과소예측량은 실제보다 짧게 예측한 부분만 세며, 절대오차는 과소예측량과 과대예측량의 합이다.

'''+ '\n'.join(table)+'''

미리 정한 주된 비교는 앉기 시간·움직임을 모두 추가한 모델과 시간을 맞춘 단순 모델이다. 긴 사람의 절대오차는 6.279→5.946초로 0.333초, 과소예측은 6.111→5.903초로 0.207초 줄었다. 과대예측도 0.168→0.042초로 줄어, 이번 작은 개선이 과대예측 증가로 얻어진 것은 아니다.

다섯 사람 분할 평균에서는 모두 개선했지만, 같은 세 사람을 반복 평가한 결과이며 독립된 다섯 집단이 아니다. 사람별로 보면 한 명만 개선했다.

'''+ '\n'.join(individual)+'''

sub-246을 평가 집계에서 추가로 빼면 나머지 두 사람의 과소예측은 평균 0.569초, 절대오차는 0.381초 악화한다. 이 확인은 저장된 예측의 집계이며 그 사람까지 제외한 재학습은 아니다. 평균 개선이 특정 사람에 집중돼 있음을 보여준다.

과소예측 차이(앉기 추가−단순)의 조건부 사람 재표집 95% 범위는 [-1.761, +0.690]초, 절대오차 차이는 [-1.761, +0.448]초로 모두 0을 포함했다. 이 범위는 세 사람의 이미 학습된 예측을 재표집한 것이므로 새 집단의 불확실성을 충분히 나타내지는 못한다.

전체 100명의 절대오차는 1.139→1.195초, 나머지 97명은 0.981→1.048초로 소폭 악화했다. 전체 사람의 안정적인 개선도 확인되지 않았다. 앉기 시간만 추가하면 전체 오차는 1.111초였으나 긴 사람의 과소예측 감소는 약 0.090초에 그쳤다. 움직임만 추가한 경우도 과소예측 감소는 약 0.111초였다.

짧은 3초 보행 기준과 비교하면 앉기 전체 추가의 긴 사람 과소예측은 5.870→5.903초로 오히려 0.033초 커졌다. 절대오차는 6.047→5.946초로 0.101초 줄었다. 기준을 어떻게 잡아도 큰 추가 이득이 나타난 결과는 아니다.

## 앞선 결과와의 관계

앞선 실험은 준비·일어서기·회전의 여러 특징이 이미 있는 상태에서 앉기 정보를 뺐을 때의 변화를 봤다. 이번에는 보행 속도 세 값과 회전 시간 하나만 있는 단순 모델에 앉기를 추가했다. 함께 쓰는 정보가 다르므로 앞선 제거 실험의 이득이 이번 추가 실험에 그대로 적용되는 것은 아니다.

앉기 움직임에는 골반의 수직 높이 변화·범위, 수직 속도의 평균·변동·큰 값, 속도 변화, 거의 멈춘 비율이 들어 있다. 이 요약이 근력이나 균형의 원인을 직접 측정한다는 뜻은 아니다. 현재 자료에서는 단순 기준을 유지하고 앉기는 검증 후보로 남기는 편이 결과에 부합한다. 앉기 정보가 일반적으로 쓸모없다는 결론도 아니다.

공개 기록과 임상 계시 시행의 정확한 cone/line 대응, 자동 구간 경계의 수동 정답, 단일 카메라 측정 정확도는 이번 실험으로 해결되지 않았다. 기존 개발자료를 반복 분석한 탐색 결과이며 외부 검증은 아니다. 더 큰 237명 집단의 결과와 여기의 100명 결과를 직접 비교하지 않았다.

## 구현과 검산

`study.py`는 기존 100명과 분할을 불러오고, 앉기와 같은 시간만큼 기존 깨끗한 보행 구간을 더 잘라 기준 모델의 속도를 계산한다. 앉기 입력의 특징은 기존 배열에서 그대로 가져온다. `train.py`는 각 입력 조합에 대해 학습 사람만으로 설정을 고르고 예측을 저장한다. `check_results.py`는 학습 함수를 재사용하지 않고 scikit-learn 객체를 직접 만들어 예측과 내부 선택을 다시 계산한다.

로컬 CPU에서 최종 Ridge 125개를 학습했고, 125개 모두 독립 재학습 및 예측 재현을 통과했다. 내부 검증 학습 1,500회도 별도로 다시 계산해 일치했다. 사람 분할, 42.59초 사례 제외, 원래 정답, 모든 사람의 동일 입력시간, 특징 추출, 사람별·분할별 지표, 조건부 재표집 범위를 검산했다.

학습 전 입력 검사에서 새 속도 배열과 회전 시간값을 연결할 때 float32 자동 변환으로 최대 약 0.000000223초 차이가 발생했다. 속도 배열을 float64로 올린 후 원래 회전 시간값을 연결하도록 수정해 두 입력의 시간 특징을 정확히 일치시켰다. 초기 프로토콜은 보존했으며 이 수정 전에는 모델 학습이 실행되지 않았다. 결과에 맞춰 평가 기준이나 정답을 바꾸지 않았다.

기존 연구 폴더는 수정하지 않았다. 새 패키지 설치나 GPU 작업은 없다. 재현 순서는 `study.py` → `check_inputs.py` → `train.py` → `assess.py` → `check_results.py` → `finalize.py`이며 이미 완료한 준비·학습은 덮어쓰지 않는다.

근거: `protocol.json`, `observation-audit.json`, `metrics.csv`, `long-people.csv`, `partition-metrics.csv`, `summary.json`, `input-verification.json`, `results-verification.json`, `sit-simple-comparison.png`.
'''
    (HERE/'STATUS.md').write_text(text,encoding='utf8')
    sys.path.insert(0,str(HERE.parents[1]/'control/manual-sessions/20261028-camera-step-pilot/deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
    fig,ax=plt.subplots(1,2,figsize=(12.5,5.5))
    views=['MatchedSpeedTurn','PlusSitTime','PlusSitMotion','PlusSitAll'];labels=['단순 기준','앉기 시간 추가','앉기 움직임 추가','앉기 전체 추가']
    values=[metric(v,'ge20')['underprediction'] for v in views]
    bars=ax[0].bar(np.arange(4),values,color=['#8c9bad','#83a8f3','#83a8f3','#2360de']);ax[0].bar_label(bars,fmt='%.2f')
    ax[0].set_xticks(np.arange(4),labels,rotation=12);ax[0].set_ylim(0,7.2);ax[0].set_ylabel('평균 과소예측량 (초)');ax[0].set_title('긴 3명의 평균: 작은 감소')
    ids=['sub-246','sub-267','sub-310'];delta=[]
    for pid in ids:
        g=people[people.id==pid].set_index('view');delta.append(g.loc['PlusSitAll','underprediction']-g.loc['MatchedSpeedTurn','underprediction'])
    bars=ax[1].barh(np.arange(3),delta,color=['#2360de' if x<0 else '#de6b36' for x in delta]);ax[1].bar_label(bars,fmt='%+.2f',padding=4)
    ax[1].set_yticks(np.arange(3),['22.18초 사례','22.50초 사례','20.06초 사례']);ax[1].invert_yaxis();ax[1].axvline(0,color='black',lw=.8)
    ax[1].set_xlim(-2.3,1.2);ax[1].set_xlabel('앉기 추가 후 과소예측 변화 (초)\n음수: 개선 / 양수: 악화');ax[1].set_title('한 사람 개선 · 두 사람 악화')
    fig.suptitle('보행 속도·회전 시간에 앉기 정보를 더하면?',fontsize=17)
    fig.text(.5,.015,'같은 100명·사람 분할·총 입력시간 5.70~10.53초 / 42.59초 사례 완전 제외 / 외부 검증 아님',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(HERE/'sit-simple-comparison.png',dpi=160);plt.close(fig)
    import sklearn
    save('environment.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__})
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,h in protocol['source_hashes'].items():assert sha(HERE.parent/name)==h
    for p in HERE.glob('*.py'):py_compile.compile(str(p),doraise=True)
    save('artifact-manifest.json',{p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'})
    print(json.dumps({'finalized':True,'models_verified':125,'inner_fits_verified':1500,'tail_under':values}),flush=True)
if __name__=='__main__':main()
