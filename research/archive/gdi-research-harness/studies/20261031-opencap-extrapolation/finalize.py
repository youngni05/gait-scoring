from study import *
from collections import Counter
import platform,sklearn,scipy,joblib
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 verification=json.loads((HERE/'results-verification.json').read_text());assert verification['passed'] and verification['new_models']==verification['RF_models']==900
 results=json.loads((HERE/'summary.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text());families=['RF','Ridge','Huber','Gamma','CVSelect'];labels={'RF':'기존 RF','Ridge':'Ridge','Huber':'Huber','Gamma':'Gamma','CVSelect':'학습 내부 자동 선택'}
 primary=results['prep_event']['models']['PrepAll'];geometry=results['geometry']['models']['Walk3Turn'];rows=[];geo_rows=[]
 for family in families:
  m=primary[family];rows.append(f"| {labels[family]} | {m['all']['MAE']:.3f} | {m['ge20']['MAE']:.3f} | {m['ge20']['underprediction']:.3f} | {m['ge20']['overprediction']:.3f} | {m['lt20']['MAE']:.3f} |")
  g=geometry[family];geo_rows.append(f"| {labels[family]} | {g['all']['MAE']:.3f} | {g['ge20']['MAE']:.3f} | {g['ge20']['underprediction']:.3f} |")
 pp=pd.read_csv(HERE/'prep_event/person-results.csv');b=pp[(pp.view=='PrepAll')&(pp.family=='RF')&(pp.target>=20)].set_index('id');a=pp[(pp.view=='PrepAll')&(pp.family=='Ridge')].set_index('id').loc[b.index];gain=b.MAE-a.MAE;share=float(gain.loc['sub-283']/gain.sum())
 person_rows='\n'.join(f"| {pid} | {b.loc[pid,'target']:.2f} | {b.loc[pid,'prediction']:.2f} | {a.loc[pid,'prediction']:.2f} | {b.loc[pid,'MAE']:.2f} → {a.loc[pid,'MAE']:.2f} |" for pid in b.index)
 p=pd.read_csv(HERE/'prep_event/predictions.csv');rf=pd.read_csv(source(protocol['cohorts']['prep_event'])/'predictions.csv');rf=rf[rf.method=='PrepAll'].copy();rf['view']='PrepAll';rf['family']='RF';raw=pd.concat([p,rf],ignore_index=True);worst=raw[(raw.view=='PrepAll')&(raw.id=='sub-283')];worst.to_csv(HERE/'sub283-predictions.csv',index=False)
 extremes={f:{'mean':float(g.prediction.mean()),'min':float(g.prediction.min()),'max':float(g.prediction.max()),'MAE':float(np.mean(abs(g.prediction-g.target)))} for f,g in worst.groupby('family')};save('case-audit.json',{'sub283':extremes,'fraction_of_tail_MAE_reduction_from_sub283':share})
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False;fig,axes=plt.subplots(1,2,figsize=(12,5.7));names=[labels[f] for f in families];under=np.array([primary[f]['ge20']['underprediction'] for f in families]);over=np.array([primary[f]['ge20']['overprediction'] for f in families]);ax=axes[0];ax.barh(names,under,label='과소예측',color='#2563eb');ax.barh(names,over,left=under,label='과대예측',color='#ea580c');ax.invert_yaxis();ax.set_title('긴 TUG 4명의 평균 절대오차');ax.set_xlabel('오차 (초) = 과소예측 + 과대예측');ax.legend();ax.grid(axis='x',alpha=.2)
 for i,v in enumerate(under+over):ax.text(v+.1,i,f'{v:.2f}',va='center',fontsize=9)
 ax.set_xlim(0,16)
 ax=axes[1]
 for i,f in enumerate(families):
  v=extremes[f];ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['min']],[v['max']-v['mean']]],fmt='o',capsize=4,color='#2563eb' if f=='Ridge' else '#64748b')
 ax.axvline(42.59,color='#16a34a',ls='--',label='실제 42.59초');ax.set_yticks(range(5),names);ax.invert_yaxis();ax.set_xlim(0,145);ax.set_title('가장 긴 사례의 예측: 평균과 최소-최대');ax.set_xlabel('예측 TUG (초)');ax.legend();ax.grid(axis='x',alpha=.2)
 fig.suptitle('같은 101명·입력·사람 분할에서 모델만 변경',fontsize=15);fig.text(.5,.02,'Ridge·Huber·Gamma: 5개 사람 분할 / RF: 같은 분할과 3개 초기값 · 독립 외부 검증 아님',ha='center',fontsize=10);fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(HERE/'extrapolation-comparison.png',dpi=150);plt.close(fig)
 comparisons=[]
 for cohort,info in protocol['cohorts'].items():
  for view in info['views']:
   s=results[cohort]['models'][view]
   for family in families:
    for band in ['all','lt20','ge20']:comparisons.append({'cohort':cohort,'view':view,'family':family,'band':band,**s[family][band]})
 pd.DataFrame(comparisons).to_csv(HERE/'all-comparisons.csv',index=False)
 selected_rows=[]
 for cohort,info in protocol['cohorts'].items():
  view=info['views'][1];rfm=results[cohort]['models'][view]['RF'];new=results[cohort]['models'][view]['CVSelect'];selected_rows.append(f"| {cohort} / {view} | {info['people']} / {info['long_people']} | {rfm['all']['MAE']:.3f} → {new['all']['MAE']:.3f} | {rfm['ge20']['MAE']:.3f} → {new['ge20']['MAE']:.3f} |")
 q=results['prep_event']['comparisons']['PrepAll:Ridge minus PrepAll:RF / ge20'];within=results['prep_event']['comparisons']['PrepAll:Ridge minus TurnContext:Ridge / ge20'];geo=results['geometry']['comparisons']['Walk3Turn:Gamma minus Walk3Turn:RF / ge20'];failures=sum(s['candidate_failures'] for s in results.values())
 report=f'''# OpenCap 학습 정답 범위 밖 예측 비교 — 2026-10-31

**이번 개발자료에서는 Ridge가 긴 TUG의 과소예측과 절대오차를 함께 줄였다.** 준비 동작·기립·착석을 포함한 동일 101명 입력에서 긴 4명의 평균 절대오차는 RF 11.474초→Ridge 3.507초, 과소예측량은 11.474초→3.177초였다. 그러나 Gamma는 같은 극단 사례를 최대 138.22초로 과대예측했고, 학습 내부 자동 모델 선택도 이를 충분히 막지 못했다. 출력 범위를 넓히는 것만으로 항상 좋아지는 것은 아니다.

## 무엇을 고정하고 무엇을 바꿨나

실제 임상 TUG 시간, 사람·방문·원본 기록, 특징 배열, 구간 길이, 기존 5가지 사람 분할과 내부 3묶음 검증 분할을 모두 그대로 사용했다. 기존 238명 회전 비교와 앞선 기립·착석의 다섯 비교를 재사용했다. 모델만 교체했으며 추가 자료 선별·구간 수정·정답 보정·긴 사람 가중치 변경은 없었다. 원본 입력·분할·기록의 해시를 고정해 확인했다.

각 비교의 두 입력도 앞선 실험과 동일한 관찰시간이다. 238명 자료는 같은 시간의 직선과 직선+회전, 나머지는 회전·주변 보행과 기립/착석 추가의 비교다. 시간은 선택 구간 합계이며 전체 촬영시간은 아니다. 구간 선정에는 전체 기록을 이용했다. 준비 동작 포함 입력은 기존과 같은 6.73~24.80초이다.

기존 RF는 학습 정답들을 평균하므로 학습 최댓값을 넘지 못한다. 다음 세 모델은 학습한 관계를 더 큰 입력값까지 연장할 수 있다. 이는 범위 밖 예측이 가능하다는 뜻이며 정확하다는 보장은 아니다.

- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html): 입력과 시간의 직선 관계를 학습하며 계수가 지나치게 커지는 것을 억제한다. 정규화 강도 0.1, 1, 10, 100을 비교했다.
- [Huber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html): 큰 잔차의 영향을 줄이는 선형 회귀다. 강도 0.01, 1, 10, 100, epsilon 1.35를 사용했다. 큰 실제 TUG를 자동으로 정확하게 예측하는 방법은 아니다.
- [Gamma 회귀](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html): 로그 연결을 사용해 양수 시간을 예측한다. 강도 0.001, 0.01, 0.1, 1을 비교했다. 범위 밖에서 값이 빠르게 커질 수 있다.

표준화와 모델 설정 선택은 학습 사람 내부에서만 수행했다. 모든 후보는 내부 검증의 초 단위 평균 절대오차로 선택했다. 각 모델별 결과 외에 세 모델 중 하나를 같은 내부 기준으로 선택하는 `CVSelect`도 미리 정해 평가했다. 최종 평가 정답으로 모델을 선택하지 않았다. 이후 관찰한 Ridge의 좋은 결과는 후보 발견으로 해석해야 하며 외부 검증된 최종 채택이 아니다.

세 새 모델은 초기값을 무작위로 바꾸는 방식이 아니므로, 같은 결과를 3번 복제해 독립 반복처럼 세지 않았다. 각 모델은 5개 사람 분할을 평가했다. RF는 동일 분할의 기존 초기값 3개에서 나온 **오차를 평균**해 비교했다. RF 예측을 먼저 평균한 뒤 오차를 내는 앙상블 비교로 바꾸지 않았다.

음수 선형 예측만 0초로 처리하고 원래 값도 저장했다. 학습 최댓값이나 임의의 큰 시간으로 상한을 두지 않았다. 실제 큰 과대예측도 평가에서 제외하지 않았다. Gamma의 첫 Newton 최적화에서 수렴 경고가 발생해 전체 초기 실행을 중단했고, 같은 모델·정규화 후보를 L-BFGS로 다시 계산했다. 모델 성능을 읽기 전의 계산 방법 수정이며 데이터나 평가 기준은 바꾸지 않았다. 최종 실행에서 내부 후보 실패는 {failures}건이었다. Huber 19건과 Gamma 1건의 수렴 실패이며, 미리 정한 규칙대로 해당 설정을 선택 대상에서 제외했다. 각 학습 묶음에는 성공한 후보가 남았고, 최종 선택된 신규 모델 900개는 모두 학습과 독립 재학습 검산을 통과했다.

## 준비·기립·착석 입력: 같은 101명, 긴 사례 4명

단위는 초이다. 절대오차는 과소예측량과 과대예측량의 합이다. 각 실행의 오차를 먼저 계산한 뒤 평균했다. 20초 이상은 이번 분석의 묶음 기준이며 임상 판정선이 아니다.

| 모델 | 전체 절대오차 | 긴 4명 절대오차 | 긴 4명 과소예측 | 긴 4명 과대예측 | 나머지 97명 절대오차 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Ridge는 긴 4명 모두의 평균 절대오차를 줄였고, 다섯 사람 분할에서 모두 평균 개선 방향이었다. RF 대비 긴 사람 절대오차 변화는 {q['MAE']['difference']:.3f}초, 사람 재표집 95% 범위는 [{q['MAE']['conditional_person_bootstrap95'][0]:.3f}, {q['MAE']['conditional_person_bootstrap95'][1]:.3f}]초였다. 과소예측량 변화는 {q['underprediction']['difference']:.3f}초였다. 다른 97명의 평균 오차도 1.009→0.983초로 악화하지 않았다. 다만 이 범위는 이미 학습한 모델에서 사람별 오차를 재표집한 조건부 범위이며 새 집단에서 다시 학습한 검증이 아니다.

| 사람 | 실제 TUG | RF 평균 예측 | Ridge 평균 예측 | 평균 절대오차 RF→Ridge |
|---|---:|---:|---:|---:|
{person_rows}

42.59초 사례의 Ridge 예측은 다섯 분할에서 {extremes['Ridge']['min']:.2f}~{extremes['Ridge']['max']:.2f}초였다. 학습 정답의 최댓값 22.18~22.50초를 **실제로 넘어서 예측**했다. 평균 예측은 {extremes['Ridge']['mean']:.2f}초지만, 평균끼리의 차이 0.31초를 모델 오차로 제시하면 안 된다. 실제 분할별 절대오차의 평균은 {extremes['Ridge']['MAE']:.2f}초이다.

긴 4명의 절대오차 감소 총량 중 약 {100*share:.1f}%가 이 한 사례에서 나왔다. 네 명 모두 좋아졌다는 사실과 별개로 큰 효과의 일반화 여부는 특히 조심해서 판단해야 한다.

## 모델을 바꾸면 기립·착석 정보의 의미도 달라지나

같은 101명에서 Ridge를 양쪽 입력에 모두 적용했다. 관찰시간을 맞춘 회전·주변 보행만 쓰면 긴 사람 절대오차가 6.094초, 준비·기립·착석을 추가하면 3.507초였다. 과소예측량은 6.027→3.177초였다. 다섯 분할 평균은 모두 개선됐지만, 사람별로는 3명 개선·1명 악화였고 절대오차 차이의 조건부 95% 범위는 [{within['MAE']['conditional_person_bootstrap95'][0]:.3f}, {within['MAE']['conditional_person_bootstrap95'][1]:.3f}]초로 0을 포함한다.

앞선 RF의 부정적 결과가 기립·착석 정보가 없다는 증거는 아니었다. 이번 결과는 입력과 모델의 조합이 중요하다는 근거다. 단, 상한 하나만 제거한 실험은 아니며 모델 구조와 학습 손실도 달라졌으므로 모든 개선을 RF 상한 하나의 인과효과로 단정할 수 없다.

## 긴 9명이 모두 있는 238명 회전 자료

| 모델 | 전체 절대오차 | 긴 9명 절대오차 | 긴 9명 과소예측 |
|---|---:|---:|---:|
{chr(10).join(geo_rows)}

Gamma가 이 집단에서는 RF보다 긴 사람 절대오차를 7.592→6.417초, 과소예측을 7.449→5.793초로 줄였다. 그러나 긴 9명의 절대오차 개선은 5명, 악화는 4명이었고 차이의 조건부 95% 범위 [{geo['MAE']['conditional_person_bootstrap95'][0]:.3f}, {geo['MAE']['conditional_person_bootstrap95'][1]:.3f}]초는 0을 포함한다. 직선 속도만 쓰면 Gamma의 긴 사람 절대오차는 12.692초로 RF 11.571초보다 나빴다. 외삽 가능한 모델로 바꾸기만 하면 모든 입력에서 좋아지는 결과는 아니다.

## 자동 모델 선택과 과대예측 문제

101명 준비 입력에서 Gamma는 42.59초 사례를 평균 {extremes['Gamma']['mean']:.2f}초, 최대 {extremes['Gamma']['max']:.2f}초로 예측했다. 과소예측량은 줄었지만 긴 사람 절대오차는 14.051초로 RF보다 나빠졌다. 낮은 과소예측량만으로 성공이라고 평가하면 이 문제를 놓친다.

내부 평균 절대오차로 모델 종류를 고르는 방식은 준비 입력의 25개 바깥 학습 묶음에서 Gamma 13회·Huber 9회·Ridge 3회를 선택했다. 이 방식도 큰 과대예측을 막지 못했다. 학습에 극단 사례가 없는 경우, 학습 내부의 평균 오차만으로 범위 밖 동작을 보장할 수 없다는 한계가 드러났다.

미리 정한 자동 선택 방식의 전체 결과도 함께 남긴다. 화살표는 같은 입력의 RF→자동 선택이다. 각 행의 사람이 달라 행 사이 숫자는 직접 비교하지 않는다.

| 집단 / 입력 | 전체 / 긴 사람 | 전체 절대오차 | 긴 사람 절대오차 |
|---|---:|---:|---:|
{chr(10).join(selected_rows)}

## 확인 범위와 남은 제한

입력은 실제 공개 OpenCap MOT에서 만든 기존 특징이며, 진짜 임상 TUG 시간과 연결한 개발자료 비교다. 같은 방문의 정확한 계시 시행 대응, 준비 동작 경계의 수동 정답, 일부 기록의 착석 누락·추적 급변은 앞선 실험과 동일하게 남아 있다. 42.59초 사례의 임상시간과 골반 기립 핵심~착석 간격 사이 차이도 정답을 수정하지 않고 유지했다. 데이터 연결 문제가 모델 개선으로 해결됐다고 보지 않는다.

이번 결과는 Ridge를 다음 검증 후보로 삼을 근거다. 반복해서 본 개발자료에서 모델들을 비교한 결과이므로 새 긴 TUG 사례나 독립 자료에서 같은 개선이 유지되는지는 확인하지 못했다. SPPB 예측이나 단일 카메라 임상 측정 정확도를 검증한 결과도 아니다.

## 구현과 검산

`study.py`의 `fit()`은 기존 특징의 학습 사람 부분만 표준화하고 모델을 학습한 뒤 평가 사람의 시간을 예측한다. `fit_models.py`는 기존 내부 분할로 정규화 강도와 모델 종류를 선택한다. 입력을 다시 자르지 않으므로 관찰시간이 바뀌지 않는다. `assess.py`는 RF 초기값 평균과 새 결정적 모델의 반복 수 차이를 처리하고, 과소·과대예측을 함께 집계한다.

로컬 Windows CPU에서 신규 모델 900개를 학습했다. `check_results.py`에서 저장된 신규 900개를 재현하고 **학습 사람만으로 독립 재학습**해 일치를 확인했다. 수식으로 계산한 선형·로그 연결 예측, 표준화, 내부 설정·모델 선택도 검산했다. 비교에 사용한 기존 RF 900개도 다시 불러 예측을 재현했다. 사람 분할·입력 해시·시간 일치·실제 정답 유지, 지표와 재표집 범위가 통과했다. 이것은 구현 검증이며 임상 검증과 구분한다.

재현 순서는 `check_inputs.py` → `fit_models.py` → `assess.py` → `check_results.py` → `finalize.py`이다. 완료된 학습은 덮어쓰지 않도록 막았다. 기존 Python과 설치된 scikit-learn을 사용했으며 새 패키지 설치·GPU 업로드·기존 연구 폴더 수정은 없었다. 환경은 `environment.json`, 파일 해시는 `artifact-manifest.json`에 있다.

주요 산출물: `all-comparisons.csv`, `summary.json`, 각 집단의 `long-people.csv`, `sub283-predictions.csv`, `extrapolation-comparison.png`, `input-verification.json`, `results-verification.json`.
'''
 (HERE/'STATUS.md').write_text(report,encoding='utf8');assert '\ufffd' not in report
 save('environment.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__,'device':'Windows CPU, four independent folds, one numerical thread'})
 for path in HERE.glob('*.py'):compile(path.read_text(encoding='utf8'),str(path),'exec')
 for cohort,info in protocol['cohorts'].items():
  for n,h in info['hashes'].items():assert sha(source(info)/n)==h
  assert len(list((HERE/cohort/'models').glob('*.joblib')))==150
 files=[p for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact-manifest.json'];save('artifact-manifest.json',{'new_models_verified':900,'RF_models_verified':900,'files':{p.relative_to(HERE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}})
 print(json.dumps({'finalized':True,'files':len(files),'sub283':extremes,'largest_case_share':share,'candidate_failures':failures}),flush=True)
if __name__=='__main__':main()
