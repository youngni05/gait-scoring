"""Write the requested comparison and fixed scientific figures after verification."""
from prepare import *
import numpy as np
LABEL={'single_CNN':'Single CNN','multi_STT':'Multi STT','multi_CNN':'Multi CNN','CNN_bag':'CNN bag','fusion':'Fusion','plus_general':'Fusion + general','plus_ranksim':'Fusion + RankSim','MSE':'MSE','Rank1':'RankSim','HuberOnly':'Huber','HuberRank1':'Huber + RankSim','MAEOnly':'MAE','MAERank1':'MAE + RankSim','plus_HuberOnly':'Fusion + Huber','plus_HuberRank1':'Fusion + HuberRank','plus_MAEOnly':'Fusion + MAE','plus_MAERank1':'Fusion + MAERank'}
def fmt(x):return '해당 없음' if x is None else f'{x:.4f}'
def main():
 v=read(H/'verification.json');assert v['passed'] and sha(H/'results.json')==v['results_sha256'];r=read(H/'results.json');p=read(H/'protocol.json');names=r['models']
 lines=['# 기존 GDI 모델의 특수 사례 비교','', '같은 222명·377기록에서 기존 학습 모델 94개로 구성한 17개 예측 방식을 비교했다. 새 학습이나 보정계수 조정은 하지 않았다. 이미 결과를 본 표본을 다시 분석한 **사후 진단**이다. 새로운 미사용 최종 시험이나 임상 검증이 아니다.','', '오차(MAE)는 낮을수록 정답에 가깝다. 편향은 예측−정답이며 음수이면 과소예측이다. 사람별로 기록과 좌우를 먼저 평균해, 기록이 많은 사람의 영향이 커지지 않게 했다.','', '## 전체·극단값·같은 대상 변화','', '|모델|전체 MAE|90 이상 MAE|90 이상 편향|60 미만 MAE|영상 안 구간 SD|다른 기록 변화 오차|','|---|---:|---:|---:|---:|---:|---:|']
 for n in names:
  z=r['full'][n];lines.append('|'+LABEL[n]+'|'+ '|'.join(fmt(x) for x in [z['all']['mae'],z['90plus']['mae'],z['90plus']['bias'],z['below60']['mae'],r['native_stability'][n]['sample_sd_min3']['mean'],r['repeat_records'][n]['delta_error_MAE']['mean']])+'|')
 lines+=['','고득점 구분은 각 사람의 실제 좌우·모든 기록 평균이 90 이상인 경우다. 세부 인원은 다음과 같다.','']
 for g,z in r['groups'].items():lines.append(f'- {g}: {z["people"]}명, {z["records"]}기록')
 lines+=['','관측 점수 자체가 90·100 이상인 좌우 관측만 따로 집계한 결과는 results.json의 observation_groups에 있다. 사람 평균 기준과 표본 수가 다르다.','', '영상 안 구간 SD는 같은 영상에서 124프레임씩 위치를 바꿨을 때 점수의 표준편차다. 3구간 이상인 기록으로 계산했다. 값이 작아도 정확도가 좋다는 뜻은 아니다. 항상 비슷한 오답을 내는 모델도 안정적으로 보일 수 있다.','', '다른 기록 변화 오차는 동일인의 모든 기록 쌍에서 (예측 변화−실제 변화)의 절댓값을 계산한 것이다. 검사 사이에 실제 보행이 달라질 수 있으므로 재촬영 반복정밀도나 ICC로 해석하지 않는다. 날짜 순서를 가정하지 않았다.','', '## 사용할 수 있는 구간을 줄였을 때','', '모든 모델은 기본적으로 124프레임을 입력받는다. 이미 전처리된 자료에서 처음 유효 구간부터 124·186·248·372·496프레임 범위 안의 완전한 구간만 평균했다. 모든 길이에서 같은 222명·377기록을 유지했다. 원래 Single CNN은 별도 고정 구간을 쓰므로 이 평균 구간 수 비교에서 제외하고 위의 공통 구간 위치 검사에만 포함했다.','', '**원본 영상을 짧게 잘라 전처리부터 다시 실행한 시험이 아니다.** 전체 기록 전처리 결과를 재사용하므로 구간 평균 수에 대한 의존성을 확인한 것이다. 124프레임 미만은 임의로 늘리지 않고 측정 불가로 둔다. 초 단위 환산이나 실제 스마트폰 촬영 성능을 뜻하지 않는다.','', '|모델|124 MAE|186 MAE|248 MAE|372 MAE|496 MAE|124−전체 MAE [설명용 95% 구간]|','|---|---:|---:|---:|---:|---:|---|']
 for n in names:
  if n=='single_CNN':continue
  z=r['shorter_input'][n];delta=z['124']['mae_difference_from_full'];lines.append('|'+LABEL[n]+'|'+ '|'.join(fmt(z[str(b)]['metrics']['all']['mae']) for b in [124,186,248,372,496])+f'|{fmt(delta["mean"])} [{fmt(delta["ci95"][0])}, {fmt(delta["ci95"][1])}]|')
 lines+=['','## 모델들이 같은 대상에 비슷하게 반응하는가','', '|비교|전체 점수 차이(MAD)|영상 내 구간 변화 상관|영상 내 중심화 반응 차이|','|---|---:|---:|---:|']
 fixed=[('fusion','plus_ranksim'),('plus_ranksim','plus_general'),('fusion','plus_HuberRank1'),('fusion','plus_MAERank1'),('MSE','Rank1'),('HuberOnly','HuberRank1'),('MAEOnly','MAERank1')]
 for a,b in fixed:
  ag=next(q for q in r['model_agreement'] if {q['model_a'],q['model_b']}=={a,b});rr=next(q for q in r['within_record_response'] if {q['model_a'],q['model_b']}=={a,b});lines.append(f'|{LABEL[a]} / {LABEL[b]}|{fmt(ag["person_equal_MAD"])}|{fmt(rr["centered_window_response_correlation"])}|{fmt(rr["centered_window_MAD"])}|')
 lines+=['', '영상 내 상관은 각 기록의 평균 점수를 뺀 뒤, 같은 구간 위치 변화에 대한 반응만 비교한다. 사람마다 점수 수준이 달라 생기는 높은 상관과 구분한다. 두 모델이 비슷하게 틀릴 수 있으므로 일치도와 정확도를 함께 봐야 한다. 전체 모델 쌍 136개의 결과는 model-agreement.csv와 within-record-response.csv에 있다.','', '## 확인 범위와 파일','', '- 좌우 실제 차이가 5·10 이상인 기록의 예측 차이 오차, 고득점 100 이상, 저득점 50 미만, 초기·중간·마지막 구간, 다른 기록 변화 방향은 results.json에 고정 기준 그대로 포함했다. 해당 인원이 없으면 빈 집단으로 기록했다.','- 불확실성 구간은 사람 단위 10,000회 재추출(seed 20261105)로 계산했다. 여러 사후 비교 중 우수한 결과만 골라 새 최선 모델로 승격하지 않는다.','- 기존 7개 방식의 모든 최종 예측을 0.0001점 이내로 재현했고, 94개 모델의 기존 예측 대조·학습/평가 사람 분리·입력 연결·원본 파일 지문과 독립 계산을 확인했다. 검산 근거는 verification.json이다.','- 대량 구간 예측과 개별 모델 기록: D:/gait-research-data/D_P/gdi-case-audit-20261105. 원래 결과는 변경하지 않았다.','- 전체·고득점 표: model-comparison.csv, 길이별 표: shorter-input.csv, 기계 판독 전체 결과: results.json.','']
 (H/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'figure.dpi':140})
 with np.load(BIG/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(BIG/'case-details.npz',allow_pickle=False) as z:y=z['y']
 fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
 for n in ['multi_CNN','multi_STT','CNN_bag','fusion','plus_ranksim','plus_HuberRank1','plus_MAERank1']:
  bs=[124,186,248,372,496];axes[0].plot(bs,[r['shorter_input'][n][str(b)]['metrics']['all']['mae'] for b in bs],marker='o',label=LABEL[n])
 for n in names:
  x=r['native_stability'][n]['sample_sd_min3']['mean'];yy=r['full'][n]['all']['mae'];axes[1].scatter(x,yy);axes[1].annotate(LABEL[n],(x,yy),fontsize=7,xytext=(3,2),textcoords='offset points')
 axes[0].set(xlabel='Prepared-input frame budget',ylabel='Person-equal MAE (GDI points)',title='Fewer complete windows; same 222 people');axes[0].legend(fontsize=7);axes[1].set(xlabel='Within-record window SD (GDI points)',ylabel='Person-equal full MAE (GDI points)',title='Stability and accuracy are separate');fig.savefig(H/'length-and-stability.png');plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(11,8),layout='constrained')
 for row,record in enumerate(p['figures']['case_indices']):
  ix=np.flatnonzero(d['window_record']==record)
  for side in [0,1]:
   ax=axes[row,side]
   for n in p['figures']['predeclared_representatives']:ax.plot(d['window_start'][ix],d['window__'+n][ix,side],marker='.',label=LABEL[n])
   ax.axhline(y[record,side],color='black',ls='--',label='Actual GDI');ax.set(title=f'Fixed record {record}, side {side+1}',xlabel='Native 124-frame window start',ylabel='GDI score')
 axes[0,0].legend(fontsize=7);fig.savefig(H/'fixed-window-responses.png');plt.close(fig)
 save(H/'report-receipt.json',dict(passed=True,report_sha256=sha(H/'README.md'),verification_sha256=sha(H/'verification.json'),results_sha256=sha(H/'results.json'),figures={n:sha(H/n) for n in ['length-and-stability.png','fixed-window-responses.png']}));print('Report and figures saved',flush=True)
if __name__=='__main__':main()
