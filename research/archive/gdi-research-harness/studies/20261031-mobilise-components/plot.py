"""Plot all six observation conditions and the within-gait comparison."""
from pathlib import Path
import sys, importlib.util
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
if importlib.util.find_spec('matplotlib') is None:
    sys.path.insert(0,str(HERE.parents[2]/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font=Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
m=pd.read_csv(HERE/'aggregate-metrics.csv');m=m[m.draw.eq(0)]
fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
conditions=[(b,k) for b in ['short','long'] for k in [1,3,5]]
names=['4–10초\n1개','4–10초\n3개','4–10초\n5개','30–60초\n1개','30–60초\n3개','30–60초\n5개']
colors={'direct6':'#606b75','RF_components':'#157f9e','DS_components':'#d36430'}
labels={'direct6':'기존 총점 Deep Sets','RF_components':'영역별 RF의 합','DS_components':'세 영역 공동 Deep Sets'}
for ax,group,metric,title in [(axes[0,0],'all','mae','전체 SPPB 총점 MAE'),(axes[0,1],'low_sppb','mae','낮은 SPPB(≤3) 총점 MAE'),(axes[1,0],'low_sppb','p90','낮은 SPPB의 큰 오차: 절대오차 90백분위')]:
    for model in colors:
        d=m[m.endpoint.eq('total')&m.group.eq(group)&m.model.eq(model)].set_index(['band','k']).loc[conditions]
        y=d[metric+'_mean'].to_numpy();lo=d[metric+'_min'].to_numpy();hi=d[metric+'_max'].to_numpy()
        ax.errorbar(np.arange(6),y,yerr=np.maximum([y-lo,hi-y],0),marker='o',capsize=3,color=colors[model],label=labels[model])
    ax.set_xticks(np.arange(6),names);ax.set_title(title,fontweight='bold');ax.set_ylabel('오차(점), 낮을수록 좋음');ax.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=9)
ax=axes[1,1]
for model,color,label in [('gait_lookup','#606b75','실측 걷기 점수 기준선'),('RF','#157f9e','영역별 RF'),('DS','#d36430','공동 Deep Sets')]:
    d=m[m.band.eq('long')&m.k.eq(5)&m.endpoint.eq('remaining')&m.model.eq(model)&m.group.isin([f'gait_{i}' for i in range(5)])].set_index('group').loc[[f'gait_{i}' for i in range(5)]]
    y=d.mae_mean.to_numpy();ax.errorbar(np.arange(5),y,yerr=np.maximum([y-d.mae_min.to_numpy(),d.mae_max.to_numpy()-y],0),marker='o',capsize=3,color=color,label=label)
ax.set_xticks(np.arange(5),[f'걷기 {i}점\n({int(n)}명)' for i,n in enumerate(d.people)])
ax.set_title('같은 걷기 점수 안의 균형+일어서기 MAE\n30–60초 구간 5개',fontweight='bold')
ax.set_ylabel('나머지 두 점수 합계의 오차(점)');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
fig.suptitle('보행 특징으로 SPPB 세 영역을 예측한 결과\n같은 2,174명·구간·사람 분할, 초기값 5개',fontsize=17,fontweight='bold')
fig.supxlabel('점과 선: 초기값 5개의 평균 성능 / 오차 막대: 초기값 간 최솟값–최댓값(신뢰구간 아님)\n원점수 회귀 결과. 센서 보행 요약 특징 실험이며 영상·외부 임상 검증은 아니다.',fontsize=10)
fig.savefig(HERE/'component-comparison.png',dpi=170)
print('FIGURE_SAVED',flush=True)
