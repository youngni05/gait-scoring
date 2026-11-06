"""Static scientific comparison: windows are repeated observations of one person."""
import os,sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'tmp/ljmu-analysis-python-deps')]
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib-cache')
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
r=json.loads((HERE/'results.json').read_text())['summary'];fig,axes=plt.subplots(1,2,figsize=(10,4.8),sharey=True)
for ax,kind,title in zip(axes,['primary','straight'],['기존 입력 구간: 앞 약 16.5초','미리 고정한 직선 보행 구간']):
    s=r[kind]['best']
    for x,clip,label,color in [(0,'clip1','정상','#277da1'),(1,'clip0','비정상','#df7053')]:
        vals=np.array(s[clip]['window_means']);jitter=np.linspace(-.10,.10,len(vals))
        ax.scatter(x+jitter,vals,color=color,s=34,alpha=.8,label='124프레임 구간')
        ax.plot([x-.24,x+.24],[s[clip]['mean']]*2,color=color,lw=3)
        ax.text(x,s[clip]['mean']+1.4,'평균 %.2f'%s[clip]['mean'],ha='center',fontsize=11,color=color,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    ax.set_xticks([0,1],['정상','비정상']);ax.set_title(title,fontsize=12,pad=12);ax.set_xlim(-.5,1.5);ax.grid(axis='y',alpha=.2)
    ax.text(.5,.04,'정상 - 비정상: +%.2f점'%s['normal_minus_abnormal'],transform=ax.transAxes,ha='center',fontsize=12)
axes[0].set_ylabel('고정 모델의 GDI 예측값');axes[0].set_ylim(55,86)
fig.suptitle('같은 사람의 두 영상: 기존 최선 CNN·STT 결합',fontsize=15)
fig.text(.5,.02,'점은 같은 사람의 반복 구간이며 독립 참가자가 아니다. 실제 GDI 정답·임상 구분 성능은 미확인이다.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(HERE/'score-comparison.png',dpi=180);plt.close(fig)
