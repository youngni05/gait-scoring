"""Standalone research figure from verified repeated held-out predictions."""
from common import *
import pandas as pd
import os
import importlib.util
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
if importlib.util.find_spec('matplotlib') is None:
    sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    runs=pd.read_csv(HERE/'run-metrics.csv');people=pd.read_csv(HERE/'person-error-analysis.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    choices=[('none','Median','대표값\n기준선'),('mean_speed','RF','평균 속도\nRF'),('speed_summary','RF','속도 요약\nRF'),('pelvis_motion','RF','골반 움직임\nRF'),('full','RF','전체 입력\nRF'),('full','CNN','전체 입력\nCNN')]
    colors=plt.cm.viridis(np.linspace(.15,.85,5))
    for x,(view,model,label) in enumerate(choices):
        g=runs[(runs.view==view)&(runs.model==model)]
        if g.empty:continue
        jitter=np.linspace(-.18,.18,len(g))
        axes[0].scatter(x+jitter,g.MAE,c=[colors[int(r)] for r in g['repeat']],s=30,alpha=.8)
        axes[0].plot([x-.24,x+.24],[g.MAE.mean()]*2,color='black',lw=2)
    axes[0].set_xticks(range(len(choices)),[c[2] for c in choices],fontsize=10)
    axes[0].set_ylabel('평균 절대오차 (초) — 낮을수록 좋음')
    axes[0].set_title('사람 분할 5개 × 학습 난수 3개\n색: 사람 분할 / 검은 선: 전체 반복 평균')
    for high,label,color in [(False,'나머지 238명','#3975a3'),(True,'큰 오차 상위 60명','#d86542')]:
        g=people[people.high_error==high]
        axes[1].scatter(g.target,g.mean_abs_error,s=26,alpha=.75,label=label,color=color)
    axes[1].set_xlabel('실제 TUG 시간 (초)')
    axes[1].set_ylabel('사람별 15회 절대오차의 평균 (초)')
    axes[1].set_title('전체 입력 RF: 반복해서 틀리는 사람\n상위 20%는 분석 기준이며 임상 위험 기준이 아님')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'stability-and-errors.png',dpi=170)
    plt.close(fig)

if __name__=='__main__':main()
