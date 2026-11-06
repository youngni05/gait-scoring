from common import *
import os,importlib.util
os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig')
if importlib.util.find_spec('matplotlib') is None:sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

def main():
    font=Path('C:/Windows/Fonts/malgun.ttf')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    s=json.loads((HERE/'summary.json').read_text());cause=json.loads((HERE/'cause-summary.json').read_text());case=cause['largest_TUG_person']
    tr=json.loads((HERE/'long-trajectories.json').read_text())[case['id']]
    fig,ax=plt.subplots(1,2,figsize=(15,6),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    ax[0].plot(tr['time_from_window_start'],tr['speed'],color='#c75b3b',lw=1.7)
    ax[0].axhline(case['speed_mean'],ls='--',color='#385b83',label=f"평균 {case['speed_mean']:.2f}")
    ax[0].axhline(case['speed_median'],ls=':',color='#2b8b79',label=f"중앙값 {case['speed_median']:.2f}")
    ax[0].set(xlabel='선택한 3초 구간의 시간 (초)',ylabel='좌표에서 계산한 골반 수평 속도 (m/s)',title='실제 TUG 42.59초 사례의 속도 급변\n좌표 추정 오류 가능성, 영상으로 확정한 것은 아님')
    ax[0].legend(frameon=False)
    labels={'BaselineRF':'기준 RF','RFSelectBalanced':'균형 선택 RF','RobustRF':'중앙값·IQR RF','RobustRFSelectBalanced':'중앙값·IQR + 균형 선택','WeightedRF3':'가중치 3 RF','RobustWeightedRF3':'중앙값·IQR + 가중치 3','WeightedRF':'가중치 10 RF','RobustWeightedRF':'중앙값·IQR + 가중치 10','InverseRidge':'역속도 회귀','InverseRidgeBalanced':'역속도 회귀 + 균형 선택'}
    for i,(method,label) in enumerate(labels.items()):
        r=s['models'][method];m=r['long'];lo,hi=m['MAE_range']
        ax[1].errorbar(m['MAE'],i,xerr=[[m['MAE']-lo],[hi-m['MAE']]],fmt='o',color='#2c7892' if method!='BaselineRF' else '#333333',capsize=3)
    ax[1].set_yticks(range(len(labels)),[f"{label} (전체 {s['models'][method]['all']['MAE']:.2f}초)" for method,label in labels.items()],fontsize=10)
    ax[1].invert_yaxis();ax[1].set(xlabel='긴 TUG 9명의 평균 절대오차 (초)',title='방법별 긴 TUG 오차와 전체 오차\n점: 반복 평균 / 선: 반복별 최솟값~최댓값')
    for a in ax:a.grid(alpha=.2);a.spines[['top','right']].set_visible(False)
    fig.savefig(HERE/'tail-causes-and-tradeoffs.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
