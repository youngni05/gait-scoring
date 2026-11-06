"""Per-video values and within-video window ranges; no confidence-interval claims."""
import sys,os,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'));os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
rows=json.loads((HERE/'results.json').read_text(encoding='utf8'))['records'];fig,ax=plt.subplots(figsize=(11,4.8))
for r in rows:
    if r['score'] is None:continue
    c='#277da1' if r['condition']=='normal' else '#d97051';i=r['number'];s=r['score']
    ax.plot([i,i],[r['window_min'],r['window_max']],color=c,alpha=.5,lw=3)
    ax.scatter(i,s,color=c,s=60,zorder=3);ax.annotate('%.2f'%s,(i,s),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10,color=c)
ax.set_xticks(range(1,14));ax.set_xlabel('영상 번호 (파랑: 정상 1·2·3·6·8·9번 / 주황: 비정상 4·5·7·10·11·12·13번)');ax.set_ylabel('고정 모델의 GDI 예측값');ax.set_title('Test01 영상별 CNN·STT 결합 모델 점수',fontsize=15,pad=18);ax.grid(axis='y',alpha=.2)
ax.set_xlim(.4,13.6);low=min(r['window_min'] for r in rows if r['score'] is not None);high=max(r['window_max'] for r in rows if r['score'] is not None);ax.set_ylim(low-5,high+6)
fig.text(.5,.025,'점: 영상 평균 / 세로선: 같은 영상의 구간 점수 범위. 실제 GDI 정답과 임상적 판별 정확도는 미확인이다.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(HERE/'video-scores.png',dpi=180);plt.close(fig)
