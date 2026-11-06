import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
scores=json.loads((ROOT/'person-scores.json').read_text());result=json.loads((ROOT/'results.json').read_text())
key='primary_age65plus';s=[r for r in scores['stroke'] if r['usable'] and r['characteristics']['Age']>=65];h=[r for r in scores['healthy'] if r['usable'] and r['characteristics']['Age']>=65]
x=np.array([r['scores'][key]['bilateral_mean'] for r in s]);y=np.array([r['characteristics']['POMA'] for r in s])
a=result['analyses'][key]['clinical_associations']['POMA'];lo,hi=a['bootstrap_95_percentile_ci']
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
axes[0].scatter(x,y,s=50,c='#2463a6',alpha=.8,edgecolors='white',linewidth=.6)
axes[0].set(xlabel='보행 편차 (도, 클수록 정상 기준과 다름)',ylabel='POMA (높을수록 기능이 좋음)',title=f'고령 뇌졸중 {len(s)}명: 편차와 기능 점수')
axes[0].text(.98,.97,f'순위상관 {a["spearman_rho"]:.2f}\n95% 범위 [{lo:.2f}, {hi:.2f}]',transform=axes[0].transAxes,ha='right',va='top',bbox={'facecolor':'white','edgecolor':'#cccccc','alpha':.9})
groups=[[r['scores'][key]['bilateral_mean'] for r in rows] for rows in [h,s]]
axes[1].boxplot(groups,tick_labels=[f'정상 {len(h)}명\n본인 제외 기준',f'뇌졸중 {len(s)}명'],showfliers=False,widths=.45)
rng=np.random.default_rng(20261103)
for i,values in enumerate(groups,1):axes[1].scatter(i+rng.normal(0,.035,len(values)),values,s=24,alpha=.65,color=['#509580','#2463a6'][i-1])
axes[1].set(ylabel='보행 편차 (도)',title='65세 이상: 사람별 점수 분포')
for ax in axes:ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('3D 관절각 기반 예비 실험 · 정상 기준 65세 이상',fontsize=14)
fig.savefig(ROOT/'pilot-results.png',dpi=180);plt.close(fig)
DATA=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-pilot')
curves=np.load(DATA/'person-curves.npz')['healthy'];link=json.loads((ROOT/'person-linkage.json').read_text())
ids=np.array([i for i,r in enumerate(link['healthy']['rows']) if r['values']['Age']>=65 and np.isfinite(curves[i]).all()]);valid=np.isfinite(curves).all(axis=(1,2,3));time=np.linspace(0,100,1001)
labels=['골반 전후 기울기','골반 좌우 기울기','골반 회전','엉덩관절 굽힘','엉덩관절 모음','엉덩관절 회전','무릎 굽힘','발목 굽힘','발 진행각']
fig,axs=plt.subplots(3,3,figsize=(12,8),layout='constrained')
for k,ax in enumerate(axs.ravel()):
 for j,name in [(0,'정상 65세 이상 · 왼쪽'),(1,'정상 65세 이상 · 오른쪽')]:ax.plot(time,curves[ids,j,k].mean(axis=0),label=name,lw=1.6)
 ax.plot(time,curves[valid,:,k].mean(axis=(0,1)),color='#666666',ls='--',label='정상 전체 · 양측 평균')
 ax.set(title=labels[k],xlabel='보행 주기 (%)',ylabel='도');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8);fig.suptitle('정상 기준 파형 확인: 각 다리의 착지부터 다음 착지까지')
fig.savefig(ROOT/'reference-curves.png',dpi=150);plt.close(fig)
print('Two plots saved')
