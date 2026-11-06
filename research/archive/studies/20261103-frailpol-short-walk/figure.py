from pathlib import Path
import json,os
H=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');B=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103');os.environ['MPLCONFIGDIR']=str(B/'cache/mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=json.loads((B/'heldout-predictions.json').read_text());summary=json.loads((H/'comparison.json').read_text())['results']['tug']['models']['TabICLv2']
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
for ax,condition,title in zip(axes,['speed','features'],['속도만','속도와 보행 특징']):
    r=[r for r in p if r['target']=='tug' and r['kind']=='TabICLv2' and r['condition']==condition];y=np.array([r['actual'] for r in r]);pred=np.array([r['prediction'] for r in r]);ax.scatter(y,pred,c=np.where(y>=20,'#b1432f','#267db0'),s=18,alpha=.65,edgecolors='none');ax.plot([0,38],[0,38],'--',c='gray',lw=1);ax.set_xlim(4,38);ax.set_ylim(4,38);ax.set_title(title+'\n평균오차 '+f"{summary[condition]['mae']:.2f}초");ax.set_xlabel('실제 TUG 시간 (초)');ax.grid(alpha=.15)
axes[0].set_ylabel('예측한 TUG 시간 (초)');fig.suptitle('594명의 학습 제외 예측 · TabICLv2',y=1.01);fig.text(.5,-.02,'점선에 가까울수록 정확하다. 빨간 점은 실제 TUG 20초 이상인 9명이다.',ha='center',fontsize=10);fig.tight_layout();fig.savefig(H/'tug-predictions.png',dpi=150,bbox_inches='tight');plt.close(fig)
