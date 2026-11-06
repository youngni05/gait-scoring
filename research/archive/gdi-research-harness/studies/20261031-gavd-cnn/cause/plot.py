import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
dev=json.loads((HERE/'development.json').read_text());s=json.loads((HERE/'sensitivity_summary.json').read_text())['variants'];h=json.loads((HERE/'highres_summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
b=dev['bins'];truth=[v['true_mean'] for v in b];held=[v['oof_mean'] for v in b]
axes[0].plot([50,110],[50,110],ls='--',color='.65',label='Ideal agreement');axes[0].plot(truth,held,'o-',label='Held-out predictions');axes[0].plot(truth,[v['training_mean'] for v in b],'s:',label='Training predictions');axes[0].set_xlabel('True GDI bin mean');axes[0].set_ylabel('Frozen CNN score');axes[0].set_title('A. Compression in development data');axes[0].legend(fontsize=8)
variants=['translate','scale_half','horizontal_reflection','time_0.8','time_1.2','confidence_mask_0_1','jitter_bbox_1pct','toe_gap10']
labels=['Translation (25)','Scale x0.5 (25)','Reflection (25)','Time step x0.8 (13)','Time step x1.2 (13)','Low-confidence mask (25)','1% body-height jitter (25)','10-frame toe gap (25)']
axes[1].barh(labels,[s[k]['mean_abs_delta'] for k in variants],color=['#4477aa']*5+['#cc6677']*3);axes[1].invert_yaxis();axes[1].set_xlabel('Mean absolute score change');axes[1].set_title('B. Controlled input sensitivity')
for i,r in enumerate(h['records']):
    color='#4477aa' if r['label']=='Normal Gait' else '#cc6677';axes[2].plot([0,1],[r['baseline_score'],r['highres_score']],'o-',color=color,label=r['label'] if i in [0,2] else None);axes[2].annotate(r['video_id'],(1,r['highres_score']),xytext=(5,0),textcoords='offset points',fontsize=7)
axes[2].set_xticks([0,1],['368 px','544 px']);axes[2].set_xlim(-.2,1.85);axes[2].set_xlabel('OpenPose network input height');axes[2].set_ylabel('Frozen CNN score');axes[2].set_title('C. Same four videos, re-extracted poses')
for ax in axes:ax.grid(axis='y',alpha=.15)
fig.suptitle('Cause diagnostics: unchanged CNN, no training or score-based selection\nInput sensitivity and re-extraction are not anatomical ground truth',fontsize=12)
fig.savefig(HERE/'cause_summary.png',dpi=180);fig.savefig(HERE/'cause_summary.svg');plt.close(fig)
