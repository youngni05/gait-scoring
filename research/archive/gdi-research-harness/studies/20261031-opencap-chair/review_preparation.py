from context import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=json.loads((STUDY/'prep_event/manifest.json').read_text());chosen=[r for r in rows if r['target']>=20];other=sorted([r for r in rows if r['target']<20],key=lambda r:r['core_rise'][0]-r['phase_windows']['rise'][0]);chosen+=other[-2:]
for page in range(2):
 fig,axes=plt.subplots(3,1,figsize=(13,10))
 for ax,r in zip(axes,chosen[page*3:page*3+3]):
  a=read(PRIOR/'raw'/r['turn_file']);t=a['time'];ax.plot(t,a['pelvis_ty'],label='pelvis height');ax.axvspan(*r['phase_windows']['rise'],color='green',alpha=.15,label='preparation + rise');ax.axvspan(*r['core_rise'],color='green',alpha=.3,label='rise core');ax.axvspan(r['turn_start'],r['turn_end'],color='orange',alpha=.2,label='turn');ax.axvspan(*r['phase_windows']['sit'],color='purple',alpha=.2,label='sit');ax.axvspan(*r['context_window'],facecolor='none',edgecolor='black',ls='--',label='matched control');ax.set_title(r['id']+f" / selected input {r['observed_seconds']:.2f}s");ax.set_xlabel('record time (s)');ax.set_ylabel('height (m)');ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(STUDY/f'preparation-review-{page+1}.png',dpi=110);plt.close(fig)
save('preparation-review-selection.json',{'people':[r['id'] for r in chosen],'flagged_angles':[r['id'] for r in rows if r['flagged_angle']]});print('PREPARATION_REVIEW_WRITTEN',flush=True)
