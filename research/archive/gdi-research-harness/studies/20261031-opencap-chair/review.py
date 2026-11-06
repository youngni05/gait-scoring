from context import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows=json.loads((PRIOR/'geometry/manifest.json').read_text());audit={r['id']:r for r in json.loads((STUDY/'transition-audit.json').read_text())};manifest={r['id']:r for r in json.loads((STUDY/'event_strict/manifest.json').read_text())}
long=[r for r in rows if r['target']>=20];normal=sorted([r for r in manifest.values() if r['target']<20],key=lambda r:r['observed_seconds']);chosen=long+[normal[i] for i in np.linspace(0,len(normal)-1,6).astype(int)];save('review-selection.json',[m['id'] for m in chosen])
for page in range(5):
 fig,axes=plt.subplots(3,1,figsize=(13,10))
 for ax,m in zip(axes,chosen[page*3:page*3+3]):
  a=read(PRIOR/'raw'/m['turn_file']);t=a['time'];ax.plot(t,a['pelvis_ty'],label='height');ax.axvspan(m['turn_start'],m['turn_end'],alpha=.2,color='orange',label='turn')
  for k,color in [('rise','green'),('sit','purple')]:
   if 'window' in audit[m['id']][k]:ax.axvspan(*audit[m['id']][k]['window'],color=color,alpha=.2,label=k)
  if m['id'] in manifest:
   r=manifest[m['id']];ax.axvspan(*r['context_window'],facecolor='none',edgecolor='black',ls='--',label='matched control')
  ax.set_title(m['id']);ax.set_xlabel('record time (s)');ax.set_ylabel('height (m)');ax.legend(loc='upper right',fontsize=8);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(STUDY/f'final-phase-review-{page+1}.png',dpi=110);plt.close(fig)
print('REVIEW_WRITTEN',len(chosen),flush=True)
