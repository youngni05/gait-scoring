from common import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=[r for r in json.loads((HERE/'geometry/manifest.json').read_text()) if r['target']>=20]
for page in range(3):
 fig,axes=plt.subplots(3,2,figsize=(12,8.5))
 for i,r in enumerate(rows[page*3:page*3+3]):
  a=read(HERE/'raw'/r['turn_file']);m=(a['time']>=r['turn_start'])&(a['time']<=r['turn_end']);ax=axes[i,0];ax.plot(a['pelvis_tx'],a['pelvis_tz'],color='gray');ax.plot(a['pelvis_tx'][m],a['pelvis_tz'][m],color='#ea580c',lw=2);ax.set_aspect('equal',adjustable='datalim');ax.set_title(r['id']+' turning region, '+str(round(r['turn_observed_seconds'],2))+' s');ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)')
  b=read(SOURCE/'raw'/r['walk_file']);t=b['time'];v=np.hypot(np.gradient(b['pelvis_tx'],t),np.gradient(b['pelvis_tz'],t));ax=axes[i,1];ax.plot(t,v,color='#2563eb',lw=1);ax.axvspan(r['old_walk_start'],r['old_walk_start']+3,color='#f97316',alpha=.2,label='Old 3 s');ax.axvspan(r['walk_start'],r['walk_end'],color='#16a34a',alpha=.2,label='Clean matched window');ax.set_xlabel('Walking record time (s)');ax.set_ylabel('Horizontal speed (m/s)');ax.set_title('Selected-window max speed '+str(round(r['walk_max_speed'],2))+' m/s');ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(HERE/f'geometry-quality-{page+1}.png',dpi=125);plt.close(fig)
print('GEOMETRY_REVIEW_PAGES',3)
