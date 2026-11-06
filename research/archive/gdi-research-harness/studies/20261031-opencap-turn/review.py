from common import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
audit=json.loads((HERE/'turn-audit.json').read_text());manifest=json.loads((HERE/'event_matched/manifest.json').read_text());chosen={r['file']:r for r in audit if r['detection']=='passed'}
labels=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit'])
names=[]
for r in manifest:
 if r['target']>=20:names.append(r['turn_file'])
for r in sorted(chosen.values(),key=lambda r:r['turn_seconds'])[::max(1,len(chosen)//6)]:names.append(r['file'])
for r in sorted(chosen.values(),key=lambda r:r['raw_yaw_jump'],reverse=True)[:2]:names.append(r['file'])
names=list(dict.fromkeys(names));records=[]
for page in range((len(names)+4)//5):
 batch=names[page*5:(page+1)*5];fig,axes=plt.subplots(len(batch),2,figsize=(12,2.5*len(batch)),squeeze=False)
 for i,name in enumerate(batch):
  r=chosen[name];a=read(HERE/'raw'/name);t=a['time'];yaw=np.rad2deg(np.unwrap(np.deg2rad(a['pelvis_rotation'])));yaw-=np.median(yaw[:max(1,int(.5/np.median(np.diff(t))))]);m=(t>=r['start'])&(t<=r['end'])
  ax=axes[i,0];ax.plot(a['pelvis_tx'],a['pelvis_tz'],lw=1,color='#64748b');ax.plot(a['pelvis_tx'][m],a['pelvis_tz'][m],color='#e8791b',lw=2);ax.scatter(a['pelvis_tx'][0],a['pelvis_tz'][0],color='green',s=12);ax.set_aspect('equal',adjustable='datalim');ax.set_title(r['id']+' / '+r['task']+' / turn '+str(round(r['turn_seconds'],2))+'s',fontsize=10);ax.set_xlabel('Pelvis X (m)');ax.set_ylabel('Pelvis Z (m)')
  ax=axes[i,1];ax.plot(t,yaw,color='#2563eb',lw=1);ax.axvspan(r['start'],r['end'],color='#fed7aa',alpha=.7);ax.axvline(r['mid'],color='#e8791b',ls='--');ax.set_xlabel('Recording time (s)');ax.set_ylabel('Unwrapped pelvis yaw (deg)');ax.grid(alpha=.2)
  records.append({'file':name,'image':f'quality-review-{page+1}.png','row':i+1,'detected_start':r['start'],'detected_end':r['end']})
 fig.suptitle('Kinematic turn-window review; not original RGB validation',fontsize=12);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(HERE/f'quality-review-{page+1}.png',dpi=130);plt.close(fig)
save('quality-review-selection.json',records);print('REVIEW_PAGES',(len(names)+4)//5,'RECORDS',len(names))
