from common import *
sys.path.append(str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
audit=json.loads((HERE/'turn-audit.json').read_text());old=pd.read_csv(PREV/'manifest.csv');people=old[old.target>=20].to_dict('records');records=[]
for page in range(3):
 fig,axes=plt.subplots(3,3,figsize=(15,9))
 for i,m in enumerate(people[page*3:page*3+3]):
  r=next(x for x in audit if x['id']==m['id']);a=read(HERE/'raw'/r['file']);t=a['time'];yaw=np.rad2deg(np.unwrap(np.deg2rad(a['pelvis_rotation'])));yaw-=np.median(yaw[:max(1,round(.5/np.median(np.diff(t))))])
  axes[i,0].plot(a['pelvis_tx'],a['pelvis_tz'],lw=1);axes[i,0].set_aspect('equal',adjustable='datalim');axes[i,0].set_title(m['id']+' TUG path; clinical '+str(m['target'])+'s');axes[i,0].set_xlabel('X (m)');axes[i,0].set_ylabel('Z (m)')
  axes[i,1].plot(t,yaw,lw=1);axes[i,1].set_title('TUG yaw: '+r['detection']);axes[i,1].set_xlabel('Recording time (s)');axes[i,1].set_ylabel('Pelvis yaw (deg)')
  if r['detection']=='passed':axes[i,1].axvspan(r['start'],r['end'],color='orange',alpha=.2)
  b=read(SOURCE/'raw'/m['file']);tt=b['time'];v=np.hypot(np.gradient(b['pelvis_tx'],tt),np.gradient(b['pelvis_tz'],tt));wy=np.rad2deg(np.unwrap(np.deg2rad(b['pelvis_rotation'])));wy-=wy[0]
  ax=axes[i,2];ax.plot(tt,v,color='tab:blue',label='horizontal speed');ax.set_ylabel('Walking speed (m/s)',color='tab:blue');ax.set_xlabel('Recording time (s)');ax.axvspan(m['start_time'],m['start_time']+3,color='orange',alpha=.2);ax.set_title('Original walking clip')
  twin=ax.twinx();twin.plot(tt,wy,color='tab:red',alpha=.6);twin.set_ylabel('Yaw (deg)',color='tab:red')
  records.append({'id':m['id'],'target':m['target'],'turn_file':r['file'],'detection':r['detection'],'image':f'long-quality-{page+1}.png','row':i+1})
 fig.tight_layout();fig.savefig(HERE/f'long-quality-{page+1}.png',dpi=120);plt.close(fig)
save('long-quality-review-selection.json',records);print('LONG_QUALITY_PAGES',3)
