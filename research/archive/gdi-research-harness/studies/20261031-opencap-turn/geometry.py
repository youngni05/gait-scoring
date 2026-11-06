"""Trajectory-based sensitivity declared from input-quality findings, not scores."""
from common import *
from scipy.ndimage import uniform_filter1d
from prepare import old,walk_window
NAMES=['turn_seconds','horizontal_speed_mean','horizontal_speed_std','horizontal_speed_change','path_length','lateral_range','direction_change_deg','direction_speed_mean','direction_speed_p95','slow_fraction']
def resample(a,start,end):
 t=a['time'];lo=np.searchsorted(t,start-1e-7);hi=np.searchsorted(t,end-1e-7);assert hi<len(t)
 tt=t[lo:hi+1];grid=tt[0]+np.arange(round((end-start)*30)+1)/30
 assert grid[-1]<=tt[-1]+1e-6
 return np.vstack([np.interp(grid,tt,a[n][lo:hi+1]) for n in ['pelvis_tx','pelvis_tz']]).T
def root_qc(a,start,end):
 part=a[(a['time']>=start-1e-7)&(a['time']<=end+1e-7)];pos=np.column_stack([part['pelvis_tx'],part['pelvis_tz']]);speed=np.linalg.norm(np.diff(pos,axis=0),axis=1)/np.diff(part['time'])
 return float(max(speed))
def trajectory_turn(a):
 t=a['time'];dt=np.median(np.diff(t));assert np.all(np.diff(t)>0)
 if max(np.diff(t))>.05:raise ValueError('time_gap')
 pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);smooth=uniform_filter1d(pos,max(3,round(.25/dt)),axis=0,mode='nearest');origin=np.median(smooth[:max(1,round(.5/dt))],axis=0)
 distance=np.linalg.norm(smooth-origin,axis=1);far=np.median(smooth[distance>=np.quantile(distance,.97)],axis=0);axis=far-origin;axis/=np.linalg.norm(axis)
 progress=(smooth-origin)@axis;extent=float(np.quantile(progress,.99))
 if extent<1.5:raise ValueError('short_excursion')
 mid=int(np.argmax(progress));level=extent-.75;before=np.flatnonzero(progress[:mid]<=level);after=np.flatnonzero(progress[mid:]<=level)
 if not len(before) or not len(after) or min(progress[mid:])>.65*extent:raise ValueError('incomplete_return')
 start=float(np.ceil(t[before[-1]]*30-1e-6)/30);end=float(np.floor(t[mid+after[0]]*30+1e-6)/30)
 if not .4<=end-start<=27:raise ValueError('turn_interval_duration')
 peak=root_qc(a,start,end)
 if peak>6:raise ValueError('turn_root_speed')
 return {'start':start,'end':end,'mid':float(t[mid]),'axis':axis.tolist(),'extent':extent,'max_speed':peak}
def features(a,start,end,axis):
 pos=resample(a,start,end);velocity=np.gradient(pos,1/30,axis=0);v=np.linalg.norm(velocity,axis=1)
 smooth=uniform_filter1d(pos,7,axis=0,mode='nearest');sv=np.gradient(smooth,1/30,axis=0);theta=np.unwrap(np.arctan2(sv[:,1],sv[:,0]));omega=abs(np.gradient(theta,1/30))*180/np.pi;moving=np.linalg.norm(sv,axis=1)>.1
 lateral=pos@np.array([-axis[1],axis[0]])
 first=np.mean(sv[:min(10,len(sv))],axis=0);last=np.mean(sv[-min(10,len(sv)):],axis=0);angle=np.degrees(np.arccos(np.clip(np.dot(first,last)/max(np.linalg.norm(first)*np.linalg.norm(last),1e-12),-1,1)))
 return np.array([end-start,v.mean(),v.std(),abs(np.diff(v)).mean(),np.linalg.norm(np.diff(pos,axis=0),axis=1).sum(),np.ptp(lateral),angle,omega[moving].mean(),np.quantile(omega[moving],.95),np.mean(v<.2)])
def choose_walk(a,budget,preferred):
 spans=old.active(a);candidates=[preferred]
 for lo,hi,dur in spans:
  if dur>=budget:candidates.extend(np.arange(a['time'][lo],a['time'][hi]-budget+1e-7,1/30))
 for start in candidates:
  idx=int(np.searchsorted(a['time'],start-1e-7));start=float(a['time'][idx]);end=start+budget
  if not any(a['time'][lo]<=start+1e-6 and a['time'][hi]>=end-1e-6 for lo,hi,_ in spans):continue
  speed=root_qc(a,start,end)
  if speed>6:continue
  pos=resample(a,start,end);smooth=uniform_filter1d(pos,7,axis=0,mode='nearest');distance=np.linalg.norm(smooth[-1]-smooth[0]);length=np.linalg.norm(np.diff(smooth,axis=0),axis=1).sum();ratio=distance/max(length,1e-9)
  if ratio>=.85:return start,speed,float(ratio)
 raise ValueError('no_clean_matched_straight_window')
def main():
 out=HERE/'geometry';out.mkdir(exist_ok=True);assert not (HERE/'preparation-geometry.json').exists()
 views=['OldWalk3Speed','Walk3Speed','WalkMatchedSpeed','Walk3TurnTime','Walk3TurnShape','Walk3Turn']
 protocol={'declared_before_geometry_training':True,'motivation':'Visual review of all9 long people showed far-end trajectory turns even when pelvis yaw failed; worst prior walking input has initial root-position jumps. This is an exploratory method addition after input QC, before examining model metrics.',
 'turn':'region within0.75m of 99th-percentile far-end progress along origin-to-farthest3percent median axis; includes approach and departure, not pure angular turn; 0.25s position smoothing for boundaries; no yaw inputs',
 'quality':'complete finite source, max source gap0.05s, excursion>=1.5m, return<=65percent, native root speed<=6m/s inside selected windows; no filtering on unused joint angles',
 'straight':'prior start if clean and long enough, otherwise earliest clean start in same10mWT record; smoothed displacement/path ratio>=0.85; duration3+D; first3s at that same start for short baseline',
 'observations':'3s clean straight plus D-second turn region versus3+D straight; model sees no full TUG elapsed time or absolute timestamps; full recordings locate windows',
 'views':views,'turn_feature_names':NAMES,'folds':'same old outer and inner person folds restricted to this cohort','RF':'same128trees and innerleaf2/5, all15 runs','timed_trial_limitation':'same person/visit clinical label; exact cone/line timed trial mapping unavailable'}
 save('geometry/protocol.json',protocol)
 manifests=pd.read_csv(PREV/'manifest.csv');olddata=dict(np.load(PREV/'input.npz'));receipts=json.loads((HERE/'acquisition.json').read_text());lookup={};audit=[]
 for rr in receipts:
  pid,v,s=identity(rr['name']);r={'file':rr['name'],'id':pid,'visit':v,'session':s,'sha256':rr['sha256'],'task':'cone' if 'tug_cone' in rr['name'] else 'line'}
  try:a=read(HERE/'raw'/r['file']);r.update(trajectory_turn(a));r['status']='passed';lookup.setdefault((pid,v),[]).append((r,a))
  except ValueError as e:r['status']=str(e)
  audit.append(r)
 save('geometry/turn-audit.json',audit)
 rows=[];zs=[];excluded=[]
 for oi,m in manifests.iterrows():
  a=read(SOURCE/'raw'/m.file);chosen=None;reasons=[]
  for r,b in sorted(lookup.get((m.id,int(m.visit)),[]),key=lambda pair:(pair[0]['session']!=m.session,pair[0]['task']!='cone',pair[0]['session'],pair[0]['file'])):
   duration=r['end']-r['start'];budget=3+duration
   try:start,speed,ratio=choose_walk(a,budget,float(m.start_time))
   except ValueError as e:reasons.append(str(e));continue
   base,_=walk_window(a,start,3);long,_=walk_window(a,start,budget);tf=features(b,r['start'],r['end'],r['axis']);orig=olddata['x'][oi,17];oldfeat=np.array([orig.mean(),orig.std(),abs(np.diff(orig)).mean()])
   zs.append({'OldWalk3Speed':oldfeat,'Walk3Speed':base,'WalkMatchedSpeed':long,'Walk3TurnTime':np.r_[base,duration],'Walk3TurnShape':np.r_[base,tf[1:]],'Walk3Turn':np.r_[base,tf]})
   chosen={'old_sample':int(oi),'id':m.id,'visit':int(m.visit),'target':float(m.target),'walk_file':m.file,'walk_sha256':sha(SOURCE/'raw'/m.file),'walk_start':start,'walk_end':start+budget,'old_walk_start':float(m.start_time),'start_changed':abs(start-m.start_time)>1e-5,'walk_max_speed':speed,'straight_ratio':ratio,'turn_file':r['file'],'turn_sha256':r['sha256'],'task':r['task'],'turn_start':r['start'],'turn_end':r['end'],'turn_mid':r['mid'],'axis':r['axis'],'turn_max_speed':r['max_speed'],'turn_observed_seconds':duration,'matched_input_seconds':budget};break
  if chosen is None:excluded.append({'id':m.id,'target':float(m.target),'reasons':sorted(set(reasons)) or ['no_valid_geometry_turn']})
  else:rows.append(chosen)
 mapping={r['old_sample']:i for i,r in enumerate(rows)};plan=[]
 for rep in json.loads((PREV/'splits.json').read_text()):
  ss=[]
  for f in rep['splits']:
   remap=lambda arr:[mapping[i] for i in arr if i in mapping]
   ss.append({'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]})
  plan.append({**rep,'splits':ss})
 np.savez_compressed(out/'input.npz',ids=np.array([r['id'] for r in rows]),y=np.array([r['target'] for r in rows]),old_sample=np.array(list(mapping)),**{v:np.stack([z[v] for z in zs]) for v in views})
 save('geometry/manifest.json',rows);save('geometry/splits.json',plan);save('geometry/excluded.json',excluded)
 result={'people':len(rows),'long_people':sum(r['target']>=20 for r in rows),'changed_walk_start':sum(r['start_changed'] for r in rows),'long':[r for r in rows if r['target']>=20],'excluded_long':[r for r in excluded if r['target']>=20],'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json','excluded.json','protocol.json']}}
 save('preparation-geometry.json',{'cohorts':{'geometry':result},'code_sha256':sha(HERE/'geometry.py')});print(json.dumps({k:v for k,v in result.items() if k!='long'}),flush=True)
if __name__=='__main__':main()
