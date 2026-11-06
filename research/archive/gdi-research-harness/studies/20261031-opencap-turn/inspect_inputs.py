from common import *
from scipy.ndimage import uniform_filter1d
def detect(a):
 t=a['time'];dt=float(np.median(np.diff(t)))
 if len(t)<60 or not np.all(np.diff(t)>0) or max(np.diff(t))>.05:raise ValueError('time')
 cols=['pelvis_tx','pelvis_ty','pelvis_tz','pelvis_rotation','pelvis_list','lumbar_bending']
 if not all(np.isfinite(a[n]).all() for n in cols):raise ValueError('nonfinite')
 yaw=np.rad2deg(np.unwrap(np.deg2rad(a['pelvis_rotation'])))
 width=max(3,round(.25/dt));angle=uniform_filter1d(yaw,width,mode='nearest');angle-=np.median(angle[:max(1,round(.5/dt))])
 pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);smooth=uniform_filter1d(pos,width,axis=0,mode='nearest')
 origin=np.median(smooth[:max(1,round(.5/dt))],axis=0);distance=np.linalg.norm(smooth-origin,axis=1)
 extent=float(np.quantile(distance,.99))
 if extent<1.5:raise ValueError('short_excursion')
 candidates=np.flatnonzero((abs(angle)>=90)&(abs(angle)<140)&(distance>.6*extent))
 if not len(candidates):raise ValueError('no_midturn')
 mid=int(candidates[0]);sign=float(np.sign(angle[mid]));progress=sign*angle
 before=np.flatnonzero(progress[:mid]<=20);after=np.flatnonzero(progress[mid:]>=160)
 if not len(before) or not len(after):raise ValueError('incomplete_turn')
 start=int(before[-1]);end=int(mid+after[0]);left=t[start]-.25;right=t[end]+.25
 if left<t[0] or right>t[-1] or t[end]-t[start]<.3 or t[end]-t[start]>27:raise ValueError('turn_duration_or_edges')
 if min(distance[start],distance[end])<.45*extent:raise ValueError('turn_far_from_apex')
 if np.min(distance[end:])>.65*extent:raise ValueError('no_return')
 return {'start':float(left),'end':float(right),'mid':float(t[mid]),'turn_seconds':float(right-left),'extent':extent,'sign':sign,'raw_yaw_jump':float(np.max(abs(np.diff(yaw)))),'max_speed':float(np.max(np.linalg.norm(np.diff(pos,axis=0),axis=1)/np.diff(t)))}
if __name__=='__main__':
 rows=[]
 for p in sorted((HERE/'raw').rglob('*.mot')):
  try:r=detect(read(p));r['status']='detected'
  except ValueError as e:r={'status':str(e)}
  rows.append({'file':p.relative_to(HERE/'raw').as_posix(),**r})
 save('input-inspection.json',rows)
 print(pd.DataFrame(rows).groupby('status').size().to_string())
 print(pd.DataFrame(rows).select_dtypes('number').describe().round(3).to_string())
