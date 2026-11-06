"""Local height transitions; input-only refinement after the initial drift audit."""
from context import *
from scipy.signal import find_peaks

def transition(a,m,kind):
 t=a['time'];dt=float(np.median(np.diff(t)));h=uniform_filter1d(a['pelvis_ty'],max(3,round(.25/dt)),mode='nearest');vel=np.gradient(h,t)
 pos=np.column_stack([a['pelvis_tx'],a['pelvis_tz']]);origin=np.median(pos[t<=t[0]+.5],axis=0);progress=(uniform_filter1d(pos,max(3,round(.25/dt)),axis=0,mode='nearest')-origin)@m['axis'];extent=float(np.quantile(progress,.99))
 sign=1 if kind=='rise' else -1;v=sign*vel
 eligible=(progress<.4*extent)&(t>t[0]+.5)&(t<t[-1]-.5)&((t<m['turn_start']) if kind=='rise' else (t>m['turn_end']))
 peaks,_=find_peaks(v,height=.12,distance=max(1,round(.5/dt)),prominence=.08)
 reasons=[]
 for peak in sorted([int(p) for p in peaks if eligible[p]],key=lambda p:-v[p]):
  before=np.flatnonzero((v[:peak]<=.05)&(t[:peak]>=t[peak]-6));after=np.flatnonzero((v[peak:]<=.05)&(t[peak:]<=t[peak]+6))
  if not len(before) or not len(after):continue
  lo=int(before[-1]);hi=int(peak+after[0])
  if sign*(h[hi]-h[lo])<.08:continue
  left=float(np.floor((t[lo]-.25)*30+1e-6)/30);right=float(np.ceil((t[hi]+.25)*30-1e-6)/30)
  if left<t[0] or right>t[-1] or not .2<=t[hi]-t[lo]<=12:continue
  if (kind=='rise' and right>=m['turn_start']) or (kind=='sit' and left<=m['turn_end']):continue
  if kind=='sit':
   # Incomplete return recordings cannot supply an observed seated endpoint.
   seated0=np.median(h[t<=t[0]+.5]);seated1=np.median(h[t>=t[-1]-.5])
   if abs(seated1-seated0)>.12 or np.median(h[(t>=right-.15)&(t<=right)])>seated0+.12:continue
  part=a[(t>=left-1e-7)&(t<=right+1e-7)];vertical=abs(np.diff(part['pelvis_ty'])/np.diff(part['time']))
  return {'window':[left,right],'peak':float(t[peak]),'height_change':float(sign*(h[hi]-h[lo])),'max_native_vertical_speed':float(max(vertical)),'strict_quality':bool(max(vertical)<=3)}
 raise ValueError(kind+'_not_identifiable')

def features(a,window):
 left,right=window;t=a['time'];mask=(t>=left-1e-7)&(t<=right+1e-7);tt=t[mask];grid=tt[0]+np.arange(round((right-left)*30)+1)/30
 assert grid[-1]<=tt[-1]+1e-6
 h=np.interp(grid,tt,a['pelvis_ty'][mask]);v=np.gradient(h,1/30)
 # Vertical-only representation: unused horizontal/angular tracking errors are not inputs.
 return np.array([right-left,h[-1]-h[0],np.ptp(h),v.mean(),v.std(),np.quantile(abs(v),.95),abs(np.diff(v)).mean(),np.mean(abs(v)<.05)])

def audit():
 rows=json.loads((PRIOR/'geometry/manifest.json').read_text());out=[]
 for m in rows:
  a=read(PRIOR/'raw'/m['turn_file']);r={'id':m['id'],'target':m['target']}
  for k in ['rise','sit']:
   try:r[k]=transition(a,m,k)
   except ValueError as e:r[k]={'error':str(e)}
  out.append(r)
 save('transition-audit.json',out)
 print(json.dumps({'rise':sum('window' in r['rise'] for r in out),'sit':sum('window' in r['sit'] for r in out),'both':sum(all('window' in r[k] for k in ['rise','sit']) for r in out),'long':[r for r in out if r['target']>=20]}),flush=True)
if __name__=='__main__':audit()
