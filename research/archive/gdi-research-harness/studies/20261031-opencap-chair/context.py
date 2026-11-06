from pathlib import Path
import sys,json,hashlib
STUDY=Path(__file__).resolve().parent
PRIOR=STUDY.parent/'20261031-opencap-turn'
sys.path.insert(0,str(PRIOR))
from common import np,pd,read,sha,SOURCE,PREV,ROOT,SEEDS
import geometry as geom
from scipy.ndimage import uniform_filter1d

def save(name,value):
 p=STUDY/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def chair(a,m):
 t=a['time'];dt=np.median(np.diff(t));h=uniform_filter1d(a['pelvis_ty'],max(3,round(.25/dt)),mode='nearest')
 seated0=float(np.median(h[t<=t[0]+.5]));seated1=float(np.median(h[t>=t[-1]-.5]));standing=float(np.median(h[(t>=m['turn_start'])&(t<=m['turn_end'])]))
 dr=standing-seated0;ds=standing-seated1
 if not (.12<=dr<=.7 and .12<=ds<=.7):raise ValueError('seated_standing_height_contrast')
 r=(h-seated0)/dr;s=(h-seated1)/ds
 if max(np.median(r[t<=t[0]+.5]),np.median(s[t>=t[-1]-.5]))>.2:raise ValueError('seated_endpoints')
 upper=np.flatnonzero((r>=.8)&(t<m['turn_start']))
 if not len(upper):raise ValueError('rise_missing')
 re=int(upper[0]);lower=np.flatnonzero(r[:re]<=.2)
 if not len(lower):raise ValueError('rise_start_missing')
 rs=int(lower[-1])
 lower=np.flatnonzero((s<=.2)&(t>m['turn_end']))
 if not len(lower):raise ValueError('sit_missing')
 se=int(lower[0]);upper=np.flatnonzero((s[:se]>=.8)&(t[:se]>m['turn_end']))
 if not len(upper):raise ValueError('sit_start_missing')
 ss=int(upper[-1])
 windows={}
 for name,lo,hi in [('rise',rs,re),('sit',ss,se)]:
  # A quarter-second context at both ends, outward aligned to native 30 Hz grid.
  left=float(np.floor((t[lo]-.25)*30+1e-6)/30);right=float(np.ceil((t[hi]+.25)*30-1e-6)/30)
  if left<t[0] or right>t[-1] or not .2<=t[hi]-t[lo]<=25:raise ValueError(name+'_edges_or_duration')
  part=a[(t>=left-1e-7)&(t<=right+1e-7)];pos=np.column_stack([part[n] for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]);v=np.diff(pos,axis=0)/np.diff(part['time'])[:,None]
  if max(np.linalg.norm(v[:,[0,2]],axis=1))>6 or max(abs(v[:,1]))>3:raise ValueError(name+'_position_jump')
  windows[name]=[left,right]
 if not windows['rise'][1]<m['turn_start']<m['turn_end']<windows['sit'][0]:raise ValueError('overlapping_phases')
 return {**windows,'seated_initial':seated0,'seated_final':seated1,'standing':standing,'height_rise':dr,'height_sit':ds}

def central(a,m,seconds,bounds):
 # Keep the complete original turn; add context on both sides without chair phases.
 lo,hi=bounds;left=(m['turn_start']+m['turn_end']-seconds)/2
 left=np.floor(left*30+1e-6)/30;left=max(lo,min(left,hi-seconds));right=left+seconds
 if left>m['turn_start']+1e-6 or right<m['turn_end']-1e-6 or left<lo-1e-6 or right>hi+1e-6:raise ValueError('central_context_insufficient')
 if geom.root_qc(a,left,right)>6:raise ValueError('central_context_position_jump')
 return [float(left),float(right)]
