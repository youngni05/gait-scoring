"""Join published clinical labels to independently recomputed stride features."""
from pathlib import Path
import sys,json,hashlib,math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0,'D:/Projects/D_P/outputs/gdi-ridge-experiment-20261016/deps')
import numpy as np,pandas as pd,openpyxl
S=Path(__file__).resolve().parent;B=Path('D:/gait-research-data/D_P/larger-older-sppb-20261103')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def num(x):
 try:r=float(x);return r if math.isfinite(r) else None
 except (ValueError,TypeError):return None
KEYS=['CycleDuration[s]','StrideLength[m]','Per_Swing[%CD]','Stride_Velocity[m/s]','TurningAngle[degrees]']
def sides(p,dataset):
 df=pd.read_csv(p);df.columns=df.columns.str.strip();out={}
 for side in (['R','L'] if dataset=='gait2care' else ['S']):
  prefix=side+'_' if side!='S' else '';a=df[[prefix+k for k in KEYS]].to_numpy(float)
  valid=np.isfinite(a).all(1)&(a[:,0]>0)&(a[:,1]>0)&(a[:,2]>0)&(a[:,2]<100)&(a[:,3]>0)&(abs(a[:,4])<20)
  out[side]={'array':a[valid],'mask':valid,'all':a,'index':df[prefix+'index'].to_numpy()}
 return out

def features(ss):
 arrays=[s['array'] for s in ss.values()];allx=np.concatenate(arrays);f={'speed':float(allx[:,3].mean())}
 for j,name in enumerate(['stride_time_CV','stride_length_CV','swing_percent_CV']):
  n=sum(len(a) for a in arrays);dof=sum(len(a)-1 for a in arrays)
  sse=sum(float(((a[:,j]-a[:,j].mean())**2).sum()) for a in arrays)
  f[name]=float(100*np.sqrt(sse/dof)/allx[:,j].mean())
 return f

def coordination(pid,ss):
 base=B/'gait2care/extracted';seg=pd.read_csv(base/f'_GAIT/SEGMENTATION/Week0/{pid}-GAIT_SEGMENTATION_Week0.csv');t={};events={};audit={}
 for side in ['R','L']:
  p=base/f'_IMU/Week0/{pid}-{side}F-Week0.csv'
  if not p.exists():return None,{'status':'missing_raw_clock'}
  raw=pd.read_csv(p,usecols=['timestamp [s]','timestamp_NoCal[s]']);cal=raw['timestamp [s]'].to_numpy();clock=raw['timestamp_NoCal[s]'].to_numpy()
  assert np.isfinite(clock).all() and (np.diff(clock)>0).all(),(pid,side,'clock')
  a=seg[[f'{side}_HS_ini[sample]',f'{side}_HS_fin[sample]']].to_numpy();ok=np.isfinite(a).all(1)&(a[:,0]>0)&(a[:,1]>0)&(seg[f'{side}_index'].to_numpy()>0)
  assert np.array_equal(seg[f'{side}_index'].to_numpy()[ok],ss[side]['index'][ok])
  ix=a[ok].astype(int);assert (ix==a[ok]).all() and ix.min()>=1 and ix.max()<=len(clock)
  duration=ss[side]['all'][ok,0];delta=cal[ix[:,1]-1]-cal[ix[:,0]-1];error=abs(delta-duration)
  assert np.max(error)<1e-6,(pid,side,'source duration mismatch',np.max(error))
  t[side]=clock;events[side]={'intervals':clock[ix-1],'good':ss[side]['mask'][ok],'allstrikes':np.unique(clock[ix.ravel()-1])}
  audit[side]={'sample_index_base':1,'empty_padding_rows':int((~ok).sum()),'max_source_duration_error':float(error.max()),'clock_start':float(clock[0]),'clock_end':float(clock[-1]),'n_samples':len(clock),'max_calibration_drift_seconds':float(np.max(abs(cal-clock)))}
 phases={}
 for side,opp in [('R','L'),('L','R')]:
  e=events[side];other=events[opp]['allstrikes'];goodstarts=events[opp]['intervals'][events[opp]['good'],0];p=[]
  for (a,b),valid in zip(e['intervals'],e['good']):
   if not valid:continue
   inside=other[(other>a)&(other<b)]
   if len(inside)==1 and np.any(goodstarts==inside[0]):p.append(float(360*(inside[0]-a)/(b-a)))
  phases[side]=p;audit[side]['n_phases']=len(p)
 if min(map(len,phases.values()))<15:return None,{'status':'insufficient_phases','details':audit}
 vals=[100*np.std(x,ddof=1)/np.mean(x)+100*np.mean(abs(np.array(x)-180))/180 for x in phases.values()]
 audit.update(status='computed_from_original_common_clock',PCI_sides=[float(v) for v in vals],phases=phases)
 return float(np.mean(vals)),audit

def clinical():
 out={};w=openpyxl.load_workbook(B/'Gait2Care_DDBB.xlsx',read_only=True,data_only=True)
 for r in list(w['Week0'].values)[4:]:
  if not r[0]:continue
  out[r[0]]=dict(id=r[0],dataset='gait2care',age=num(r[3]),sex=r[4],weight=num(r[5]),height=num(r[6]),community=r[9]=='HOME',sppb=num(r[21]),balance=num(r[17]),gait_subscore=num(r[18]),chair=num(r[20]),tug=num(r[22]),published_speed=num(r[34]),published_duration=num(r[24]),published_analyzed=num(r[26]))
 w=openpyxl.load_workbook(B/'GSTRIDE_DDBB.xlsx',read_only=True,data_only=True)
 for r in list(w.active.values)[3:]:
  if not r[1]:continue
  age=str(r[9]);lo=num(age.split('-')[0]);out['gstride-'+r[1]]=dict(id='gstride-'+r[1],source_id=r[1],dataset='gstride',age=lo,age_band=age,sex=r[10],weight=num(r[12]),height=num(r[13]),community=str(r[11]).upper()=='NO',sppb=num(r[28]),balance=num(r[25]),gait_subscore=num(r[26]),chair=num(r[27]),tug=num(r[29]),published_speed=num(r[41]),published_duration=num(r[31]),published_analyzed=num(r[33]))
 return out

def main():
 full='--full' in sys.argv;people=[];excluded=[];ca={};c=clinical()
 for pid,r in c.items():
  ds=r['dataset'];reason=[]
  if r['age'] is None or r['age']<65:reason.append('age')
  y=r['sppb'];subs=[r[k] for k in ['balance','gait_subscore','chair']]
  if y is None or not 0<=y<=12 or int(y)!=y:reason.append('missing_invalid_sppb')
  if y is not None and all(v is not None for v in subs) and abs(sum(subs)-y)>1e-8:reason.append('published_total_differs_from_subscore_sum')
  p=B/ds/'extracted'/('_GAIT/PARAMETERS/Week0/'+pid+'-GAIT_PARAMETERS_Week0.csv' if ds=='gait2care' else '_GAIT/_PARAMETERS/'+r['source_id']+'_GAIT_PARAMETERS.csv')
  if not p.exists():reason.append('no_stride_file')
  if reason:excluded.append({'id':pid,'reason':reason});continue
  ss=sides(p,ds);counts={k:len(x['array']) for k,x in ss.items()}
  if min(counts.values())<15:excluded.append({'id':pid,'reason':['insufficient_straight_strides'],'counts':counts});continue
  r.update(features=features(ss),stride_counts=counts,source_file=str(p));r['features']['PCI']=None
  if full and ds=='gait2care':r['features']['PCI'],ca[pid]=coordination(pid,ss)
  people.append(r)
 duplicate=[]
 for a in [p for p in people if p['dataset']=='gait2care']:
  for b in [p for p in people if p['dataset']=='gstride']:
   if all(a[k]==b[k] and a[k] is not None for k in ['sex','height','weight']):duplicate.append([a['id'],b['id']])
 audit=dict(protocol_sha256=sha(S/'protocol.json'),full_clocks=full,registered={ds:sum(p['dataset']==ds for p in c.values()) for ds in ['gait2care','gstride']},included={ds:sum(p['dataset']==ds for p in people) for ds in ['gait2care','gstride']},excluded=excluded,exact_demographic_fingerprint_pairs=duplicate,cross_cohort_independence='Unresolved: pseudonyms not linked across cohorts; identical sex/height/weight is only a screen, not identity proof. Cross-dataset transfer will be provisional.',source_md5='Verified on downloaded workbooks; ZIP members CRC verified; full ZIP MD5 not checked because partial download.')
 save(S/'people.json',people);save(S/'input-audit.json',audit)
 if full:save(B/'coordination-audit.json',ca)
 print(json.dumps(audit,ensure_ascii=False,indent=2));print('PCI count',sum(p['features']['PCI'] is not None for p in people),flush=True)
if __name__=='__main__':main()
