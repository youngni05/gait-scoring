"""Read released events and joint centres; prepare labelled simulations, never RGB claims."""
from pathlib import Path
import sys,json,csv,hashlib,importlib.util,warnings
ROOT=Path('D:/Projects/D_P'); HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/older44-coordination-sppb-20261103')
RAW=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261030-sppb-timeseries/raw')
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import openpyxl
from PIL import Image,ImageDraw
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pre=module('pinned_pre',ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle/preprocess.py')
NS=pre.source_functions()
NAMES=['LHJC','RHJC','LKJC','RKJC','LAJC','RAJC','LTOE','RTOE','LHEE','RHEE','LASI','RASI','LPSI','RPSI','LKNE','RKNE','LANK','RANK']
BODY=[12,9,13,10,14,11,19,22]
VARIANTS=['primary','toe60','distance4']

def section(rows,name):
 i=rows.index([name]);out=[]
 for row in rows[i+5:]:
  if not row or not row[0].strip():break
  out.append(row)
 return rows[i+2],rows[i+3],rows[i+4],out

def load_trial(path):
 rows=list(csv.reader(path.open(encoding='utf-8-sig'),delimiter='\t'))
 sections={k:section(rows,k) for k in ['Trajectories','Model Outputs']}
 blocks={};dup=[];frames=None
 for sec,(names,axes,units,data) in sections.items():
  f=np.array([int(v[0]) for v in data]);assert (np.diff(f)==1).all()
  if frames is not None:np.testing.assert_array_equal(frames,f)
  frames=f
  for name in NAMES:
   wanted=name+'__py' if sec=='Model Outputs' else name
   js=[j for j,n in enumerate(names) if n.split(':')[-1]==wanted]
   if not js:continue
   arrays=[]
   for j in js:
    assert axes[j:j+3]==['X','Y','Z'] and units[j:j+3]==['mm']*3
    a=np.array([[float(r[k]) if k<len(r) and r[k].strip() else np.nan for k in range(j,j+3)] for r in data])/1000
    arrays.append(a)
   for a in arrays[1:]:np.testing.assert_allclose(a,arrays[0],atol=0,rtol=0,equal_nan=True)
   blocks[name]=arrays[0]
   if len(js)>1:dup.append(name)
 assert all(n in blocks for n in NAMES)
 q=np.stack([blocks[n] for n in NAMES],axis=1)
 i=rows.index(['Events']);events={s:{k:[] for k in ['Foot Strike','Foot Off']} for s in ['Left','Right']}
 for r in rows[i+3:]:
  if not r or not r[0]:break
  if r[1] in events and r[2] in events[r[1]]:events[r[1]][r[2]].append(float(r[3]))
 for v in events.values():
  for a in v.values():assert a==sorted(set(a))
 i=rows.index(['Gait Cycle Parameters']);params={}
 for r in rows[i+3:]:
  if not r or not r[0]:break
  params[r[1]+'_'+r[2]]=float(r[3])
 return frames,q,events,params,dup

def align(q):
 pelvis=q[:,10:14].mean(1);good=np.flatnonzero(np.isfinite(pelvis).all(1));assert len(good)>1
 direction=pelvis[good[-1]]-pelvis[good[0]];direction[2]=0
 direction/=np.linalg.norm(direction);left=np.cross([0,0,1.],direction)
 origin=(pelvis[good[0]]+pelvis[good[-1]])/2;origin[2]=0
 local=(q-origin)@np.stack([direction,left,[0,0,1.]]).T
 assert np.nanmedian(local[:,0,1]-local[:,1,1])>0
 return local,direction,pelvis

def event_features(frames,q,events):
 local,direction,pelvis=align(q);times=frames/100
 def at(t,j):
  v=t*100-frames[0];lo=int(np.floor(v+1e-8));hi=int(np.ceil(v-1e-8))
  if lo<0 or hi>=len(q):return np.full(3,np.nan)
  if hi==lo:return q[lo,j]
  return q[lo,j]*(hi-v)+q[hi,j]*(v-lo)
 values={s:{k:[] for k in ['phase','stride','length','swing']} for s in ['Left','Right']};cycles=[];reject=[]
 for side,other,j,k in [('Left','Right',8,9),('Right','Left',9,8)]:
  strikes=events[side]['Foot Strike'];opp=events[other]['Foot Strike'];offs=events[side]['Foot Off']
  for t0,t1 in zip(strikes[:-1],strikes[1:]):
   between=[v for v in opp if t0<v<t1];off=[v for v in offs if t0<v<t1]
   if len(between)!=1 or len(off)!=1 or not(times[0]<=t0<t1<=times[-1]):
    reject.append(dict(side=side,t0=t0,t1=t1,reason='nonalternating_or_out_of_range'));continue
   length=float((at(t0,j)-at(t0,k))@direction)
   if not np.isfinite(length) or length<=0:
    reject.append(dict(side=side,t0=t0,t1=t1,reason='invalid_heel_separation'));continue
   phase=360*(between[0]-t0)/(t1-t0);swing=100*(t1-off[0])/(t1-t0)
   vals=[phase,t1-t0,length,swing]
   for key,value in zip(['phase','stride','length','swing'],vals):values[side][key].append(value)
   cycles.append(dict(side=side,t0=t0,t1=t1,opposite=between[0],off=off[0],phase=phase,stride=t1-t0,length=length,swing=swing))
 good=np.flatnonzero(np.isfinite(pelvis).all(1));speed=float((pelvis[good[-1]]-pelvis[good[0]])@direction/((frames[good[-1]]-frames[good[0]])/100))
 return values,cycles,reject,speed,local

def project(local,distance=3.5,toe=0):
 x=local.copy()
 for j,k in [(6,8),(7,9)]:
  vec=x[:,j]-x[:,k];vec/=np.linalg.norm(vec,axis=1)[:,None];x[:,j]+=toe*vec
 pos=np.array([0.,-distance,1.]);target=np.zeros((len(x),3));target[:,:2]=x[:,:2,:2].mean(1);target[:,2]=1
 forward=target-pos;forward/=np.linalg.norm(forward,axis=1)[:,None]
 right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right,axis=1)[:,None];down=np.cross(forward,right)
 cam=np.einsum('fij,fnj->fni',np.stack([right,down,forward],1),x-pos)
 uv=cam[:,:,:2]/cam[:,:,2,None]*(320/np.tan(np.pi/6))+[320,240]
 return uv,cam[:,:,2],np.rad2deg(np.arctan2(forward[:,0],forward[:,1]))

def render(uv,title):
 im=Image.new('RGB',(640,480),'white');d=ImageDraw.Draw(im);d.text((10,10),title,fill='black')
 for ids,color in [([0,2,4,6,8],'blue'),([1,3,5,7,9],'orange')]:
  for a,b in zip(ids[:-1],ids[1:]):
   if np.isfinite(uv[[a,b]]).all():d.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=3)
  for j in ids:
   if np.isfinite(uv[j]).all():
    x,y=uv[j];d.ellipse((x-4,y-4,x+4,y+4),fill=color)
 return im

def pooled_cv(groups):
 n=sum(len(v) for v in groups);dof=sum(max(0,len(v)-1) for v in groups)
 mean=sum(sum(v) for v in groups)/n
 return 100*np.sqrt(sum(float(np.sum((np.array(v)-np.mean(v))**2)) for v in groups if v)/dof)/mean

def main():
 manifest=list(csv.DictReader((ROOT/'gdi-research-harness/studies/20261030-older44-sppb-timeseries/manifest.csv').open()))
 records=[];windows=[];owner=[];variants=[];starts=[];previews=[];phases={};center_offsets=[]
 for r in manifest:
  path=RAW/r['file'];assert sha(path)==r['sha256'];frames,q,ev,params,dup=load_trial(path)
  values,cycles,reject,speed,local=event_features(frames,q,ev)
  info=dict(file=r['file'],id=r['id'],fold=int(r['fold']),source_sha256=r['sha256'],first_frame=int(frames[0]),last_frame=int(frames[-1]),duration=(len(frames)-1)/100,events=ev,values=values,cycles=cycles,rejected=reject,speed=speed,duplicate_centres=dup,projection={})
  # Published summary is selected-cycle, not all-cycle; inspect closest genuine cycle without changing events.
  info['summary_stride_check']={s:min(abs(c['stride']-params[s+'_Stride Time']) for c in cycles if c['side']==s) for s in ['Left','Right']}
  center_offsets.append({n:float(np.nanmedian(np.linalg.norm(q[:,a]-q[:,b],axis=1))*1000) for n,a,b in [('left_knee',2,14),('right_knee',3,15),('left_ankle',4,16),('right_ankle',5,17)]})
  t=np.arange(int(np.floor((len(q)-1)/100*29.97))+1)/29.97
  pos=t*100;lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,len(q)-1);w=(pos-lo)[:,None,None]
  sampled=local[lo]*(1-w)+local[hi]*w;sampled[np.isclose(pos-lo,0)]=local[lo[np.isclose(pos-lo,0)]]
  ix=len(records);common=None;prepared={}
  for variant in VARIANTS:
   uv,depth,yaw=project(sampled,distance=4 if variant=='distance4' else 3.5,toe=.06 if variant=='toe60' else 0)
   raw=np.full((len(t),25,3),np.nan)
   for j,k in enumerate(BODY):raw[:,k,:2]=uv[:,j]
   inside=np.isfinite(uv[:,:8]).all(2)&(uv[:,:8,0]>=0)&(uv[:,:8,0]<640)&(uv[:,:8,1]>=0)&(uv[:,:8,1]<480)&(depth[:,:8]>0)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning);processed=NS['process_video_and_add_cols'](raw.reshape(len(t),75),swap_orientation=False)
   feat=np.stack([processed[:,pre.LEFT],processed[:,pre.RIGHT]])
   valid=[s for s in range(0,500-124,31) if s+124<=len(t) and inside[s:s+124].all() and np.isfinite(feat[:,s:s+124]).all()]
   prepared[variant]=feat;common=set(valid) if common is None else common&set(valid)
   info['projection'][variant]=dict(frames=len(t),valid_starts=valid,yaw_min=float(np.nanmin(yaw)),yaw_max=float(np.nanmax(yaw)),in_frame_fraction=float(inside.all(1).mean()))
   if r['id'] in ['1','10','20','30','40'] and r['file'].endswith('_01.csv') and variant=='primary':
    strip=Image.new('RGB',(1920,480),'white')
    for col,fi in enumerate([0,len(t)//2,len(t)-1]):strip.paste(render(uv[fi],f'{r["file"]} / {t[fi]:.2f}s / yaw {yaw[fi]:.1f}deg'),(640*col,0))
    previews.append(strip.resize((1200,300)))
  info['common_starts']=sorted(common)
  for variant in VARIANTS:
   for s in sorted(common):
    windows.append(prepared[variant][:,s:s+124].astype(np.float32));owner.append(ix);variants.append(variant);starts.append(s)
  records.append(info)
  if len(records)%40==0:print('audited',len(records),flush=True)
 wb=openpyxl.load_workbook(ROOT/'outputs/gait-public-access-20261015/OlderAdults44/01_demography_sppb_mmse.xlsx',data_only=True)
 labels={str(int(str(r[0]).split()[-1])):dict(age=r[2],sppb=r[7]) for r in list(wb.active.values)[1:] if str(r[0]).startswith('Subject ')};wb.close();assert len(labels)==44
 people=[]
 for pid in sorted(labels,key=int):
  rr=[r for r in records if r['id']==pid];assert len(rr)==5
  counts={s:sum(len(r['values'][s]['phase']) for r in rr) for s in ['Left','Right']}
  pcis={}
  for s in ['Left','Right']:
   v=np.array([v for r in rr for v in r['values'][s]['phase']]);pcis[s]=float(100*v.std(ddof=1)/v.mean()+100*np.abs(v-180).mean()/180)
  features=dict(speed=float(np.mean([r['speed'] for r in rr])),PCI=float(np.mean(list(pcis.values()))))
  for key,name in [('stride','stride_time_CV'),('length','step_length_CV'),('swing','swing_percent_CV')]:features[name]=float(pooled_cv([r['values'][s][key] for r in rr for s in ['Left','Right']]))
  people.append(dict(id=pid,fold=rr[0]['fold'],**labels[pid],features=features,phase_counts=counts,PCI_by_side=pcis,event_eligible=min(counts.values())>=15,gdi_trials=sum(bool(r['common_starts']) for r in rr)))
 np.savez_compressed(BIG/'features.npz',x=np.array(windows),owner=np.array(owner),variant=np.array(variants),start=np.array(starts))
 save(HERE/'trial-audit.json',records);save(HERE/'people.json',people)
 summary=dict(protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'features.npz'),raw_files=220,people=44,event_eligible=sum(p['event_eligible'] for p in people),gdi_eligible=sum(p['gdi_trials']>0 for p in people),age65plus=sum(p['age']>=65 for p in people),phase_counts_range=[min(min(p['phase_counts'].values()) for p in people),max(max(p['phase_counts'].values()) for p in people)],summary_stride_max_discrepancy=max(max(r['summary_stride_check'].values()) for r in records),center_offsets_mm={n:[min(v[n] for v in center_offsets),max(v[n] for v in center_offsets)] for n in center_offsets[0]},windows=len(windows),geometry='Perspective fixed-position yaw pan; right camera, image forward leftward; height/FOV assumed',anatomical_equivalence=False,RGB_validation=False,inference_performed=False)
 save(HERE/'input-audit.json',summary)
 sheet=Image.new('RGB',(1200,300*len(previews)),'white')
 for i,im in enumerate(previews):sheet.paste(im,(0,300*i))
 sheet.save(HERE/'camera-contact-sheet.png');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
