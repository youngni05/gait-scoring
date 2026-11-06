"""Project actual C3D point trajectories; never disguise markers as BODY25."""
from pathlib import Path
import sys,json,hashlib,math,warnings
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'20261103-reference188-pilot'
BIG=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot')
sys.path[:0]=[str(BIG/'deps'),str(BASE/'deps'),str(Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))]
import numpy as np
import ezc3d,cv2
from PIL import Image,ImageDraw,ImageFont
PHYSICAL='LFHD RFHD LBHD RBHD C7 T10 CLAV STRN LSHO LELB LWRA LWRB LFIN RSHO RELB RWRA RWRB RFIN LASI RASI LTHI LKNE LTIB LANK LHEE LTOE RTHI RKNE RTIB RANK RHEE RTOE'.split()
VIRTUAL='LFEO LFEP LTIO LTIP RFEO RFEP RTIO RTIP'.split()
NAMES=PHYSICAL+VIRTUAL
CONDITIONS=['side_fixed','side_pan','oblique_fixed']
P=json.loads((HERE/'protocol.json').read_text());FOCAL=320/np.tan(np.deg2rad(P['camera']['horizontal_fov_deg']/2))
K=np.array([[FOCAL,0,320],[0,FOCAL,240],[0,0,1.]])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
def read_source(path):
 c=ezc3d.c3d(str(path));labels=[v.split(':')[-1].strip() for v in c['parameters']['POINT']['LABELS']['value']]
 assert len(set(labels))==len(labels),'Ambiguous multiple subjects'
 idx=[labels.index(n) for n in NAMES]
 xyz=c['data']['points'][:3,idx,:].transpose(2,1,0)/1000
 valid=c['data']['meta_points']['residuals'][0,idx,:].T>=0
 xyz[~valid]=np.nan
 assert c['parameters']['POINT']['UNITS']['value']==['mm']
 return c,xyz,labels,idx

def resample(x,rate):
 t=np.arange(math.floor((len(x)-1)/rate*30+1e-9)+1)/30
 at=t*rate;lo=np.floor(at+1e-10).astype(int);hi=np.minimum(lo+1,len(x)-1);w=(at-lo)[:,None,None]
 y=x[lo]*(1-w)+x[hi]*w
 exact=np.isclose(w[:,0,0],0,atol=1e-9);y[exact]=x[lo[exact]]
 return t,y

def align(x):
 center=(x[:,NAMES.index('LASI')]+x[:,NAMES.index('RASI')])/2
 assert np.isfinite(center[[0,-1]]).all()
 direction=center[-1]-center[0];direction[2]=0
 assert np.linalg.norm(direction)>.5,'Insufficient displacement to establish walkway direction'
 forward=direction/np.linalg.norm(direction);left=np.cross([0,0,1.],forward)
 basis=np.array([forward,left,[0,0,1.]])
 origin=(center[0]+center[-1])/2;origin[2]=0
 local=(x-origin)@basis.T
 lateral=local[:,NAMES.index('LASI'),1]-local[:,NAMES.index('RASI'),1]
 assert np.nanmedian(lateral)>0,'Walking and anatomical forward directions disagree'
 return local,basis,origin

def project(x,kind):
 pos=np.array([0.,-3.5,1.]);target=np.zeros((len(x),3));target[:,2]=1
 if kind=='side_pan':target[:,:2]=(x[:,NAMES.index('LASI'),:2]+x[:,NAMES.index('RASI'),:2])/2
 if kind=='oblique_fixed':pos[0]=-3.5*np.tan(np.deg2rad(15))
 fwd=target-pos;fwd/=np.linalg.norm(fwd,axis=1)[:,None]
 right=np.cross(fwd,np.array([0,0,1.]));right/=np.linalg.norm(right,axis=1)[:,None]
 down=np.cross(fwd,right);rot=np.stack([right,down,fwd],axis=1)
 camera=np.einsum('fij,fnj->fni',rot,x-pos)
 uv=camera[:,:,:2]/camera[:,:,2,None]*FOCAL+np.array([320,240])
 uv[camera[:,:,2]<=0]=np.nan
 return uv,rot,pos,camera[:,:,2]

def source_gdi(c):
 a=c['parameters'].get('ANALYSIS',{});out=[]
 for i,n in enumerate(a.get('NAMES',{}).get('value',[])):
  if n.upper()=='GDI':out.append(dict(index=i,name=n,context=a['CONTEXTS']['value'][i],value=float(a['VALUES']['value'][i]),units=a['UNITS']['value'][i]))
 proc={k:np.asarray(v['value']).tolist() for k,v in c['parameters'].get('PROCESSING',{}).items() if 'GDI' in k.upper()}
 return dict(analysis=out,processing=proc,usable_as_ground_truth=False,reason='Reference implementation, trial linkage and conflicting-field provenance unresolved')

def render(uv,title,t):
 im=Image.new('RGB',(640,480),'#f7f9fc');d=ImageDraw.Draw(im)
 d.rectangle((1,1,638,478),outline='#b8c3d0',width=2)
 d.text((14,10),title,fill='#162e49',font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',22))
 d.text((14,40),'%.2f s | surface markers, not BODY25'%t,fill='#4b5d70',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16))
 for side,color in [('L','#287ec2'),('R','#e58731')]:
  for chain in [[side+'ASI',side+'KNE',side+'ANK',side+'TOE',side+'HEE',side+'ANK'],[side+'SHO',side+'ELB',side+'WRA',side+'FIN'],[side+'SHO',side+'ASI']]:
   pts=[uv[NAMES.index(n)] for n in chain]
   for a,b in zip(pts[:-1],pts[1:]):
    if np.isfinite([a,b]).all():d.line([tuple(a),tuple(b)],fill=color,width=3)
  for n in PHYSICAL:
   if n.startswith(side):
    a=uv[NAMES.index(n)]
    if np.isfinite(a).all():d.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill=color)
 for a,b in [('LASI','RASI'),('LSHO','RSHO'),('LFHD','RFHD'),('LBHD','RBHD'),('LFHD','LBHD'),('RFHD','RBHD')]:
  x,y=uv[NAMES.index(a)],uv[NAMES.index(b)]
  if np.isfinite([x,y]).all():d.line([tuple(x),tuple(y)],fill='#61718a',width=2)
 d.text((14,452),'Blue: left markers     Orange: right markers',fill='#4b5d70',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16))
 return im

def main():
 manifest=json.loads((HERE/'download-manifest.json').read_text());result=[];arrays={};contact=[]
 health=json.loads((BASE/'person-linkage.json').read_text())['healthy']['rows'];stroke=json.loads((BASE/'speed-control/participants.json').read_text())
 for record in manifest:
  person=record['person'];path=Path(record['path']);assert sha(path)==record['sha256']
  c,source,labels,indices=read_source(path);rate=float(c['header']['points']['frame_rate']);assert rate==100
  t,sampled=resample(source,rate);local,basis,origin=align(sampled)
  arrays[person+'_source_m']=source;arrays[person+'_sampled_m']=sampled;arrays[person+'_local_m']=local;arrays[person+'_time_s']=t;arrays[person+'_basis']=basis;arrays[person+'_origin']=origin
  if person.startswith('SUBJ'):
   row=next(r for r in health if r['supplement_candidate_ids']==['SUBJ'+str(int(person[4:]))]);age=row['values']['Age'];group='healthy'
  else:age=next(r['age'] for r in stroke if r['id']==person);group='stroke'
  assert age>=65
  out=dict(person=person,age=age,group=group,source_frames=len(source),source_rate=rate,source_first_frame_zero_based=c['header']['points']['first_frame'],source_last_frame_zero_based=c['header']['points']['last_frame'],source_elapsed_s=(len(source)-1)/rate,target_frames=len(t),target_last_s=float(t[-1]),source_valid_fraction=float(np.isfinite(source).all(2).mean()),duration_pass=len(t)>=124,strict_gdi_input_pass=False,inference_count=0,reasons=['anatomical_landmark_mismatch']+(['fewer_than_124_frames'] if len(t)<124 else []),source_gdi=source_gdi(c),cameras={})
  for kind in CONDITIONS:
   uv,rot,pos,depth=project(local,kind)
   arrays[person+'_'+kind+'_uv']=uv;arrays[person+'_'+kind+'_R']=rot;arrays[person+'_'+kind+'_C']=pos;arrays[person+'_'+kind+'_depth']=depth
   p=uv[:,:len(PHYSICAL)];valid=np.isfinite(p).all(2);inside=valid&(p[:,:,0]>=0)&(p[:,:,0]<640)&(p[:,:,1]>=0)&(p[:,:,1]<480)
   out['cameras'][kind]=dict(valid_point_fraction=float(valid.mean()),in_frame_given_valid=float(inside.sum()/valid.sum()),all_physical_markers_in_frame_fraction=float(inside.all(1).mean()),yaw_deg_min=np.rad2deg(np.arctan2(rot[:,2,0],rot[:,2,1])).min().item())
   out['cameras'][kind]['yaw_deg_max']=float(np.rad2deg(np.arctan2(rot[:,2,0],rot[:,2,1])).max())
  for kind in CONDITIONS[1:]:
   diff=arrays[person+'_'+kind+'_uv']-arrays[person+'_side_fixed_uv'];out['cameras'][kind]['paired_pixel_displacement_rms']=float(np.sqrt(np.nanmean(np.sum(diff[:,:len(PHYSICAL)]**2,axis=2))))
  result.append(out)
  # One real instant, three views; animate at native time using every third 30fps sample.
  mid=len(t)//2;strip=Image.new('RGB',(1920,480))
  for j,kind in enumerate(CONDITIONS):strip.paste(render(arrays[person+'_'+kind+'_uv'][mid],person+' / '+kind,t[mid]),(640*j,0))
  contact.append(strip.resize((1440,360),Image.Resampling.LANCZOS))
  frames=[]
  for fi in range(0,len(t),3):
   frame=Image.new('RGB',(1920,480))
   for j,kind in enumerate(CONDITIONS):frame.paste(render(arrays[person+'_'+kind+'_uv'][fi],person+' / '+kind,t[fi]),(640*j,0))
   frames.append(frame.resize((1200,300),Image.Resampling.LANCZOS))
  frames[0].save(BIG/(person+'-cameras.gif'),save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
  print(person,len(t),'frames; strict GDI input blocked',flush=True)
 np.savez_compressed(BIG/'projections.npz',**arrays,names=np.array(NAMES),K=K)
 sheet=Image.new('RGB',(1440,360*len(contact)))
 for i,im in enumerate(contact):sheet.paste(im,(0,i*360))
 sheet.save(HERE/'projection-contact-sheet.png')
 dump('results.json',dict(protocol_sha256=sha(HERE/'protocol.json'),data_sha256=sha(BIG/'projections.npz'),source_names=NAMES,trials=result,gdi_inference_runs=0,training_runs=0,eligible_duration_trials=sum(r['duration_pass'] for r in result),strict_eligible_trials=0))
if __name__=='__main__':main()
