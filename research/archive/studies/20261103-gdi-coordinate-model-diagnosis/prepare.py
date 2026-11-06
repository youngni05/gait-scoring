"""Reconstruct original selected coordinates and apply prespecified input interventions."""
from pathlib import Path
import json,hashlib,importlib.util,sys,struct,zlib,time,warnings
H=Path(__file__).resolve().parent;ROOT=Path('D:/Projects/D_P');S=ROOT/'gdi-research-harness/studies/20261030-gpu-motionbert-study';C=S/'cnn-multiwindow695'
B=Path('D:/gait-research-data/D_P/gdi-coordinate-model-diagnosis-20261103');CACHE=B.parent/'gdi-research-harness/studies/20261030-gpu-motionbert-study'
sys.path.insert(0,str(ROOT/'studies/20261103-reference188-pilot/deps'))
import numpy as np
from scipy.ndimage import gaussian_filter1d

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
pre=module('original_preparation',C/'prepare_inputs.py');ns=pre.source_functions();ORIG=ns['preprocess_frames']
LEFT=[28,29,26,27,24,25,38,39,50,52,54,56];RIGHT=[22,23,20,21,18,19,44,45,51,53,55,57];JOINTS=[14,11,13,10,12,9,19,22]
# Roundoff guard preserves the mathematical definition of acos.
def angle(a,b,c,p):
 u=p[:,[2*a,2*a+1]]-p[:,[2*b,2*b+1]];v=p[:,[2*c,2*c+1]]-p[:,[2*b,2*b+1]]
 ratio=np.sum(u*v,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1));return np.arccos(np.clip(ratio,-1,1))
ns['get_angle']=angle

def observed(raw):
 p=raw.reshape(-1,25,3);valid=np.isfinite(p[:,:,:2]).all(2)&(p[:,:,:2]!=0).all(2)
 length=[]
 for hip,knee in [(12,13),(9,10)]:
  good=valid[:,hip]&valid[:,knee];length.extend(np.linalg.norm(p[good,hip,:2]-p[good,knee,:2],axis=1))
 length=np.array(length);scale=float(np.median(length));assert np.isfinite(scale) and scale>0
 return p,valid,scale,length

def spike_mask(p,valid,scale):
 mask=np.zeros(valid.shape,bool);mid=(p[:-2,:,:2]+p[2:,:,:2])/2
 condition=valid[:-2]&valid[1:-1]&valid[2:]&(np.linalg.norm(p[1:-1,:,:2]-mid,axis=2)>.25*scale)&(np.linalg.norm(p[2:,:,:2]-p[:-2,:,:2],axis=2)<.1*scale)
 mask[1:-1,JOINTS]=condition[:,JOINTS];return mask

def transform(raw,name,video):
 p,valid,scale,length=observed(raw);changed=raw.copy().reshape(-1,25,3);mask=spike_mask(p,valid,scale);dose=0.
 if name=='spike_repair':
  midpoint=(p[:-2,:,:2]+p[2:,:,:2])/2;changed[1:-1,:,:2][mask[1:-1]]=midpoint[mask[1:-1]]
 elif name.startswith('jitter'):
  level=int(name[6:8]);seed=int(name[-1]);rng=np.random.default_rng(20261103+seed+int(video));noise=rng.normal(size=(len(p),len(JOINTS),2))*(level/100)*scale
  for j,k in enumerate(JOINTS):changed[valid[:,k],k,:2]+=noise[valid[:,k],j]
  diff=(changed[:,:,:2]-p[:,:,:2])[:,JOINTS]/scale;dose=float(np.sqrt(np.mean(diff[valid[:,JOINTS]]**2)))
 elif name=='knee_bias03':
  for k in [10,13]:changed[valid[:,k],k,0]+=.03*scale
 assert np.array_equal(np.isnan(changed),np.isnan(p));assert np.array_equal(changed[:,:,2],p[:,:,2],equal_nan=True)
 ns['filter_frames']=lambda frames:gaussian_filter1d(frames,2 if name=='smooth2' else 1,axis=0)
 ns['preprocess_frames']=ORIG
 if name=='stable_scale':
  def stable(a,swap_orientation=False):
   normalized=ORIG(a,swap_orientation=swap_orientation);q=ns['drop_confidence_cols'](a);q[q==0]=np.nan;q=ns['filter_frames'](ns['impute_frames'](q))
   scale=(np.linalg.norm(q[:,24:26]-q[:,26:28],axis=1)+np.linalg.norm(q[:,18:20]-q[:,20:22],axis=1))/2
   return normalized*scale[:,None]/np.median(scale)
  ns['preprocess_frames']=stable
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning);features=ns['process_video_and_add_cols'](changed.reshape(-1,75),swap_orientation=False)
 ns['preprocess_frames']=ORIG
 return features,dict(thigh_pixels=scale,spikes=int(mask.sum()),coordinate_noise_sd_relative=dose)

def summary(x):
 # window,leg,time,12 channels -> window,leg,96 descriptors
 return np.stack([x.mean(2),x.std(2),*np.quantile(x,[.05,.25,.5,.75,.95],axis=2),np.sqrt(np.mean(np.diff(x,axis=2)**2,axis=2))],axis=-1).reshape(len(x),2,96)

def quality(raw,starts):
 p,valid,scale,length=observed(raw);mask=spike_mask(p,valid,scale);out=[]
 for start in starts:
  q=p[start:start+124][:,JOINTS];v=valid[start:start+124][:,JOINTS];usable=v[:-2]&v[1:-1]&v[2:];a=np.linalg.norm(q[2:,:,:2]-2*q[1:-1,:,:2]+q[:-2,:,:2],axis=2)/scale
  window_lengths=[]
  for hip,knee in [(12,13),(9,10)]:
   good=valid[start:start+124,hip]&valid[start:start+124,knee];segment=p[start:start+124];window_lengths.extend(np.linalg.norm(segment[good,hip,:2]-segment[good,knee,:2],axis=1))
  window_lengths=np.array(window_lengths)
  out.append([float((~np.isfinite(q[:,:,:2])|(q[:,:,:2]==0)).mean()),float((np.nan_to_num(q[:,:,2],nan=0)<.3).mean()),float(mask[start:start+124][:,JOINTS].mean()),float(np.median(a[usable])) if usable.any() else 0.,float(window_lengths.std()/window_lengths.mean())])
 return np.mean(out,axis=0)

def main():
 started=time.monotonic();protocol=read(H/'protocol.json')
 for item in read(H/'input-manifest.json'):assert sha(item['path'])==item['sha256']
 d=np.load(C/'bundle/data.npz');src=read(C/'input_sources.json');allrows=src['records'];rows={allrows[int(ix)]['folder']:(i,allrows[int(ix)]) for i,ix in enumerate(d['original_row_index'])}
 archive=Path('D:/gait-research-data/D_P/outputs/gait-public-access-20261015/Stanford/video-gait-v1.zip');assert archive.stat().st_size==src['archive_stat']['size'];archive_hash=sha(archive)
 (B/'raw').mkdir(exist_ok=True);(B/'inputs').mkdir(exist_ok=True)
 arrays={k:np.lib.format.open_memmap(B/'inputs'/f'{k}.npy',mode='w+',dtype='float32',shape=d['x'].shape) for k in protocol['conditions']}
 summaries={k:np.empty((len(d['y']),2,96)) for k in ['baseline','smooth2','spike_repair','stable_scale']};qout=np.empty((len(d['y']),5));audits=[]
 def process(row,members,stream):
  i,meta=row;last=0
  while last+1 in members and last<1499:last+=1
  raw=np.full((last,75),np.nan);content_hash=hashlib.sha256()
  for frame in range(1,last+1):
   m=members[frame];stream.seek(m['offset']);head=stream.read(30);assert head[:4]==b'PK\x03\x04';nl,el=struct.unpack_from('<HH',head,26);assert stream.read(nl).decode()==m['name'];stream.seek(el,1);packed=stream.read(m['compressed']);content=zlib.decompress(packed,-15) if m['method']==8 else packed;assert len(content)==m['size'] and zlib.crc32(content)&0xffffffff==m['crc'];content_hash.update(content);candidates=[]
   for person in json.loads(content)['people']:
    a=np.array(person['pose_keypoints_2d'],float)
    if a.shape==(75,) and np.isfinite(a).all() and a[::3].max()<320:candidates.append(a)
   if len(candidates)==1:raw[frame-1]=candidates[0]
  observed_rows=np.flatnonzero(np.isfinite(raw).any(1));raw=raw[:observed_rows[-1]+1]
  old=read(C/'records'/f"{meta['index']:04d}.json");assert content_hash.hexdigest()==old['raw_content_sha256']
  wi=np.flatnonzero(d['window_record']==i);starts=d['window_start'][wi];assert starts.tolist()==old['eligible_sliding_window_starts']
  rawpath=B/'raw'/f'{i:04d}.npz';np.savez_compressed(rawpath,raw=raw)
  qout[i]=quality(raw,starts);details={}
  for name in arrays:
   p,detail=transform(raw,name,meta['video_id']);x=np.stack([np.stack([p[k:k+124,LEFT],p[k:k+124,RIGHT]]) for k in starts]).astype('float32');assert np.isfinite(x).all(),(i,name)
   if name=='baseline':
    difference=float(np.max(np.abs(x-d['x'][wi])));assert difference<1e-6,(i,difference);detail['baseline_max_difference']=difference
   arrays[name][wi]=x
   if name in summaries:summaries[name][i]=summary(x.astype('float64')).mean(0)
   details[name]=detail
  audits.append(dict(record=i,original_row=meta['index'],person=meta['person_id'],video=meta['video_id'],source_content_sha256=content_hash.hexdigest(),raw_sha256=sha(rawpath),window_indices=wi.tolist(),conditions=details))
  if len(audits)%40==0:print('prepared',len(audits),'/1061 elapsed',round(time.monotonic()-started),flush=True)
 sys.path.insert(0,str(ROOT/'tools/stanford-sample-check'));import inspect_archive;inspect_archive.ARCHIVE=archive;directory_info=inspect_archive.directory_info
 with archive.open('rb',buffering=4*1024*1024) as directory,archive.open('rb') as stream:
  count,_,offset=directory_info(directory);directory.seek(offset);current=None;members={}
  for _ in range(count):
   h=directory.read(46);assert h[:4]==b'PK\x01\x02';f=struct.unpack('<4s6H3L5H2L',h);name=directory.read(f[10]).decode();directory.seek(f[11]+f[12],1);group='/'.join(name.split('/')[:2])
   if group!=current:
    if current in rows:process(rows[current],members,stream)
    current=group;members={}
   if group in rows and name.endswith('_keypoints.json'):
    frame=int(name.rsplit('_',2)[-2])
    if frame<=1499:members[frame]=dict(name=name,offset=f[16],size=f[9],compressed=f[8],crc=f[7],method=f[4])
  if current in rows:process(rows[current],members,stream)
 assert len(audits)==1061 and len({a['record'] for a in audits})==1061
 for x in arrays.values():x.flush()
 np.savez_compressed(B/'summaries.npz',**summaries,quality=qout,**{k:d[k] for k in ['y','person_id','video_id','fold','window_record','window_start','original_row_index']})
 save(H/'source-audit.json',sorted(audits,key=lambda a:a['record']));save(H/'preparation.json',dict(protocol_sha256=sha(H/'protocol.json'),archive_path=str(archive),archive_sha256=archive_hash,summary_sha256=sha(B/'summaries.npz'),input_sha256={k:sha(B/'inputs'/f'{k}.npy') for k in arrays},records=1061,people=692,windows=11261,baseline_max_difference=max(a['conditions']['baseline']['baseline_max_difference'] for a in audits),elapsed_seconds=time.monotonic()-started,quality_names=['missing_fraction','confidence_below03','spike_fraction','normalized_acceleration_median','thigh_length_cv']))
 print('Preparation complete',round(time.monotonic()-started),flush=True)
if __name__=='__main__':main()
