"""Matched-window features; no clinical label is used for extraction or exclusion."""
from pathlib import Path
import sys,json,hashlib,importlib.util,zipfile
HERE=Path(__file__).resolve().parent
ROOT=Path('D:/Projects/D_P'); OLD=ROOT/'studies/20261103-frozen-transfer-and-deviation'
BIG=Path('D:/gait-research-data/D_P/poma-incremental-prediction-20261103'); OB=BIG.parent/'frozen-transfer-and-deviation-20261103'
BASE=ROOT/'studies/20261103-reference188-pilot'
spec=importlib.util.spec_from_file_location('oldprepare',OLD/'prepare.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
np=old.np
from scipy.ndimage import gaussian_filter1d
read=old.read;save=old.save;sha=old.sha

def stats(v):
 return np.stack([v.mean(1),v.std(1),*np.quantile(v,[.05,.25,.5,.75,.95],axis=1),np.sqrt(np.mean((np.diff(v,axis=1)*30)**2,axis=1))],axis=-1)

def describe(points):
 # leg,time,(ankle,knee,hip),coordinate. Same calculation for 2D and 3D.
 p=gaussian_filter1d(points.astype(float),1,axis=1)
 p-=p[:,:,2].mean(0)[None,:,None,:]
 length=np.linalg.norm(p[:,:,2]-p[:,:,1],axis=-1).mean();assert length>1e-9
 p/=length
 a=p[:,:,2]-p[:,:,1];b=p[:,:,0]-p[:,:,1]
 cosine=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
 angle=np.arccos(np.clip(cosine,-1,1))
 delta=p[:,:,0,0]-p[::-1,:,0,0]
 v=np.concatenate([p.reshape(2,len(p[0]),-1),angle[:,:,None],delta[:,:,None]],axis=-1)
 s=stats(v); pooled=np.concatenate([s.mean(0).ravel(),np.abs(s[0]-s[1]).ravel()])
 periodic=[]
 for leg in range(2):
  signal=angle[leg]-angle[leg].mean();corr=np.array([np.dot(signal[:-lag],signal[lag:])/(np.linalg.norm(signal[:-lag])*np.linalg.norm(signal[lag:])+1e-12) for lag in range(15,91)])
  ix=int(np.argmax(corr));periodic.append([(15+ix)/30,float(corr[ix])])
 periodic=np.array(periodic);den=angle.std(1).prod()
 lr=float(np.mean((angle[0]-angle[0].mean())*(angle[1]-angle[1].mean()))/den) if den>1e-12 else 0.
 return np.r_[pooled,periodic.mean(0),np.abs(periodic[0]-periodic[1]),lr]

def main():
 for m in read(HERE/'input-manifest.json'):assert sha(m['path'])==m['sha256']
 rows=read(OLD/'rows.json');z=np.load(OB/'features.npz');pred=np.load(OB/'frozen-predictions.npz');assert np.array_equal(z['owner'],pred['owner'])
 selected=[i for i,r in enumerate(rows) if r['primary'] and r['starts']];ids=sorted({rows[i]['person'] for i in selected})
 accum={p:[] for p in ids};sources=[]
 for number,i in enumerate(selected):
  r=rows[i];wi=np.flatnonzero(z['owner']==i);assert z['start'][wi].tolist()==r['starts']
  with zipfile.ZipFile(r['archive']) as archive:data=archive.read(r['member'])
  assert hashlib.sha256(data).hexdigest()==r['source_sha256']
  tmp=BIG/'extract-current.c3d';tmp.write_bytes(data);c=old.proj.ezc3d.c3d(str(tmp));labels=[n.split(':')[-1].strip() for n in c['parameters']['POINT']['LABELS']['value']]
  xyz=np.full((c['data']['points'].shape[2],len(old.proj.NAMES),3),np.nan)
  for name in list(old.JOINTS.values())+['LASI','RASI','LHEE','RHEE']:
   j=labels.index(name);v=c['data']['points'][:3,j].T/1000;v[c['data']['meta_points']['residuals'][0,j]<0]=np.nan;xyz[:,old.proj.NAMES.index(name)]=v
  t,sampled=old.proj.resample(xyz,r['source_rate']);local,_,_=old.proj.align(sampled);uv,*_=old.proj.project(local,'side_pan')
  joints=[[old.proj.NAMES.index(s+n) for n in ['TIO','FEO','FEP']] for s in 'LR']
  pts3=np.stack([local[:,j] for j in joints]);pts2=np.stack([uv[:,j] for j in joints])
  f2=[];f3=[];fc=[]
  for w,start in zip(wi,r['starts']):
   f2.append(describe(pts2[:,start:start+124]));f3.append(describe(pts3[:,start:start+124]));fc.append(stats(z['x'][w][:,:,old.KEEP]).ravel())
  accum[r['person']].append(dict(common=np.mean(fc,0),joint2D=np.mean(f2,0),joint3D=np.mean(f3,0),gdi=float(pred['fusion'][wi].mean())))
  sources.append(dict(row=i,person=r['person'],source_sha256=r['source_sha256'],window_indices=wi.tolist(),starts=r['starts'],archive=r['archive'],member=r['member']))
  if (number+1)%30==0:print('extracted trials',number+1,'/',len(selected),flush=True)
 arrays={'person_id':np.array(ids)}
 for key in ['common','joint2D','joint3D','gdi']:arrays[key]=np.array([np.mean([x[key] for x in accum[p]],axis=0) for p in ids])
 for key,v in arrays.items():
  if key!='person_id':assert np.isfinite(v).all(),key
 oldscores={p['id']:p for p in read(OLD/'scores.json')}
 assert max(abs(arrays['gdi'][i]-oldscores[p]['frozen']['fusion']) for i,p in enumerate(ids))<1e-9
 linkage=read(BASE/'person-linkage.json');people={old.normid(r['supplement_candidate_ids'][0]):dict(id=old.normid(r['supplement_candidate_ids'][0]),cohort=g,mat_index=r['mat_index'],age=r['values']['Age']) for g in ['healthy','stroke'] for r in linkage[g]['rows']}
 np.savez_compressed(BIG/'matched-features.npz',**arrays)
 save(HERE/'people.json',[people[p] for p in ids]);save(HERE/'source-audit.json',sources)
 save(HERE/'preparation.json',dict(protocol_sha256=sha(HERE/'protocol.json'),features_sha256=sha(BIG/'matched-features.npz'),people=len(ids),trials=len(selected),windows=sum(len(s['starts']) for s in sources),dimensions={k:list(v.shape) for k,v in arrays.items()},all_sources_rehashed=True,same_windows_all_feature_arms=True,frozen_person_scores_reproduced=True))
 print(read(HERE/'preparation.json'))
if __name__=='__main__':main()
