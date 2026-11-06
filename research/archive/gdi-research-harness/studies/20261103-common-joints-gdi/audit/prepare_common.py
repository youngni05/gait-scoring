from pathlib import Path
import sys,json,importlib.util,warnings,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/common-joints-gdi-20261103/audit')
p=Path('D:/Projects/D_P/studies/20261103-reference188-projection-pilot/project.py')
spec=importlib.util.spec_from_file_location('projection_pilot',p);proj=importlib.util.module_from_spec(spec);spec.loader.exec_module(proj)
sys.path.insert(0,'D:/Projects/D_P/outputs/team-share-gdi-normal-cnn-20261031-v2/bundle')
from preprocess import source_functions,LEFT,RIGHT
ns=source_functions();keep=[0,1,2,3,4,5,8,11];columns=[[v[i] for i in keep] for v in [LEFT,RIGHT]]
mapnodes={12:'LFEP',9:'RFEP',13:'LFEO',10:'RFEO',14:'LTIO',11:'RTIO'}
manifest=json.loads((HERE/'manifest.json').read_text());audit=json.loads((HERE/'joint-gdi-audit.json').read_text());dynamic={r['member']:r for r in audit['records'] if not r['static']};rows=[];arrays={}
for source in manifest:
 if source['member'] not in dynamic:continue
 record=dynamic[source['member']];key=source['person']+'_'+Path(source['member']).stem.replace(' ','_').replace('(','').replace(')','')
 c,xyz,_,_=proj.read_source(Path(source['path']));t,xyz=proj.resample(xyz,float(c['header']['points']['frame_rate']));local,basis,origin=proj.align(xyz)
 row=dict(id=key,person=source['person'],member=source['member'],frames=len(t),valid_joint_centers=True,conditions={})
 for kind in proj.CONDITIONS:
  uv,R,C,depth=proj.project(local,kind);raw=np.full((len(t),25,3),np.nan)
  for target,name in mapnodes.items():raw[:,target,:2]=uv[:,proj.NAMES.index(name)]
  # No confidence invented; frozen functions explicitly drop that unused column.
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);processed=ns['process_video_and_add_cols'](raw.reshape(len(t),75),swap_orientation=False)
  features=np.stack([processed[:,cols] for cols in columns]);assert features.shape==(2,len(t),8)
  # Independent algebra on the normalized common joints verifies feature selection.
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning);normalized=ns['preprocess_frames'](raw.reshape(len(t),75),swap_orientation=False)
  independent=[]
  for hip,knee,ankle,other in [(12,13,14,11),(9,10,11,14)]:
   h=normalized[:,2*hip:2*hip+2];k=normalized[:,2*knee:2*knee+2];a=normalized[:,2*ankle:2*ankle+2];otherx=normalized[:,2*other]
   v=h-k;w=a-k;ang=np.arccos(np.sum(v*w,axis=1)/(np.linalg.norm(v,axis=1)*np.linalg.norm(w,axis=1)))
   independent.append(np.column_stack([a,k,h,ang,otherx-a[:,0]]))
  error=float(np.nanmax(np.abs(features-np.array(independent))));assert error<1e-12
  arrays[key+'_'+kind]=features.astype('float32');arrays[key+'_time_s']=t
  joint_uv=uv[:,[proj.NAMES.index(n) for n in mapnodes.values()]];inside=np.isfinite(joint_uv).all(2)&(joint_uv[:,:,0]>=0)&(joint_uv[:,:,0]<640)&(joint_uv[:,:,1]>=0)&(joint_uv[:,:,1]<480)
  # Eligible windows must have all six joints in frame throughout; no missingness imputation used to hide cropping.
  starts={str(n):[s for s in range(0,min(500,len(t))-n+1,31) if inside[s:s+n].all() and np.isfinite(features[:,s:s+n]).all()] for n in [90,124]}
  row['conditions'][kind]=dict(starts=starts,finite_features=bool(np.isfinite(features).all()),independent_feature_max_error=error,all_joints_in_frame_fraction=float(inside.all(1).mean()))
 rows.append(row)
np.savez_compressed(BIG/'common-joint-features.npz',**arrays)
summary={str(n):dict(duration_eligible_records=sum(r['frames']>=n for r in rows),side_pan_eligible_records=sum(bool(r['conditions']['side_pan']['starts'][str(n)]) for r in rows),side_pan_eligible_people=len(set(r['person'] for r in rows if r['conditions']['side_pan']['starts'][str(n)]))) for n in [90,124]}
(HERE/'common-inputs.json').write_text(json.dumps(dict(records=rows,summary=summary,features_sha256=hashlib.sha256((BIG/'common-joint-features.npz').read_bytes()).hexdigest(),joint_mapping=mapnodes,feature_columns=keep,feature_names=['ankle_x','ankle_y','knee_x','knee_y','hip_x','hip_y','knee_angle_rad','other_ankle_x_minus_this_ankle_x'],inference_runs=0,external_timebase_alignment='unverified',occlusion_modeled=False,ground_truth_labels_attached=False),indent=2))
print(json.dumps(summary))
