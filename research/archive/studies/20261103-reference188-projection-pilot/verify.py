"""Independent raw-reader, timing and projection verification; no model calls."""
from pathlib import Path
import sys,json,hashlib,math,warnings
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/dataset-audits/20261103-reference188-projection-pilot')
sys.path[:0]=[str(BIG/'deps'),'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps','D:/Projects/D_P/studies/20261103-reference188-pilot/deps']
import numpy as np,c3d,cv2
from PIL import Image
warnings.filterwarnings('ignore',category=UserWarning,module='c3d')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((HERE/'results.json').read_text());manifest=json.loads((HERE/'download-manifest.json').read_text())
assert r['protocol_sha256']==sha(HERE/'protocol.json') and r['data_sha256']==sha(BIG/'projections.npz')
with np.load(BIG/'projections.npz') as f:z={k:f[k] for k in f.files}
names=z['names'].tolist();evidence=[]
for trial,record in zip(r['trials'],manifest):
 person=trial['person'];assert record['person']==person and sha(Path(record['path']))==record['sha256']
 with open(record['path'],'rb') as f:
  reader=c3d.Reader(f);labels=[s.strip().split(':')[-1] for s in reader.point_labels];ix=[labels.index(n) for n in names];source=[];indices=[]
  for index,points,analogs in reader.read_frames():
   selected=points[ix];v=selected[:,:3].astype('float64')/1000;v[selected[:,3]<0]=np.nan;source.append(v);indices.append(index)
  source=np.array(source);assert reader.point_rate==100
  assert indices==list(range(trial['source_first_frame_zero_based']+1,trial['source_last_frame_zero_based']+2))
  assert len(source)==trial['source_frames']
  source_error=float(np.nanmax(np.abs(source-z[person+'_source_m'])))
  assert source_error<1e-12 and np.array_equal(np.isfinite(source),np.isfinite(z[person+'_source_m']))
  # Reader 2 retrieves GDI metadata independently, including conflicting values.
  param=reader.get('ANALYSIS.VALUES')
  if param is not None:
   for v in trial['source_gdi']['analysis']:assert abs(float(param.float_array[v['index']])-v['value'])<1e-9
 t=z[person+'_time_s'];expected_t=np.arange(0,trial['source_elapsed_s']+1e-10,1/30)
 assert len(t)==len(expected_t)==trial['target_frames'] and np.max(np.abs(t-expected_t))<1e-12
 expected=np.empty_like(z[person+'_sampled_m'])
 for j in range(len(names)):
  for a in range(3):expected[:,j,a]=np.interp(t,np.arange(len(source))/100,source[:,j,a])
 sample_error=float(np.nanmax(np.abs(expected-z[person+'_sampled_m'])));assert sample_error<1e-12
 basis=z[person+'_basis'];origin=z[person+'_origin'];local=z[person+'_local_m']
 assert np.max(np.abs(basis@basis.T-np.eye(3)))<1e-12 and abs(np.linalg.det(basis)-1)<1e-12
 assert np.max(np.abs(local@basis+origin-expected))<1e-12
 center=(local[:,names.index('LASI')]+local[:,names.index('RASI')])/2
 cond={}
 for kind in ['side_fixed','side_pan','oblique_fixed']:
  uv=z[person+'_'+kind+'_uv'];rot=z[person+'_'+kind+'_R'];pos=z[person+'_'+kind+'_C'];error=0.
  assert np.max(np.abs(pos-np.array([-3.5*np.tan(np.pi/12) if kind=='oblique_fixed' else 0,-3.5,1])))<1e-12
  assert np.max(np.abs(rot@rot.transpose(0,2,1)-np.eye(3)))<1e-12 and np.max(np.abs(np.linalg.det(rot)-1))<1e-12
  for i in range(len(t)):
   rv,_=cv2.Rodrigues(rot[i]);tv=-rot[i]@pos
   projected,_=cv2.projectPoints(local[i],rv,tv,z['K'],np.zeros(5));error=max(error,float(np.nanmax(np.abs(projected[:,0]-uv[i]))))
   # Verify prescribed yaw independently rather than merely checking matrix algebra.
   expected_yaw=math.atan2(center[i,0]-pos[0],center[i,1]-pos[1]) if kind=='side_pan' else (np.pi/12 if kind=='oblique_fixed' else 0)
   assert abs(math.atan2(rot[i,2,0],rot[i,2,1])-expected_yaw)<1e-12
  assert error<1e-8
  physical=uv[:,:32];inside=(physical[:,:,0]>=0)&(physical[:,:,0]<640)&(physical[:,:,1]>=0)&(physical[:,:,1]<480)
  assert abs(inside.all(1).mean()-trial['cameras'][kind]['all_physical_markers_in_frame_fraction'])<1e-12
  cond[kind]=dict(opencv_max_pixel_error=error,all_markers_in_frame_fraction=float(inside.all(1).mean()))
 gif=Image.open(BIG/(person+'-cameras.gif'));assert gif.size==(1200,300) and gif.n_frames==math.ceil(len(t)/3)
 assert trial['duration_pass']==(len(t)>=124) and not trial['strict_gdi_input_pass']
 evidence.append(dict(person=person,raw_second_reader_max_error_m=source_error,c3d_frame_start_one_based=indices[0],c3d_frame_end_one_based=indices[-1],reader_index_convention='ezc3d header 0-based, c3d/TRIAL fields 1-based; coordinate arrays align',resampling_max_error_m=sample_error,cameras=cond,preview_frames=gif.n_frames,duration_eligible=trial['duration_pass']))
assert len(evidence)==6 and sum(e['duration_eligible'] for e in evidence)==2
assert r['gdi_inference_runs']==0 and r['training_runs']==0
assert not list(HERE.glob('*prediction*')) and not list(BIG.glob('*prediction*'))
assert not list(HERE.glob('*.pt')) and not list(BIG.glob('*.pt'))
out=dict(status='passed',trials=6,camera_sequences=18,source_files_crc_verified_in_fetch=True,checks=evidence,strict_gdi_eligible=0,model_prediction_artifacts=[],protocol_sha256=sha(HERE/'protocol.json'),projections_sha256=sha(BIG/'projections.npz'),code_sha256={p.name:sha(p) for p in [HERE/'fetch.py',HERE/'project.py',HERE/'verify.py']})
(HERE/'verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
