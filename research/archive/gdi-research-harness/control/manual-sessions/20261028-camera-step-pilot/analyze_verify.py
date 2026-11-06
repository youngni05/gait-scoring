import sys,json,hashlib
from pathlib import Path
S=Path(__file__).parent;sys.path.insert(0,str(S/'deps'))
import cv2,numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
short=json.loads((S/'short_clip_results.json').read_text());full=json.loads((S/'step_results.json').read_text())
comparisons=[];timestamps=[]
for cam in ['03','06']:
 rows=json.loads((S/f'pose{cam}.json').read_text());cap=cv2.VideoCapture(str(S/f'camera{cam}.mp4'));ts=[]
 while True:
  ok,_=cap.read()
  if not ok:break
  ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
 cap.release();assert len(ts)==511;assert np.allclose(ts,np.arange(511)*20,atol=.1)
 timestamps.append(dict(camera=cam,frames=len(ts),strictly_increasing=bool(np.all(np.diff(ts)>0)),max_nominal_difference_ms=float(np.max(np.abs(np.array(ts)-np.arange(511)*20)))))
 for r in [a for a in short if a['camera']==cam]:
  selected=[p for p in rows if r['start_s']<=p['time_s']<r['end_s']];a=np.array([p['landmarks'] if p['landmarks'] else [[np.nan]*5]*33 for p in selected]);n=len(a)
  xy=a[:,:,:2]*[800,600];scale=np.linalg.norm((xy[:,23]+xy[:,24]-xy[:,11]-xy[:,12])/2,axis=1);ids=[11,12,23,24,27,28]
  v=np.all(np.isfinite(a[:,ids,:2]),axis=(1,2)) & np.all(a[:,ids,3]>=.5,axis=1)&(scale>0)&np.all((a[:,ids,:2]>=0)&(a[:,ids,:2]<=1),axis=(1,2))
  d=np.linalg.norm(xy[:,27]-xy[:,28],axis=1)/scale;pks=[];i=0
  while i<n:
   if not v[i]:i+=1;continue
   j=i+1
   while j<n and v[j]:j+=1
   if j-i>=5:
    z=np.convolve(d[i:j],np.ones(5)/5,'valid');p,_=find_peaks(z,distance=15,prominence=.1);pks.extend((p+i+2).tolist())
   i=j
  times=np.array(pks)/50+r['start_s'];same=len(times)==len(r['peak_times_s']) and np.allclose(times,r['peak_times_s'],atol=.020001)
  source=next(x for x in full['results'] if x['camera']==cam);w=next(x for x in source['windows'] if x['start_s']==r['start_s'] and x['end_s']==r['end_s'])
  comparisons.append(dict(camera=cam,start=r['start_s'],end=r['end_s'],full_inference_then_select_count=w['peak_count'],crop_coordinates_then_detect_count=len(times),fresh_short_inference_count=r['peak_count'],cropped_peak_times_s=times.tolist(),fresh_agrees_with_cropped_within_one_frame=bool(same)))
(S/'boundary_comparison.json').write_text(json.dumps(comparisons,indent=2));(S/'timestamp_verification.json').write_text(json.dumps(timestamps,indent=2))
ref=json.loads((S/'provisional_reference.json').read_text(encoding='utf-8-sig'));refevents=ref['events'];refcomparison=[]
for cam in ['03','06']:
 rr=next(x for x in full['results'] if x['camera']==cam);win=next(w for w in rr['windows'] if w['end_s']==3.0)
 record=dict(camera=cam,provisional_reference_count=2,candidate_count=win['peak_count'],count_difference=win['peak_count']-2,accuracy_status='not_validated_AI_reference_only')
 if win['peak_count']==2:
  record['candidate_minus_reference_time_ranges_s']=[[round(t-e['time_interval_s'][1],3),round(t-e['time_interval_s'][0],3)] for t,e in zip(win['peak_times_s'],refevents)]
  record['candidate_interval_s']=win['intervals_s'][0];record['reference_interval_range_s']=ref['step_interval_reference_range_s']
 else:record['time_error_not_computed_reason']='Different count; event correspondence not established.'
 refcomparison.append(record)
(S/'provisional_comparison.json').write_text(json.dumps(refcomparison,indent=2))
fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True)
for ax,cam in zip(axes,['03','06']):
 z=np.load(S/f'signal{cam}.npz');time=z['time_s'];dist=z['ankle_separation'];valid=z['valid'];mask=(time>=1.5)&(time<5.5)
 ax.plot(time[mask],dist[mask],color='lightgrey',label='all estimates');ax.plot(time[mask],np.where(valid[mask],dist[mask],np.nan),label='usable estimates')
 peaks=next(r for r in short if r['camera']==cam and r['end_s']-r['start_s']==4)['peak_times_s']
 for p in peaks:ax.axvline(p,color='darkorange',alpha=.65)
 ax.set_title(f'Camera {cam}: {len(peaks)} candidate peaks (NOT confirmed steps)');ax.set_ylabel('ankle gap / torso');ax.legend(loc='upper right',fontsize=7)
axes[-1].set_xlabel('original video time (seconds)');fig.tight_layout();fig.savefig(S/'view_comparison.png',dpi=140);plt.close(fig)
# Verify acquired artifacts and original sources without changing them.
for m in json.loads((S/'video_manifest.json').read_text()):assert hashlib.sha256((S/f"camera{m['camera']}.mp4").read_bytes()).hexdigest()==m['sha256']
manifest=json.loads((S/'app_source_manifest.json').read_text(encoding='utf-8-sig'))
for x in manifest:assert hashlib.sha256(Path(x['source']).read_bytes()).hexdigest()==x['sha256']
verification=dict(video_files=2,unique_actor_walking_trials=1,full_frames=1022,fresh_clip_runs=len(short),fresh_clip_frames=sum(r['frames'] for r in short),source_hashes_unchanged=len(manifest),video_hashes_valid=True,timestamps_verified=True,reference_status='provisional_AI_only',clinical_accuracy_verified=False,new_model_training=False,protected_gdi_data_used=False,boundary_comparisons=6)
(S/'research_verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps({'verification':verification,'boundary':comparisons,'provisional':refcomparison},indent=2))
