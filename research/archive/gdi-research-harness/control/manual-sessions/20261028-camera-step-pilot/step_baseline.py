import sys,json,math
from pathlib import Path
S=Path(__file__).parent;sys.path.insert(0,str(S/'deps'))
import numpy as np
from scipy.signal import find_peaks
settings=dict(smoothing_frames=5,min_peak_distance_seconds=.30,prominence_body_lengths=.10,min_visibility=.5,require_in_frame=True)
results=[]
for cam in ['03','06']:
 rows=json.loads((S/f'pose{cam}.json').read_text());n=len(rows);a=np.full((n,33,5),np.nan)
 for r in rows:
  if r['landmarks'] is not None:a[r['frame']]=r['landmarks']
 xy=a[:,:,:2]*[800,600];hip=(xy[:,23]+xy[:,24])/2;shoulder=(xy[:,11]+xy[:,12])/2;scale=np.linalg.norm(hip-shoulder,axis=1)
 ids=[11,12,23,24,27,28]; valid=np.all(np.isfinite(a[:,ids,:2]),axis=(1,2)) & np.all(a[:,ids,3]>=.5,axis=1) & (scale>0)
 valid &= np.all((a[:,ids,:2]>=0)&(a[:,ids,:2]<=1),axis=(1,2))
 distance=np.linalg.norm(xy[:,27]-xy[:,28],axis=1)/scale
 peaks=[];runs=[];i=0
 while i<n:
  if not valid[i]: i+=1;continue
  j=i+1
  while j<n and valid[j]:j+=1
  runs.append([i,j]); segment=distance[i:j]
  if len(segment)>=5:
   smoothed=np.convolve(segment,np.ones(5)/5,mode='valid')
   p,_=find_peaks(smoothed,distance=15,prominence=.1);peaks.extend((p+i+2).tolist())
  i=j
 event_time=np.array(peaks)/50
 windows=[]
 for start,end in [(1.5,3.5),(3.5,5.5),(1.5,5.5),(1.5,3.0)]:
  t=event_time[(event_time>=start)&(event_time<end)];intervals=np.diff(t)
  windows.append(dict(start_s=start,end_s=end,peak_count=len(t),peak_times_s=t.tolist(),intervals_s=intervals.tolist(),mean_interval_s=float(intervals.mean()) if len(intervals) else None,usable_frame_fraction=float(valid[int(start*50):int(end*50)].mean())))
 results.append(dict(camera=cam,frames=n,pose_detected_frames=int(np.isfinite(a[:,0,0]).sum()),usable_frames=int(valid.sum()),valid_runs=runs,peak_frames=peaks,windows=windows))
 np.savez_compressed(S/f'signal{cam}.npz',time_s=np.arange(n)/50,ankle_separation=distance,valid=valid)
# Same threshold, same windows. No tuning after observing these outputs.
out=dict(settings=settings,results=results,interpretation='Separation maxima are event proxies, NOT established foot-contact events or clinical scores. View agreement is not accuracy.')
(S/'step_results.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
