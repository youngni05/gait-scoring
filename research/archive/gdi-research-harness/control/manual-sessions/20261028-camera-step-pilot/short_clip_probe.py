import sys,json,time
from pathlib import Path
S=Path(__file__).parent;sys.path.insert(0,str(S/'deps'))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.signal import find_peaks
results=[]
for cam in ['03','06']:
 cap=cv2.VideoCapture(str(S/f'camera{cam}.mp4'));frames=[]
 while True:
  ok,bgr=cap.read()
  if not ok:break
  frames.append(bgr)
 cap.release()
 for start,end in [(1.5,3.5),(3.5,5.5),(1.5,5.5)]:
  rows=[];t0=time.monotonic();opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(S/'pose_landmarker_full.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
  with vision.PoseLandmarker.create_from_options(opts) as detector:
   for j,i in enumerate(range(round(start*50),round(end*50))):
    result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frames[i],cv2.COLOR_BGR2RGB)),j*20)
    pose=max(result.pose_landmarks,key=lambda p:(max(q.x for q in p)-min(q.x for q in p))*(max(q.y for q in p)-min(q.y for q in p))) if result.pose_landmarks else None
    rows.append(dict(frame=i,time_s=i/50,landmarks=[[p.x,p.y,p.z,p.visibility,p.presence] for p in pose] if pose else None))
  n=len(rows);a=np.full((n,33,5),np.nan)
  for j,r in enumerate(rows):
   if r['landmarks'] is not None:a[j]=r['landmarks']
  xy=a[:,:,:2]*[800,600];hip=(xy[:,23]+xy[:,24])/2;shoulder=(xy[:,11]+xy[:,12])/2;scale=np.linalg.norm(hip-shoulder,axis=1)
  ids=[11,12,23,24,27,28];valid=np.all(np.isfinite(a[:,ids,:2]),axis=(1,2)) & np.all(a[:,ids,3]>=.5,axis=1) & (scale>0)
  valid &= np.all((a[:,ids,:2]>=0)&(a[:,ids,:2]<=1),axis=(1,2))
  distance=np.linalg.norm(xy[:,27]-xy[:,28],axis=1)/scale;peaks=[];runs=[];i=0
  while i<n:
   if not valid[i]:i+=1;continue
   j=i+1
   while j<n and valid[j]:j+=1
   runs.append([i,j])
   if j-i>=5:
    sm=np.convolve(distance[i:j],np.ones(5)/5,mode='valid');p,_=find_peaks(sm,distance=15,prominence=.1);peaks.extend((p+i+2).tolist())
   i=j
  tag=f'{cam}_{start}_{end}';(S/f'pose_clip_{tag}.json').write_text(json.dumps(rows,separators=(',',':')))
  times=np.array(peaks)/50+start;gaps=[not bool(valid[a:b+1].all()) for a,b in zip(peaks[:-1],peaks[1:])]
  record=dict(camera=cam,start_s=start,end_s=end,frames=n,usable_frames=int(valid.sum()),peak_count=len(peaks),peak_times_s=times.tolist(),candidate_intervals_s=np.diff(times).tolist(),interval_crosses_unusable_frames=gaps,elapsed_s=time.monotonic()-t0)
  results.append(record);(S/'short_clip_results.json').write_text(json.dumps(results,indent=2));print(record,flush=True)
