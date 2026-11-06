import sys,json,time,hashlib,importlib.metadata
from pathlib import Path
S=Path(__file__).parent;sys.path.insert(0,str(S/'deps'))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
versions={k:importlib.metadata.version(k) for k in ['mediapipe','opencv-contrib-python','numpy','scipy']}
(S/'versions.json').write_text(json.dumps(versions,indent=2))
summary=[]
for cam in ['03','06']:
 start=time.monotonic();cap=cv2.VideoCapture(str(S/f'camera{cam}.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);rows=[]; i=0;missing=0
 opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(S/'pose_landmarker_full.task'),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
 with vision.PoseLandmarker.create_from_options(opts) as detector:
  while True:
   ok,bgr=cap.read()
   if not ok:break
   result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),round(i*1000/fps))
   poses=result.pose_landmarks
   if poses:
    # One actor in each provided clip. Largest visible bounding box if false extra detections occur.
    pose=max(poses,key=lambda p:(max(q.x for q in p)-min(q.x for q in p))*(max(q.y for q in p)-min(q.y for q in p)))
    arr=[[p.x,p.y,p.z,p.visibility,p.presence] for p in pose]
   else:arr=None;missing+=1
   rows.append(dict(frame=i,time_s=i/fps,poses=len(poses),landmarks=arr));i+=1
 cap.release();(S/f'pose{cam}.json').write_text(json.dumps(rows,separators=(',',':')))
 summary.append(dict(camera=cam,frames=i,missing=missing,elapsed_seconds=time.monotonic()-start,fps=fps))
 (S/'pose_summary.json').write_text(json.dumps(summary,indent=2));print(summary[-1],flush=True)
