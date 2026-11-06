"""Frozen MediaPipe baseline on real MoVi videos and fixed resolution variants."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / '20261028-camera-step-pilot'
sys.path.insert(0, str(PILOT/'deps'))
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['full','heavy'],default='full')
variant=parser.parse_args().variant
OUT = ROOT/('baseline' if variant=='full' else 'heavy')
OUT.mkdir(exist_ok=True)
MODEL = PILOT/'pose_landmarker_full.task' if variant=='full' else OUT/'pose_landmarker_heavy.task'
PLAN = {
    'model': 'MediaPipe Pose Landmarker '+variant.title(), 'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    'versions':{p:importlib.metadata.version(p) for p in ['mediapipe','opencv-contrib-python','numpy','scipy']},
    'cameras':['PG1','PG2'],'input_sizes':[[800,600],[400,300],[200,150]] if variant=='full' else [[800,600]],
    'evaluation_frames_one_based':[1341,1517],'warmup_frames_one_based':[1311,1340],
    'options':{'mode':'VIDEO','delegate':'CPU','num_poses':3,'min_pose_detection_confidence':.5,
               'min_pose_presence_confidence':.5,'min_tracking_confidence':.5},
    'actor_selection':'Largest predicted 2D bounding-box area, independent of reference',
    'preprocessing':'RGB; INTER_AREA downsampling for lower-resolution conditions; no crop, enhancement or reference input',
    'comparison':'12 corresponding shoulder/elbow/wrist/hip/knee/ankle positions; raw 2D discrepancy; 3-point geometric knee bend angles',
    'confidence_reporting':'All predictions first; also visibility>=0.5 subset with retained counts, no threshold tuning',
    'turn_context':'Framewise curves; optional 1411-1451 interval is an assistant-selected visual inspection window, not annotated turn truth',
    'limits':['Single person, grayscale lab videos with markers','Spatial calibration and reference/model landmark definitions may contribute to discrepancies',
              'Three-point bend angles are not clinical joint-coordinate-system angles','Reduced resolution is not a real-distance experiment'],
    'no_training':True,'no_accuracy_pass_threshold':True}
(OUT/'plan.json').write_text(json.dumps(PLAN,indent=2)+'\n',encoding='utf-8')
summary=[]
for cam in PLAN['cameras']:
    cap=cv2.VideoCapture(str(ROOT/f'F_{cam}_Subject_1_L.avi'))
    assert cap.get(cv2.CAP_PROP_FPS)==30
    cap.set(cv2.CAP_PROP_POS_FRAMES,1310)
    frames=[]
    for i in range(207):
        ok,bgr=cap.read(); assert ok
        frames.append(bgr)
    cap.release()
    for width,height in PLAN['input_sizes']:
        began=time.monotonic()
        options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL),delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
        landmarks=np.full((207,33,5),np.nan); world=np.full((207,33,5),np.nan); counts=[]
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i,frame in enumerate(frames):
                if width!=800:frame=cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),round(i*1000/30))
                counts.append(len(result.pose_landmarks))
                if result.pose_landmarks:
                    idx=max(range(len(result.pose_landmarks)),key=lambda k:(max(p.x for p in result.pose_landmarks[k])-min(p.x for p in result.pose_landmarks[k]))*(max(p.y for p in result.pose_landmarks[k])-min(p.y for p in result.pose_landmarks[k])))
                    landmarks[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in result.pose_landmarks[idx]]
                    world[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in result.pose_world_landmarks[idx]]
        target=OUT/f'{cam}_{width}.npz'
        np.savez_compressed(target,video_frame_one_based=np.arange(1311,1518),landmarks=landmarks,world_landmarks=world,pose_counts=np.array(counts))
        row={'camera':cam,'width':width,'height':height,'frames_processed':207,'evaluation_frames':177,
             'evaluation_detected':int(np.sum(np.array(counts[30:])>0)),'evaluation_multiple_poses':int(np.sum(np.array(counts[30:])>1)),
             'elapsed_seconds':time.monotonic()-began,'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        summary.append(row); (OUT/'inference-summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(row),flush=True)
