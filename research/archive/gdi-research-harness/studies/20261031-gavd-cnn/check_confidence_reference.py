"""Diagnose existing confidence gate failures using unchanged official CPU OpenPose."""
from pathlib import Path
import json,os,sys,time
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD');OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
cv2.setNumThreads(2)
inventory=json.loads((DATA/'video_inventory.json').read_text())
w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0,'body':1});w.start();results=[]
for video,record,frame,joint in [('WpARylM4UYU','cljxjv25j001a3n6lw4iv9fwe',104,22),('kA40kglSmYw','cll7ob018002y3o6lngai3iq4',536,5)]:
    cap=cv2.VideoCapture(inventory[video]['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read();cap.release();assert ok
    d=op.Datum();d.cvInputData=im;t=time.time();w.emplaceAndPop(op.VectorDatum([d]));a=d.poseKeypoints.copy()
    source=DATA/'reviewed_shift0'/record/('%s_%012d_keypoints.json'%(record,frame));saved=np.array([p['pose_keypoints_2d'] for p in json.loads(source.read_text())['people']]).reshape(-1,25,3)
    results.append(dict(video=video,annotation_frame=frame,joint=joint,official_confidence=a[:,joint,2].tolist(),official_max_confidence=float(a[:,:,2].max()),official_exceeds_one=bool((a[:,:,2]>1).any()),saved_confidence=saved[:,joint,2].tolist(),max_coordinate_difference=float(np.max(np.abs(a[:,:,:2]-saved[:,:,:2]))) if a.shape==saved.shape else None,seconds=time.time()-t))
    print(json.dumps(results[-1]),flush=True)
w.stop();(HERE/'confidence_reference_check.json').write_text(json.dumps(results,indent=2))
