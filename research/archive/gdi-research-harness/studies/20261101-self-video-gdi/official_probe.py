"""Python3.7: actual first confidence-contract failure, original Caffe OpenPose."""
from pathlib import Path
import sys,os,json,time,hashlib
import numpy as np,cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101');OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
cv2.setNumThreads(2);w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0});w.start()
row=json.loads((HERE/'inventory.json').read_text(encoding='utf8'))[0];c=cv2.VideoCapture(row['file']);c.set(1,23);ok,im=c.read();assert ok;c.release();d=op.Datum();d.cvInputData=im;t=time.time();w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
np.savez_compressed(str(BIG/'official-clip0-frame23.npz'),keypoints=kp)
old=json.loads((BIG/'body25/clip0/clip0_000000000023_keypoints.json').read_text())['people'];a=np.asarray(old[0]['pose_keypoints_2d']).reshape(25,3)
report=dict(clip='clip0',frame=23,official_max_confidence=float(kp[:,:,2].max()),official_people=len(kp),adapter_max_confidence=float(a[:,2].max()),max_coordinate_difference=float(np.abs(a[:,:2]-kp[0,:,:2]).max()),max_confidence_difference=float(np.abs(a[:,2]-kp[0,:,2]).max()),seconds=time.time()-t)
(HERE/'official-confidence-proof.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True);w.stop()
