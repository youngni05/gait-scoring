"""Compare the official Caffe forward pass with OpenCV using identical input."""
from pathlib import Path
import os, sys, time, json
ROOT=Path(__file__).resolve().parents[4]
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'))
os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import numpy as np
import cv2
import pyopenpose as op
HERE=Path(__file__).resolve().parent
cv2.setNumThreads(4)
im=cv2.imread(str(HERE/'pose_probe/input/frame.png'))
p={'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25',
   'net_resolution':'-1x368','render_pose':0,'heatmaps_add_parts':True,
   'heatmaps_add_bkg':True,'heatmaps_add_PAFs':True,'heatmaps_scale':3,
   'upsampling_ratio':1,'body':1}
w=op.WrapperPython();w.configure(p);w.start()
d=op.Datum();d.cvInputData=im
t=time.perf_counter();w.emplaceAndPop(op.VectorDatum([d]))
print('official_seconds',time.perf_counter()-t,flush=True)
blob=d.inputNetData[0].copy();hm=d.poseHeatMaps.copy();kp=d.poseKeypoints.copy()
np.savez_compressed(str(HERE/'pose_probe/native_arrays.npz'),blob=blob,heatmaps=hm,keypoints=kp)
w.stop()
net=cv2.dnn.readNetFromCaffe(str(OP/'models/pose/body_25/pose_deploy.prototxt'),str(OP/'models/pose/body_25/pose_iter_584000.caffemodel'))
net.setInput(blob)
t=time.perf_counter();pred=net.forward();elapsed=time.perf_counter()-t
print('opencv_seconds',elapsed,'shapes',blob.shape,hm.shape,pred.shape,flush=True)
print('heatmap_error',np.max(abs(pred[0]-hm)),np.mean(abs(pred[0]-hm)),flush=True)
p.update(body=2,upsampling_ratio=0)
w=op.WrapperPython();w.configure(p);w.start()
d=op.Datum();d.cvInputData=im;d.poseNetOutput=pred[0]
w.emplaceAndPop(op.VectorDatum([d]))
print('keypoints',d.poseKeypoints,flush=True)
np.savez_compressed(str(HERE/'pose_probe/opencv_arrays.npz'),heatmaps=pred[0],keypoints=d.poseKeypoints)
w.stop()
