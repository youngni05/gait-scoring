"""Python 3.7: original OpenPose probes at actual GAVD aspect ratios."""
from pathlib import Path
import json,os,sys,time
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
cv2.setNumThreads(2)
w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25',
    'net_resolution':'-1x368','render_pose':0,'heatmaps_add_parts':True,'heatmaps_add_bkg':True,
    'heatmaps_add_PAFs':True,'heatmaps_scale':3,'upsampling_ratio':1,'body':1});w.start()
out=DATA/'official_probe';out.mkdir(parents=True,exist_ok=True)
items=[('diCVwltkV5M',1721),('eCCYhDSDlDc',453),('-0wbleNgAwg',411)]
reports=[]
for video,frame in items:
    dest=out/(video+'.npz')
    if dest.exists():continue
    cap=cv2.VideoCapture(str(DATA/'videos'/(video+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();cap.release();assert ok
    d=op.Datum();d.cvInputData=im;t=time.time();w.emplaceAndPop(op.VectorDatum([d]))
    np.savez_compressed(str(dest),image=im,blob=d.inputNetData[0],heatmaps=d.poseHeatMaps,keypoints=d.poseKeypoints)
    report=dict(video=video,decoded_frame=frame,seconds=time.time()-t,shape=list(im.shape),people=len(d.poseKeypoints) if d.poseKeypoints is not None else 0)
    reports.append(report);print(json.dumps(report),flush=True)
w.stop();(HERE/'official_probe_run.json').write_text(json.dumps(reports,indent=2))
