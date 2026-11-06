"""Official OpenPose CPU PAF decoding stage; run in the existing Python 3.7 env."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/self-video-mediapipe-20261101')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op

def main():
    w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0});w.start();cv2.setNumThreads(2)
    print('DECODER_STARTED',flush=True)
    times=[];rows=[]
    for row in json.loads((HERE.parent/'20261101-self-video-gdi/inventory.json').read_text(encoding='utf8')):
        z=np.load(str(BIG/(row['id']+'-benchmark-heatmaps.npz')));maps=z['heatmaps'];image=np.zeros((row['height'],row['width'],3),np.uint8)
        print('LOADED',row['id'],maps.shape,flush=True)
        # Keep the NumPy storage alive while native OpenPose uses its pointer.
        warm_map=np.ascontiguousarray(maps[0])
        d=op.Datum();d.cvInputData=image;d.poseNetOutput=warm_map;w.emplaceAndPop(op.VectorDatum([d]))
        kps=[]
        for hm in maps:
            t=time.perf_counter();d=op.Datum();d.cvInputData=image;d.poseNetOutput=np.ascontiguousarray(hm);w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
            times.append(time.perf_counter()-t);kps.append([] if kp is None else kp.tolist())
        rows.append(dict(clip=row['id'],people=[len(x) for x in kps],keypoints=kps))
    w.stop()
    report=dict(frames=len(times),seconds=sum(times),fps=len(times)/sum(times),per_frame_seconds=times,
        scope='Official CPU PAF decoder only, precomputed maps loaded before timing, one warmup each clip',rows=rows)
    (HERE/'benchmark-decode.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('PAF',len(times),sum(times),flush=True)

if __name__=='__main__':main()
