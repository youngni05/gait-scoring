"""End-to-end official default OpenPose vs raw-map replay at three real sizes."""
from pathlib import Path
import os,sys,json,time
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
OP=ROOT/'tmp/openpose-cpu/openpose';sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
cv2.setNumThreads(2)
probes=[]
for p in sorted((DATA/'official_probe').glob('*.npz')):
    with np.load(str(p)) as z:probes.append((p.name,z['image'].copy(),z['heatmaps'].copy()))
params={'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0,'body':1}
w=op.WrapperPython();w.configure(params);w.start();reference=[]
for name,im,hm in probes:
    d=op.Datum();d.cvInputData=im;w.emplaceAndPop(op.VectorDatum([d]));reference.append(d.poseKeypoints.copy());print(name,'reference',d.poseKeypoints.shape,flush=True)
w.stop();params['body']=2;w=op.WrapperPython();w.configure(params);w.start();results=[]
for (name,im,hm),expected in zip(probes,reference):
    d=op.Datum();d.cvInputData=im;d.poseNetOutput=hm;w.emplaceAndPop(op.VectorDatum([d]));actual=d.poseKeypoints
    assert actual is not None and actual.shape==expected.shape
    delta=np.abs(actual-expected)
    # Pixel coordinates and unitless heatmaps are different numerical scales.
    # Require <= 4 float32 ULP per output (plus epsilon at zero); no pixel-scale tolerance.
    allowance=4*np.maximum(np.spacing(np.abs(expected)),np.finfo(np.float32).eps)
    ulp_ratio=float(np.max(delta/np.maximum(np.spacing(np.abs(expected)),np.finfo(np.float32).eps)))
    results.append(dict(file=name,people=len(actual),max_xy_error=float(delta[:,:,:2].max()),max_confidence_error=float(delta[:,:,2].max()),max_float32_ulps=ulp_ratio,pass_float32_roundoff=bool((delta<=allowance).all())))
    print(results[-1],flush=True)
w.stop();report=dict(results=results,status='passed' if all(r['pass_float32_roundoff'] for r in results) else 'failed',reference='Official BODY25 default upsampling (0), three real GAVD images',note='NP arrays kept alive during synchronous pybind call; initial uniform 1e-4 coordinate check failed at 0.000122 pixels in 854px image. Check float32 rounding units explicitly, separately from unchanged 1e-4 heatmap gate.')
(HERE/'decoder_verification.json').write_text(json.dumps(report,indent=2));assert report['status']=='passed'
