from pathlib import Path
import os,sys,json,time
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=Path('D:/gait-research-data/D_P/datasets/GAVD');OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
cv2.setNumThreads(2);out=DATA/'official_probe544';out.mkdir(exist_ok=True)
w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25','net_resolution':'-1x544','render_pose':0,'heatmaps_add_parts':True,'heatmaps_add_bkg':True,'heatmaps_add_PAFs':True,'heatmaps_scale':3,'upsampling_ratio':1,'body':1});w.start()
for source in sorted((DATA/'official_probe').glob('*.npz')):
    im=np.load(str(source))['image'].copy();d=op.Datum();d.cvInputData=im;t=time.time();w.emplaceAndPop(op.VectorDatum([d]))
    np.savez_compressed(str(out/source.name),image=im,blob=d.inputNetData[0],heatmaps=d.poseHeatMaps,keypoints=d.poseKeypoints)
    print(json.dumps(dict(file=source.name,seconds=time.time()-t,blob_shape=list(d.inputNetData[0].shape))),flush=True)
w.stop()
