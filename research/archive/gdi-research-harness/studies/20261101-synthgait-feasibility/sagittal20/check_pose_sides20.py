"""One real frame: same-setting Caffe reference, mirrored diagnostic and overlays."""
from pathlib import Path
import sys,os,json
import cv2,numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20');OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
def main():
    cv2.setNumThreads(2)
    with np.load(str(DATA/'official_probe/frame_0023.npz'),allow_pickle=False) as z:im=z['image'].copy()
    mirrored=np.ascontiguousarray(im[:,::-1]);w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/',
        'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0,'body':1});w.start();arrays=[]
    for image in [im,mirrored]:
        d=op.Datum();d.cvInputData=image;w.emplaceAndPop(op.VectorDatum([d]));assert d.poseKeypoints is not None and len(d.poseKeypoints)==1
        arrays.append(d.poseKeypoints[0].copy())
    w.stop();original,mirror=arrays
    f=DATA/'body25/vid_00002_saggital/vid_00002_saggital_000000000023_keypoints.json'
    gpu=np.asarray(json.loads(f.read_text())['people'][0]['pose_keypoints_2d']).reshape(25,3)
    ids=[9,10,11,12,13,14,19,22];valid=(original[ids,2]>0)&(gpu[ids,2]>0)
    delta=np.abs(original[ids][valid]-gpu[ids][valid])
    pairs=np.arange(25)
    for a,b in [(2,5),(3,6),(4,7),(9,12),(10,13),(11,14),(15,16),(17,18),(19,22),(20,23),(21,24)]:pairs[a]=b;pairs[b]=a
    restored=mirror[pairs].copy();restored[:,0]=im.shape[1]-1-restored[:,0]
    mask=(original[ids,2]>0)&(restored[ids,2]>0)
    distance=np.linalg.norm(original[ids,:2]-restored[ids,:2],axis=1)[mask]
    for label,image,kp in [('original',im,original),('mirrored',mirrored,mirror)]:
        image=image.copy()
        for joints,color in [([12,13,14,19],(0,0,255)),([9,10,11,22],(255,0,0))]:
            for a,b in zip(joints,joints[1:]):
                if kp[a,2]>0 and kp[b,2]>0:cv2.line(image,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),color,2)
            for j in joints:
                if kp[j,2]>0:
                    xy=tuple(kp[j,:2].astype(int));cv2.circle(image,xy,4,color,-1);cv2.putText(image,str(j),xy,cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.putText(image,'LEFT=red  RIGHT=blue; detector labels, not ground truth',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(10,10,10),1)
        cv2.imwrite(str(DATA/('pose_'+label+'.png')),image)
    np.savez_compressed(str(DATA/'same_setting_reference.npz'),original=original,mirror=mirror,gpu=gpu)
    result={'sample_id':'vid_00002_saggital','frame':23,'source_fps':16,
        'same_settings_caffe_vs_gpu_paf_max_xy_pixels':float(delta[:,:2].max()),'same_settings_max_confidence_error':float(delta[:,2].max()),
        'mirror_restored_required_joint_median_pixels':float(np.median(distance)),'mirror_restored_required_joint_max_pixels':float(distance.max()),
        'diagnostic_not_anatomical_ground_truth':True,'left_body25_ids':[12,13,14,19],'right_body25_ids':[9,10,11,22],
        'note':'Compare original and mirrored images after inverse horizontal reflection plus BODY25 left/right permutation. A mismatch measures detector instability; there is no released true skeleton/side identity.',
        'original_overlay':str(DATA/'pose_original.png'),'mirrored_overlay':str(DATA/'pose_mirrored.png')}
    (HERE/'real-frame-side-diagnostic.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
