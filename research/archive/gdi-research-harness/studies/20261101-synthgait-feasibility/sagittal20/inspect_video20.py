"""Python3.7, reuse existing OpenCV and official OpenPose CPU without installing."""
from pathlib import Path
import sys,os,json,hashlib,time
import cv2,numpy as np
ROOT=Path('D:/Projects/D_P')
CODE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20')
OP=ROOT/'tmp/openpose-cpu/openpose'
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    cv2.setNumThreads(2)
    manifest=json.loads((CODE/'download-manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for r in manifest['records']:
        cap=cv2.VideoCapture(r['files']['.mp4']['path']);assert cap.isOpened()
        fps=cap.get(cv2.CAP_PROP_FPS);declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape==(480,832,3);count+=1
        cap.release();assert count==declared and abs(fps-16)<1e-9
        strikes=sorted(r['metadata']['left_heel_strikes']+r['metadata']['right_heel_strikes'])
        rows.append(dict(sample_id=r['sample_id'],split=r['split'],fps=fps,frame_count=count,
            last_frame_seconds=(count-1)/fps,video_duration_seconds=count/fps,
            last_strike_seconds=max(strikes)/30 if strikes else None,
            native_124_frames_available=count>=124,cadence=r['metadata']['gait_features']['cadence']))
    save(CODE/'video-inventory.json',dict(records=rows,all20_decoded=True,source_fps=16,
        annotation_fps=30,training_gdi_json_fps=None,time_match_to_gdi_verified=False))
    print(json.dumps({'all20_decoded':True,'frame_counts':[r['frame_count'] for r in rows],
                      'native124':sum(r['native_124_frames_available'] for r in rows)}),flush=True)
    sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
    import pyopenpose as op
    w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25',
        'net_resolution':'-1x368','render_pose':0,'heatmaps_add_parts':True,'heatmaps_add_bkg':True,
        'heatmaps_add_PAFs':True,'heatmaps_scale':3,'upsampling_ratio':1,'body':1});w.start()
    probe=DATA/'official_probe';probe.mkdir(exist_ok=True)
    record=manifest['records'][0];n=rows[0]['frame_count'];cap=cv2.VideoCapture(record['files']['.mp4']['path']);reports=[]
    for k,frame in enumerate([n//4,n//2,3*n//4]):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
        d=op.Datum();d.cvInputData=im;t=time.time();w.emplaceAndPop(op.VectorDatum([d]))
        keypoints=d.poseKeypoints
        path=probe/('frame_%04d.npz'%frame)
        np.savez_compressed(str(path),image=im,blob=d.inputNetData[0],heatmaps=d.poseHeatMaps,
            keypoints=np.asarray(keypoints) if keypoints is not None else np.empty((0,25,3)))
        report=dict(sample_id=record['sample_id'],frame=frame,seconds=time.time()-t,people=len(keypoints) if keypoints is not None else 0,
                    path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reports.append(report);print(json.dumps(report),flush=True)
    cap.release();w.stop();save(CODE/'official-probe.json',reports)
if __name__=='__main__':main()
