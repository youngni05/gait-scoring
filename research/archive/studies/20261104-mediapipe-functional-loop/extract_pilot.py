"""Real MediaPipe extraction pilot with score-blind corridor targeting and full masks."""
import sys,json,time,hashlib
from pathlib import Path
DEP=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot');sys.path.insert(0,str(DEP/'deps'))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def main():
    task=DEP/'pose_landmarker_full.task';assert sha(task)==json.loads((DEP/'pose_landmarker_full.task.http.json').read_text())['sha256'];source=json.loads((H/'pilot-video.json').read_text());v=Path(source['file']);assert sha(v)==source['sha256'];meta=json.loads((H/'video-metadata.json').read_text());N=meta['frames'];fps=meta['fps'];W=meta['width'];height=meta['height'];cv2.setNumThreads(2)
    protocol=dict(code_sha256=sha(Path(__file__)),source_sha256=source['sha256'],model_sha256=sha(task),mediapipe=mp.__version__,runtime='CPU VIDEO mode,Full,num_poses2, thresholds.5, original30fps every decoded frame; tracker only monotone rounded source milliseconds.',target='Pilot OAW06: score-blind visual inspection at20/50/80percent shows participant in central/right corridor, assistant onleft. Fixed eligible pelvis image-x interval[.28,.85],both hip visibility>=.5. Exactly one eligible pose required; ambiguity/missing -> missing, never substitute assistant. No inference of clinical score in pilot.',camera_gate='Actual frontal/back fixed camera, no equivalence to original lateral GDI camera. New view-specific TUG feasibility; no frozen GDI inference.',representations=['raw33x/y/z/visibility/presence and world landmarks retained','candidate stable-window lower-body scale and pelvis-centered image xy','candidate image xy+bone vectors+time derivatives with confidence masks','candidate relative MediaPipe depth separately, never treated as calibrated 3D metres'],clinical_gate='14unique people in release, real TUG linkage still requires file-level ID/label audit before any supervised fit. No SPPB/trueGDI fabricated. Xsens/video not synchronized, so no framewise paired3D loss yet.',limits='One smallest-video pilot, no accuracy, person coverage or generalization evidence. Confidence is uncalibrated visibility/presence. Camera-specific corridor may fail other videos and must be checked score-blind.')
    if(H/'extraction-protocol.json').exists():assert json.loads((H/'extraction-protocol.json').read_text())==protocol
    else:save(H/'extraction-protocol.json',protocol)
    coords=np.full((N,33,5),np.nan);world=np.full((N,33,3),np.nan);counts=np.zeros(N,int);eligible=np.zeros(N,int);source_ms=np.zeros(N);tracker_ms=np.zeros(N,np.int64);selected=np.full(N,-1,int);adjustments=[];prior=-1;cap=cv2.VideoCapture(str(v));start=time.perf_counter();review=[]
    options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=2,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    with vision.PoseLandmarker.create_from_options(options) as detector:
      for i in range(N):
        ok,bgr=cap.read();assert ok;source_ms[i]=cap.get(cv2.CAP_PROP_POS_MSEC);ts=max(int(round(source_ms[i])),prior+1);tracker_ms[i]=ts
        if ts!=int(round(source_ms[i])):adjustments.append(i)
        prior=ts;result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),ts);counts[i]=len(result.pose_landmarks);good=[]
        for j,pose in enumerate(result.pose_landmarks):
            px=(pose[23].x+pose[24].x)/2
            if .28<=px<=.85 and min(pose[23].visibility,pose[24].visibility)>=.5:good.append(j)
        eligible[i]=len(good)
        if len(good)==1:
            j=good[0];selected[i]=j;coords[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in result.pose_landmarks[j]]
            if len(result.pose_world_landmarks)>j:world[i]=[[p.x,p.y,p.z] for p in result.pose_world_landmarks[j]]
        if i in [int(N*.2),int(N*.5),int(N*.8)]:
            small=cv2.resize(bgr,(480,int(height*480/W)))
            if selected[i]>=0:
                for k in [11,12,23,24,25,26,27,28,29,30,31,32]:
                    x,y,_,vis,pres=coords[i,k];cv2.circle(small,(round(x*small.shape[1]),round(y*small.shape[0])),3,(255,0,255) if min(vis,pres)>=.5 else (0,0,255),-1)
            cv2.putText(small,f'{i} poses{counts[i]} eligible{eligible[i]}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2);review.append(small)
        if(i+1)%300==0:print('frame',i+1,'selected',int(sum(selected[:i+1]>=0)),'seconds',round(time.perf_counter()-start,1),flush=True)
    assert not cap.read()[0];cap.release();valid=(coords[:,:,3]>=.5)&(coords[:,:,4]>=.5)&np.isfinite(coords[:,:,:3]).all(axis=-1);np.savez_compressed(B/'OAW06-bottom-mediapipe.npz',landmarks=coords,world_landmarks=world,valid=valid,source_ms=source_ms,tracker_ms=tracker_ms,pose_counts=counts,eligible_counts=eligible,selected_indices=selected)
    cv2.imwrite(str(H/'mediapipe-camera-pilot.jpg'),cv2.hconcat(review));save(H/'extraction-verification.json',dict(status='extraction_complete_pending_visual_and_input_review',frames=N,selected_frames=int(sum(selected>=0)),missing_frames=int(sum(eligible==0)),ambiguous_frames=int(sum(eligible>1)),pose_count_histogram={str(k):int(sum(counts==k)) for k in np.unique(counts)},hip_knee_ankle_joint_valid_fraction=np.mean(valid[:,[23,24,25,26,27,28]],axis=0).tolist(),timestamp_monotonic=bool(np.all(np.diff(source_ms)>0)),tracker_timestamp_adjustments=adjustments,seconds=time.perf_counter()-start,output_sha256=sha(B/'OAW06-bottom-mediapipe.npz'),protocol_sha256=sha(H/'extraction-protocol.json'),clinical_accuracy_tested=False));print('actual MediaPipe pilot extracted',flush=True)
if __name__=='__main__':main()
