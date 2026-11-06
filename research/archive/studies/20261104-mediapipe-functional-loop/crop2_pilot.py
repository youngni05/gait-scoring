"""Score-blind far-corridor crop fallback; keep failures and original observations."""
import sys,json,time,hashlib
from pathlib import Path
DEP=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot');sys.path.insert(0,str(DEP/'deps'))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def windows(valid,times):
    rows=[]
    for lo in range(0,len(valid)-119,120):
        a=np.all(valid[lo:lo+120,[23,24,25,26,27,28]],axis=1);run=0;longest=0
        for yes in a:run=0 if yes else run+1;longest=max(longest,run)
        rows.append(dict(start_frame=lo,end_exclusive=lo+120,start_seconds=float(times[lo]/1000),all_six_joint_valid_fraction=float(a.mean()),longest_bad_gap_frames=longest,usable=bool(a.mean()>=.8 and longest<=6)))
    return rows
def main():
    old=np.load(B/'OAW06-bottom-mediapipe-crop.npz');meta=json.loads((H/'video-metadata.json').read_text());source=json.loads((H/'pilot-video.json').read_text());N=meta['frames'];W=meta['width'];height=meta['height'];assert len(old['landmarks'])==N;task=DEP/'pose_landmarker_full.task';assert sha(task)==json.loads((DEP/'pose_landmarker_full.task.http.json').read_text())['sha256']
    roi=(.38,.10,.72,.38);x0,y0,x1,y1=round(roi[0]*W),round(roi[1]*height),round(roi[2]*W),round(roi[3]*height)
    proto=dict(code_sha256=sha(Path(__file__)),source_sha256=source['sha256'],base_pose_sha256=sha(B/'OAW06-bottom-mediapipe-crop.npz'),task_sha256=sha(task),design='Pilot-specific fixed far-corridor crop x[.38,.72],y[.10,.38], chosen from score-blind camera review. Apply only to frames lacking exactly one original eligible target. FullMediaPipe IMAGE mode,numposes2,unchanged.5detection/presence thresholds. Exactly1eligible pose, all6hip/knee/ankle points require visibility/presence>=.5 and inside crop margins.02..98. Map imagexy back to original pixels; z scaled by cropwidth/originalwidth and is not calibrateddepth. No world3D merge. Original observations preserved bitwise. No coordinate/score imputation.',quality='Fixed120frame windows,starts0/120/...; usable if all6joint visible >=80percent and maxinvalidrun<=6frames. This is operational availability only; no ground-truth joint accuracy or clinical improvement claim.',scope='One14person-dataset pilot, no clinical labels loaded. Fullbody video/camera generalization must be checked on more people/cameras. Retain provider mask; derivative features later must mask provider transitions.')
    if(H/'crop2-protocol.json').exists():assert json.loads((H/'crop2-protocol.json').read_text())==proto
    else:save(H/'crop2-protocol.json',proto)
    coords=old['landmarks'].copy();provider=old['provider'].copy();crop_counts=np.zeros(N,int);fallback=np.flatnonzero(provider<0);cap=cv2.VideoCapture(source['file']);start=time.perf_counter();review=[];cv2.setNumThreads(2)
    opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.IMAGE,num_poses=2,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
    with vision.PoseLandmarker.create_from_options(opts) as detector:
      for n,i in enumerate(fallback):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,bgr=cap.read();assert ok;roi_image=bgr[y0:y1,x0:x1];result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(roi_image,cv2.COLOR_BGR2RGB)));good=[]
        for j,pose in enumerate(result.pose_landmarks):
            if all(min(pose[k].visibility,pose[k].presence)>=.5 and .02<=pose[k].x<=.98 and .02<=pose[k].y<=.98 for k in [23,24,25,26,27,28]):good.append(j)
        crop_counts[i]=len(good)
        if len(good)==1:
            coords[i]=[[(x0+p.x*(x1-x0))/W,(y0+p.y*(y1-y0))/height,p.z*(x1-x0)/W,p.visibility,p.presence] for p in result.pose_landmarks[good[0]]];provider[i]=2
        if(n+1)%200==0:print('fallback',n+1,'recovered',int(sum(provider==2)),round(time.perf_counter()-start,1),flush=True)
    assert np.array_equal(coords[old['provider']>=0],old['landmarks'][old['provider']>=0],equal_nan=True)
    valid=(coords[:,:,3]>=.5)&(coords[:,:,4]>=.5)&np.isfinite(coords[:,:,:3]).all(axis=-1);times=old['source_ms'];before=windows(old['valid'],times);after=windows(valid,times)
    np.savez_compressed(B/'OAW06-bottom-mediapipe-crop2.npz',landmarks=coords,valid=valid,source_ms=times,provider=provider,crop_eligible_counts=crop_counts)
    for i in [int(N*.2),int(N*.5),int(N*.8)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();assert ok;small=cv2.resize(bgr,(480,int(height*480/W)))
        for k in [11,12,23,24,25,26,27,28,29,30,31,32]:
            if np.isfinite(coords[i,k,:2]).all():cv2.circle(small,(round(coords[i,k,0]*small.shape[1]),round(coords[i,k,1]*small.shape[0])),3,(255,0,255) if valid[i,k] else(0,0,255),-1)
        cv2.putText(small,f'{i} provider{provider[i]}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2);review.append(small)
    cap.release();cv2.imwrite(str(H/'mediapipe-crop2-pilot.jpg'),cv2.hconcat(review));save(H/'crop2-verification.json',dict(status='extraction_complete_pending_visual_review',frames=N,original_selected=int(sum(provider==0)),crop_recovered=int(sum(provider==2)),stage1_retained=int(sum(provider==1)),remaining_missing=int(sum(provider<0)),crop_ambiguous=int(sum(crop_counts>1)),original_observations_bitwise_preserved=True,fullbody_operational_windows_before=int(sum(q['usable'] for q in before)),fullbody_operational_windows_after=int(sum(q['usable'] for q in after)),windows_before=before,windows_after=after,output_sha256=sha(B/'OAW06-bottom-mediapipe-crop2.npz'),seconds=time.perf_counter()-start,clinical_accuracy_tested=False));print('crop comparison complete',flush=True)
if __name__=='__main__':main()
