"""Same score-blind MediaPipe pipeline for fourteen source participants, local only."""
import sys,os,json,time,hashlib
from pathlib import Path
DEP=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot');sys.path.insert(0,str(DEP/'deps'))
import cv2,numpy as np,mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');OUT=B/'bottom14';OUT.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def eligible(poses):return [j for j,p in enumerate(poses) if .35<=(p[23].x+p[24].x)/2<=.70 and min(p[23].visibility,p[24].visibility,p[23].presence,p[24].presence)>=.5]
def quality(valid,times):
    assert np.all(np.diff(times)>0),'Source presentation times must be monotone; do not invent physical timing'
    ends=np.r_[times[1:],times[-1]+np.median(np.diff(times))];rows=[]
    for start in np.arange(0,ends[-1]-4000+1e-6,4000):
        finish=start+4000;lo=int(np.searchsorted(ends,start,side='right'));hi=int(np.searchsorted(times,finish,side='left'));dur=np.maximum(0,np.minimum(ends[lo:hi],finish)-np.maximum(times[lo:hi],start));a=np.all(valid[lo:hi,[23,24,25,26,27,28]],axis=1);run=0.;longest=0.
        for good,ms in zip(a,dur):run=0. if good else run+float(ms);longest=max(longest,run)
        fraction=float(dur[a].sum()/4000);rows.append(dict(start_frame=lo,end_exclusive=hi,start_ms=float(start),end_ms=float(finish),source_frames=hi-lo,complete_joint_fraction=fraction,longest_bad_gap_ms=longest,operationally_usable=bool(fraction>=.8 and longest<=200+1e-6)))
    return rows
def main():
    manifest=json.loads((H/'all-bottom-videos-manifest.json').read_text());camera=json.loads((H/'camera14-metadata.json').read_text());byid={r['person_id']:r for r in camera['records']};task=DEP/'pose_landmarker_full.task';assert manifest['people']==14 and manifest['whole_archive_md5_verified'];assert sha(task)==json.loads((DEP/'pose_landmarker_full.task.http.json').read_text())['sha256'];cv2.setNumThreads(2)
    proto=dict(code_sha256=sha(Path(__file__)),source_manifest_sha256=sha(H/'all-bottom-videos-manifest.json'),camera_metadata_sha256=sha(H/'camera14-metadata.json'),model_sha256=sha(task),mediapipe=mp.__version__,conditions='Matched full-frame and one predeclared far-crop fallback, all14bottom-camera subjects; no clinical labels loaded here.',camera_review='Source20/50/80percent frames for all14reviewed before batching. Subject generallycentral; assistant onleft exceptOAW13right. Narrow common hip-x corridor[.35,.70] avoids knownleft/right assistants. Review is not ground-truth identity annotation at everyframe; target errors/ambiguous frames must be audited before clinical fit.',options='FullCPU,numposes2,thresholds.5; preserve every native frame and its original presentation time (files28/29/30fps). Full VIDEOtracker, crop IMAGEdetector. Preserve raw33xy/z/visibility/presence. World3D not used. No target swap or per-person threshold tuning.',target='Exactly1eligible pose by hipvisibility/presence>=.5 and pelvis-x[.35,.70]. Ifmultiple candidates mark ambiguous and doNOTfallback to another person. Ifnone, crop x[.25,.85],y[.08,.50],requireall6hip/knee/anklevisibility/presence>=.5 andinsidecropmargins.02..98; map backtooriginalimage. Cropz relativeonly,notmetres. Keep full-frame and combined arrays/provider masks. No interpolation.',operational_windows='Four seconds of RELEASED VIDEO PRESENTATION TIME, starts0/4000ms/...; duration-weighted all6jointvalid>=80percent and longestbadgap<=200ms. Do not equate120nativeframeswith4s at28/29fps. No phase/time stretching. Original camera physical clock not independently calibrated: publisher says30fps but released files28..30fps; flag this before interpreting physical cadence. Operational availability ONLY; walking/turn/stationary intervals and actoridentity need subsequent review. Firstvalid4s can be selected from longer sourcevideo; no4secondacquisitionlatency claim.',future_score_rules='Keep14unique people grouped across allwindows/cameras. No whole-video length/window count/absolute-time feature. Clinical ID/label join separately audited, no supervised clinical fit here. Later primary shortwindow score uses first qualified4s, never score-dependent bestwindow; all-window summaries are longer-observation secondary. Any120sample model input must resample released-clock4s grid with bounded valid interpolation/masks, notstretch arbitrary120frames. Never callmissinglow score.',limits='Frontal/back fixed camera; not frozen lateralGDI input. Camera/Xsensunsynchronized; originalcapture versus release timebase not verified. No rawvideo/pose uploaded. Dataset14people andoccludedfaces; no clinical/camera accuracy claim from detection coverage.')
    pp=H/'batch14-protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==proto
    else:save(pp,proto)
    receipts=[]
    for row in manifest['rows']:
      pid=row['person_id'];receipt=OUT/(pid+'-receipt.json')
      if receipt.exists():
        r=json.loads(receipt.read_text());assert r['protocol_sha256']==sha(pp) and sha(r['output'])==r['output_sha256'];receipts.append(r);continue
      assert sha(row['file'])==row['sha256'];cm=byid[pid];N=cm['frames'];W=cm['width'];height=cm['height'];assert 27.9<=cm['fps']<=30.1;raw=np.full((N,33,5),np.nan);combined=raw.copy();provider=np.full(N,-1,int);full_counts=np.zeros(N,int);eligible_counts=np.zeros(N,int);times=np.zeros(N);tracker_ms=np.zeros(N,np.int64);crop_ambiguity=0;adjustments=0;prior=-1;start=time.perf_counter();cap=cv2.VideoCapture(row['file']);review=[]
      video_opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=2,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
      image_opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.IMAGE,num_poses=2,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
      x0,y0,x1,y1=round(.25*W),round(.08*height),round(.85*W),round(.50*height)
      with vision.PoseLandmarker.create_from_options(video_opts) as full,vision.PoseLandmarker.create_from_options(image_opts) as crop:
       for i in range(N):
        ok,bgr=cap.read();assert ok;times[i]=cap.get(cv2.CAP_PROP_POS_MSEC);ts=max(int(round(times[i])),prior+1);tracker_ms[i]=ts;adjustments+=int(ts!=int(round(times[i])));prior=ts;result=full.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),ts);full_counts[i]=len(result.pose_landmarks);good=eligible(result.pose_landmarks);eligible_counts[i]=len(good)
        if len(good)==1:raw[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in result.pose_landmarks[good[0]]];combined[i]=raw[i];provider[i]=0
        elif len(good)>1:provider[i]=-2
        else:
            cr=crop.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)));cg=[]
            for j,p in enumerate(cr.pose_landmarks):
                px=(x0+(p[23].x+p[24].x)/2*(x1-x0))/W
                if .35<=px<=.70 and all(min(p[k].visibility,p[k].presence)>=.5 and .02<=p[k].x<=.98 and .02<=p[k].y<=.98 for k in [23,24,25,26,27,28]):cg.append(j)
            if len(cg)==1:combined[i]=[[(x0+p.x*(x1-x0))/W,(y0+p.y*(y1-y0))/height,p.z*(x1-x0)/W,p.visibility,p.presence] for p in cr.pose_landmarks[cg[0]]];provider[i]=1
            elif len(cg)>1:crop_ambiguity+=1
        if i in [int(N*.2),int(N*.5),int(N*.8)]:
            small=cv2.resize(bgr,(240,round(height*240/W)))
            if provider[i]>=0:
                for k in [11,12,23,24,25,26,27,28,29,30,31,32]:
                    p=combined[i,k];cv2.circle(small,(round(p[0]*small.shape[1]),round(p[1]*small.shape[0])),2,(255,0,255) if min(p[3],p[4])>=.5 else(0,0,255),-1)
            cv2.putText(small,f'{pid} {i} p{provider[i]}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1);review.append(small)
        if(i+1)%600==0:print(pid,i+1,'selected',int(sum(provider[:i+1]>=0)),round(time.perf_counter()-start,1),flush=True)
      assert not cap.read()[0];cap.release();valid=(combined[:,:,3]>=.5)&(combined[:,:,4]>=.5)&np.isfinite(combined[:,:,:3]).all(axis=-1);full_valid=(raw[:,:,3]>=.5)&(raw[:,:,4]>=.5)&np.isfinite(raw[:,:,:3]).all(axis=-1);assert np.array_equal(raw[provider==0],combined[provider==0],equal_nan=True);path=OUT/(pid+'.npz');np.savez_compressed(path,landmarks=combined,full_landmarks=raw,valid=valid,full_valid=full_valid,provider=provider,source_ms=times,tracker_ms=tracker_ms,full_pose_counts=full_counts,eligible_counts=eligible_counts);cv2.imwrite(str(H/(pid+'-batch-review.jpg')),cv2.hconcat(review));wf=quality(full_valid,times);wc=quality(valid,times);r=dict(person_id=pid,status='extraction_complete_pending_target_gait_and_physical_clock_review',frames=N,fps=cm['fps'],full_target_frames=int(sum(provider==0)),crop_recovered=int(sum(provider==1)),ambiguous_frames=int(sum(provider==-2)),remaining_missing=int(sum(provider==-1)),crop_ambiguities=crop_ambiguity,operational_windows_full=sum(q['operationally_usable'] for q in wf),operational_windows_combined=sum(q['operationally_usable'] for q in wc),windows_full=wf,windows_combined=wc,source_timestamps_strictly_monotone=bool(np.all(np.diff(times)>0)),tracker_timestamp_adjustments=adjustments,original_physical_capture_clock_verified=False,output=str(path),output_sha256=sha(path),source_sha256=row['sha256'],protocol_sha256=sha(pp),seconds=time.perf_counter()-start);save(receipt,r);receipts.append(r);save(H/'batch14-progress.json',dict(status='running',completed=len(receipts),expected=14,last=pid));print('completed',pid,flush=True)
    save(H/'batch14-receipt.json',dict(status='all14extracted_pending_identity_and_walking_review',people=14,protocol_sha256=sha(pp),receipts=receipts,clinical_model_fitted=False));print('All14actualMediaPipe extractions completed',flush=True)
if __name__=='__main__':main()
