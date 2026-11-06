"""Same extraction steps as Full, with the pinned Lite task."""
import time
import sys
from common import *
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def options(task=TASK):
    return vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task),delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,
        min_tracking_confidence=.5,output_segmentation_masks=False)

def main():
    cv2.setNumThreads(2);assert sha(TASK)==read(HERE/'model-receipt.json')['sha256'];runs=[]
    for row in read(OLD/'inventory.json'):
        assert sha(row['file'])==row['sha256']
        n=row['frames'];lm=np.full((n,33,5),np.nan);counts=np.zeros(n,dtype=np.int32)
        times=np.zeros(n);source_ms=np.zeros(n);task_ms=np.zeros(n,dtype=np.int64)
        cap=cv2.VideoCapture(row['file']);start=time.perf_counter()
        with vision.PoseLandmarker.create_from_options(options()) as detector:
            init=time.perf_counter()-start;start=time.perf_counter();prior=-1;adjustments=[]
            for i in range(n):
                ok,bgr=cap.read();assert ok and bgr.shape[:2]==(row['height'],row['width'])
                source_ms[i]=cap.get(cv2.CAP_PROP_POS_MSEC);rounded=round(source_ms[i]);ts=max(rounded,prior+1)
                if ts!=rounded:adjustments.append(dict(frame=i,source_ms=float(source_ms[i]),rounded_ms=rounded,task_ms=ts))
                task_ms[i]=ts;prior=ts
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                t=time.perf_counter();result=detector.detect_for_video(image,ts);times[i]=time.perf_counter()-t
                counts[i]=len(result.pose_landmarks);assert counts[i] in (0,1)
                if counts[i]:lm[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in result.pose_landmarks[0]]
                if (i+1)%300==0:print(row['id'],i+1,round(time.perf_counter()-start,2),'s',flush=True)
            assert not cap.read()[0];loop=time.perf_counter()-start
        cap.release();dest=BIG/(row['id']+'-pose.npz')
        np.savez_compressed(dest,landmarks=lm,raw=adapt(lm,row['width'],row['height']),counts=counts,source_ms=source_ms,task_ms=task_ms,inference_seconds=times)
        item=dict(clip=row['id'],frames=n,detected=int((counts==1).sum()),missing=int((counts==0).sum()),init_seconds=init,
            loop_seconds=loop,inference_seconds=float(times.sum()),inference_fps=n/float(times.sum()),loop_fps=n/loop,
            timestamp_adjustments=adjustments,output_sha256=sha(dest),video_sha256=sha(row['file']))
        runs.append(item);print(item,flush=True)
        save(HERE/'extraction.json',dict(versions=dict(python=sys.version,mediapipe=mp.__version__,opencv=cv2.__version__,numpy=np.__version__),
            task_sha256=sha(TASK),task_bytes=TASK.stat().st_size,protocol_sha256=sha(HERE/'protocol.json'),runs=runs))

if __name__=='__main__':main()
