"""Four interleaved passes per variant on identical predecoded50 frames."""
import time
import hashlib
import statistics
from common import *
from extract import options
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

def main():
    cv2.setNumThreads(2);inputs=[]
    assert sha(TASK)==read(HERE/'model-receipt.json')['sha256']
    assert sha(FULL_TASK)==read(FULL/'extraction.json')['task_sha256']
    for row in read(OLD/'inventory.json'):
        cap=cv2.VideoCapture(row['file']);frames=[];times=[];hashes=[]
        for i in range(25):
            ok,bgr=cap.read();assert ok;frames.append(bgr);times.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)));hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
        cap.release();inputs.append((row['id'],frames,times,hashes))
    runs=[]
    for repeat in range(4):
        order=['Full','Lite'] if repeat%2==0 else ['Lite','Full']
        for variant in order:
            task=FULL_TASK if variant=='Full' else TASK;steps=[];engine=[];counts=[]
            for clip,frames,timestamps,hashes in inputs:
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB))
                with vision.PoseLandmarker.create_from_options(options(task)) as warm:warm.detect_for_video(image,0)
                with vision.PoseLandmarker.create_from_options(options(task)) as detector:
                    for frame,ts in zip(frames,timestamps):
                        t=time.perf_counter();image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                        t0=time.perf_counter();result=detector.detect_for_video(image,ts);engine.append(time.perf_counter()-t0);steps.append(time.perf_counter()-t)
                        counts.append(len(result.pose_landmarks))
            r=dict(repeat=repeat,variant=variant,frames=50,step_seconds=sum(steps),engine_seconds=sum(engine),per_frame_seconds=steps,
                detected_frames=sum(n==1 for n in counts),missing_frames=sum(n==0 for n in counts))
            runs.append(r);print(variant,repeat,round(r['step_seconds'],4),flush=True)
    summary={}
    for variant in ['Full','Lite']:
        vals=[r['step_seconds'] for r in runs if r['variant']==variant]
        summary[variant]=dict(median_seconds=statistics.median(vals),minimum_seconds=min(vals),maximum_seconds=max(vals),fps=50/statistics.median(vals),passes=len(vals))
    save(HERE/'benchmark.json',dict(frames_per_pass=50,order='Full,Lite / Lite,Full / Full,Lite / Lite,Full',runs=runs,summary=summary,
        speed_ratio_full_over_lite=summary['Full']['median_seconds']/summary['Lite']['median_seconds'],
        input_frame_hashes={clip:hashes for clip,_,_,hashes in inputs},task_hashes={'Full':sha(FULL_TASK),'Lite':sha(TASK)},
        scope='same CPU and MediaPipe settings; RGB conversion + final landmarks; exclude load,warmup,video decode,disk IO. Fresh tracking state each clip/pass; native thread default',
        device='Intel Core Ultra 7 258V CPU; no smartphone test'))

if __name__=='__main__':main()
