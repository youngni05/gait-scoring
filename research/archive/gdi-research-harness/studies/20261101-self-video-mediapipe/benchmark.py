"""Sequential CPU benchmark on identical 50 frames, with honest stage boundaries."""
import argparse
import time
from common import *
import cv2

def main(engine):
    cv2.setNumThreads(2)
    if engine=='openpose':
        sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
        sys.path.insert(0,str(HERE.parent/'20261031-gavd-cnn'))
        import torch
        from pose_torch import Body25
        from gpu_pose_worker import blob
        torch.set_num_threads(4)
        network=HERE.parent/'20261031-cnn-high-gdi/research/pose_probe'
        net=Body25(network).eval()
        z=np.load(network/'native_arrays.npz')
        with torch.inference_mode():
            reference=net(torch.from_numpy(z['blob'])).numpy()[0]
        reference_error=float(np.max(np.abs(reference-z['heatmaps'])))
        assert reference_error<1e-4
    else:
        from extract import options
        import mediapipe as mp
        from mediapipe.tasks.python import vision
    alltimes=[];allstep=[];rows=[];totalwall=0
    for row in read(OLD/'inventory.json'):
        cap=cv2.VideoCapture(row['file']);frames=[];timestamps=[]
        for i in range(25):
            ok,frame=cap.read();assert ok;frames.append(frame)
            timestamps.append(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
        cap.release()
        if engine=='openpose':
            pin=next(r for r in read(HERE/'benchmark-inputs.json')['rows'] if r['clip']==row['id'])
            pinned_blobs=np.load(BIG/(row['id']+'-benchmark-inputs.npz'))['blobs']
            assert all(hashlib.sha256(frame.tobytes()).hexdigest()==pin['all_frame_pixel_sha256'][i] for i,frame in enumerate(frames))
            with torch.inference_mode():net(torch.from_numpy(pinned_blobs[0]))
        else:
            image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frames[0],cv2.COLOR_BGR2RGB))
            with vision.PoseLandmarker.create_from_options(options()) as warm: warm.detect_for_video(image,0)
            detector=vision.PoseLandmarker.create_from_options(options())
        times=[];steps=[];maps=[];counts=[];started=time.perf_counter()
        for i,frame in enumerate(frames):
            step_start=time.perf_counter()
            if engine=='openpose':
                x=torch.from_numpy(pinned_blobs[i]);start=time.perf_counter()
                with torch.inference_mode():y=net(x).numpy()[0]
                times.append(time.perf_counter()-start);maps.append(y)
            else:
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                start=time.perf_counter();result=detector.detect_for_video(image,timestamps[i]);times.append(time.perf_counter()-start)
                counts.append(len(result.pose_landmarks))
            steps.append(time.perf_counter()-step_start+(pin['resize_seconds'][i] if engine=='openpose' else 0))
            if (i+1)%10==0:print(engine,row['id'],i+1,flush=True)
        wall=time.perf_counter()-started;totalwall+=wall
        if engine=='mediapipe':detector.close()
        else:np.savez_compressed(BIG/(row['id']+'-benchmark-heatmaps.npz'),heatmaps=np.stack(maps))
        rows.append(dict(clip=row['id'],frames=25,engine_seconds=sum(times),step_seconds=sum(steps),loop_seconds=wall,
            detected_frames=sum(n==1 for n in counts) if counts else None))
        alltimes.extend(times);allstep.extend(steps)
    report=dict(engine=engine,device='CPU',frames=50,selection='first25 frames of each original video',
        warmup='OpenPose reference + first frame each clip; MediaPipe separate warm instance each clip, timed instance has fresh tracker',
        engine_seconds=sum(alltimes),engine_fps=50/sum(alltimes),step_seconds=sum(allstep),step_fps=50/sum(allstep),
        loop_seconds=totalwall,rows=rows,per_frame_engine_seconds=alltimes,
        excludes='model load, warmup, original video decode, disk saves; OpenPose PAF decoding timed separately',
        threads='torch4 and cv2=2; MediaPipe native XNNPACK thread count is runtime default, not forced equal',
        scope='OpenPose returns heatmaps; MediaPipe returns final landmarks with temporal tracking',
        openpose_resize='Pinned OpenCV4.8.1 blobs; separately measured resize added to step_seconds' if engine=='openpose' else None,
        reference_heatmap_max_error=reference_error if engine=='openpose' else None,
        video_hashes=[r['sha256'] for r in read(OLD/'inventory.json')])
    save(HERE/('benchmark-'+engine+'.json'),report)
    print(engine,'engine',report['engine_seconds'],'steps',report['step_seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('engine',choices=['openpose','mediapipe']);main(parser.parse_args().engine)
