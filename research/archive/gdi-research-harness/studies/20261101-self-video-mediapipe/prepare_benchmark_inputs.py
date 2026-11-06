"""Pin the OpenPose resize to its verified OpenCV 4.8.1 path (Python 3.7)."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BIG=Path('D:/gait-research-data/D_P/self-video-mediapipe-20261101')

def blob(im):
    h,w=im.shape[:2];nh=368;nw=16*int(nh*w/h/16+.5)
    scale=min((nw-1)/(w-1),(nh-1)/(h-1))
    im=cv2.warpAffine(im,np.array([[scale,0,0],[0,scale,0]],dtype=np.float64),(nw,nh),flags=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
    return np.ascontiguousarray((im.astype(np.float32)/256-.5).transpose(2,0,1)[None])

def main():
    cv2.setNumThreads(2);checks=[]
    for path in sorted(Path('D:/gait-research-data/D_P/datasets/GAVD/official_probe').glob('*.npz')):
        z=np.load(str(path));x=blob(z['image']);error=float(np.abs(x-z['blob']).max());assert error==0
        checks.append(dict(file=path.name,blob_max_error=error))
    rows=[]
    for row in json.loads((HERE.parent/'20261101-self-video-gdi/inventory.json').read_text(encoding='utf8')):
        cap=cv2.VideoCapture(row['file']);blobs=[];times=[];hashes=[]
        # Full-video hashes can verify that decode pixels are identical between versions.
        for i in range(row['frames']):
            ok,im=cap.read();assert ok;hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
            if i<25:
                t=time.perf_counter();x=blob(im);times.append(time.perf_counter()-t);blobs.append(x)
        cap.release()
        np.savez_compressed(str(BIG/(row['id']+'-benchmark-inputs.npz')),blobs=np.stack(blobs))
        rows.append(dict(clip=row['id'],all_frame_pixel_sha256=hashes,resize_seconds=times))
    (HERE/'benchmark-inputs.json').write_text(json.dumps(dict(opencv=cv2.__version__,reference_checks=checks,rows=rows),indent=2),encoding='utf8')
    print('PINNED',cv2.__version__,sum(len(r['all_frame_pixel_sha256']) for r in rows),'frame hashes',flush=True)

if __name__=='__main__':main()
