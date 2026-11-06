"""Serialize the exact official OpenPose image preprocessing, not a new video."""
from pathlib import Path
import json,hashlib,shutil
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blob(im):
    h,w=im.shape[:2];nh=368;nw=16*int(nh*w/h/16+.5)
    scale=min((nw-1)/(w-1),(nh-1)/(h-1))
    resized=cv2.warpAffine(im,np.array([[scale,0,0],[0,scale,0]],dtype=np.float64),(nw,nh),flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32)/256-.5).transpose(2,0,1))

def main():
    cv2.setNumThreads(2)
    intervals=json.loads((HERE/'intervals.json').read_text());rows=json.loads((OUT/'acquisition.json').read_text())['videos']
    folder=OUT/'inputs';folder.mkdir(exist_ok=True);manifest=[]
    for row in rows:
        dest=folder/('OAW%02d.npz'%int(row['person']));receipt=dest.with_suffix('.json')
        if dest.exists() and receipt.exists():
            old=json.loads(receipt.read_text());assert old['intervals_sha256']==sha(HERE/'intervals.json') and old['sha256']==sha(dest)
            manifest.append(old);continue
        p=Path(row['path']);assert sha(p)==row['sha256']
        cap=cv2.VideoCapture(str(p));start=intervals['start_frames'][row['person']];cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        assert cap.get(cv2.CAP_PROP_POS_FRAMES)==start
        frames=[];times=[];review=[]
        for offset in range(124):
            ok,im=cap.read();assert ok
            frames.append(blob(im));times.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)
            if offset in (0,31,62,93,123):
                small=cv2.resize(im,(216,384));cv2.putText(small,str(start+offset),(2,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);review.append(small)
        fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
        np.savez_compressed(str(dest),blobs=np.stack(frames),source_frame=np.arange(start,start+124),timestamps=times)
        cv2.imwrite(str(OUT/('interval-%02d.jpg'%int(row['person']))),np.hstack(review))
        r={'person':row['person'],'file':dest.name,'sha256':sha(dest),'source_sha256':row['sha256'],'first':start,'frames':124,'height':im.shape[0],'width':im.shape[1],
           'fps_metadata':fps,'timestamp_first':times[0],'timestamp_last':times[-1],'timestamp_max_gap':float(np.diff(times).max()),'intervals_sha256':sha(HERE/'intervals.json')}
        assert np.all(np.diff(times)>0),'Non-monotonic source timestamps'
        receipt.write_text(json.dumps(r,indent=2));manifest.append(r);print('prepared',row['person'],dest.stat().st_size,flush=True)
    (OUT/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copyfile(str(HERE/'intervals.json'),str(OUT/'intervals.json'))
    print('BLOBS_COMPLETE',sum(r['frames'] for r in manifest),flush=True)

if __name__=='__main__':main()
