"""GPU forward pass only; lossless float32 heatmaps for official local decoding."""
from pathlib import Path
import json,hashlib,time,os,zipfile
import numpy as np
import cv2,torch
from pose_torch import Body25,probe

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def blob(im):
    h,w=im.shape[:2];nh=368;nw=16*int(nh*w/h/16+.5)
    scale=min((nw-1)/(w-1),(nh-1)/(h-1))
    mat=np.array([[scale,0,0],[0,scale,0]],dtype=np.float64)
    resized=cv2.warpAffine(im,mat,(nw,nh),flags=cv2.INTER_LINEAR)
    return np.ascontiguousarray((resized.astype(np.float32)/256-.5).transpose(2,0,1)[None])
def main():
    started=time.time();torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    probe(HERE/'pose_probe','cuda')
    im=cv2.imread(str(HERE/'pose_probe/input/frame.png'))
    original=np.load(HERE/'pose_probe/native_arrays.npz')['blob']
    assert np.array_equal(blob(im),original),'Image preprocessing mismatch'
    net=Body25(HERE/'pose_probe').cuda().eval()
    rows=json.loads((HERE/'td_download.json').read_text())['videos']
    metadata={r['file']:r for r in json.loads((HERE/'video_metadata.json').read_text())}
    complete=[];index=[]
    for row in sorted(rows,key=lambda r:r['file']):
        path=HERE/'td_sagittal'/row['file'];assert sha(path)==row['sha256']
        stem=path.stem;cap=cv2.VideoCapture(str(path));chunk=[];frame=0
        with torch.inference_mode():
            while True:
                ok,im=cap.read()
                if not ok:break
                y=net(torch.from_numpy(blob(im)).cuda()).cpu().numpy()[0]
                assert np.isfinite(y).all()
                chunk.append(y);frame+=1
                if len(chunk)==50 or frame==metadata[path.name]['frames']:
                    start=frame-len(chunk);name='%s_%06d_%06d.npz'%(stem,start,frame-1)
                    dest=HERE/'heatmaps'/name;tmp=dest.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,heatmaps=np.stack(chunk))
                    tmp.replace(dest)
                    item=dict(file=name,video=path.name,first=start,last=frame-1,sha256=sha(dest),bytes=dest.stat().st_size)
                    index.append(item);save(HERE/'heatmaps/index.json',index)
                    chunk=[]
                    save(HERE/'progress.json',dict(phase='extracting',video=path.name,frame=frame,completed_videos=len(complete),elapsed=time.time()-started))
                    print(path.name,frame,'elapsed',round(time.time()-started),flush=True)
        cap.release();assert frame==metadata[path.name]['frames'],(path.name,frame)
        assert not chunk
        complete.append(dict(video=path.name,frames=frame))
    save(HERE/'complete.json',dict(videos=complete,frames=sum(r['frames'] for r in complete),chunks=len(index),seconds=time.time()-started))
    print('HEALTHY_CHILD_HEATMAPS_COMPLETE',flush=True)
if __name__=='__main__':main()
