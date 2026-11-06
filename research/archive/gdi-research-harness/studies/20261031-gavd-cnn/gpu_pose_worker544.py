"""Weight-preserving BODY25 forward pass; official local OpenPose decodes results."""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np,torch
from pose_torch import Body25
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def blob(im):
    h,w=im.shape[:2];nh=544;nw=16*int(nh*w/h/16+.5)
    scale=min((nw-1)/(w-1),(nh-1)/(h-1))
    # Official OpenPose uses cubic when enlarging and area when shrinking.
    im=cv2.warpAffine(im,np.array([[scale,0,0],[0,scale,0]],dtype=np.float64),(nw,nh),flags=cv2.INTER_CUBIC if scale>1 else cv2.INTER_AREA)
    return np.ascontiguousarray((im.astype(np.float32)/256-.5).transpose(2,0,1)[None])
def main(args):
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    net=Body25(HERE/'network').cuda().eval();checks=[]
    with torch.inference_mode():
        for path in sorted((HERE/'official_probe544').glob('*.npz')):
            z=np.load(path);x=blob(z['image']);assert np.array_equal(x,z['blob']),'Image blob changed'
            hm=net(torch.from_numpy(x).cuda()).cpu().numpy()[0]
            err=float(np.max(np.abs(hm-z['heatmaps'])));assert err<1e-4,(path.name,err)
            checks.append(dict(file=path.name,max_error=err,input_sha256=sha(path)))
    assert len(checks)==3,'Three actual-aspect-ratio checks required'
    out=HERE/(args.run or args.mode);out.mkdir(exist_ok=True);(out/'heatmaps').mkdir(exist_ok=True)
    save(out/'equivalence.json',checks)
    rows=json.loads((HERE/args.manifest).read_text())['records'];index=[];done=[];started=time.time()
    old=json.loads((out/'index.json').read_text()) if (out/'index.json').exists() else []
    oldmap={r['file']:r for r in old}
    for row in rows:
        path=HERE/row['video_file'];assert sha(path)==row['video_sha256']
        cap=cv2.VideoCapture(str(path));first=row['decoded_first'];last=row['decoded_last']
        frames=list(range(first,last+1))
        if args.mode=='pilot':frames=sorted(set([first+(last-first)//4,first+(last-first)//2,first+3*(last-first)//4]))
        for start in range(0,len(frames),25):
            ids=frames[start:start+25];name='%s_%06d_%06d.npz'%(row['record_id'],ids[0],ids[-1]);dest=out/'heatmaps'/name
            if name in oldmap:
                assert sha(dest)==oldmap[name]['sha256'];index.append(oldmap[name]);continue
            maps=[];cap.set(cv2.CAP_PROP_POS_FRAMES,ids[0])
            with torch.inference_mode():
                for k,f in enumerate(ids):
                    if k and f!=ids[k-1]+1:cap.set(cv2.CAP_PROP_POS_FRAMES,f)
                    ok,im=cap.read();assert ok,(path.name,f)
                    hm=net(torch.from_numpy(blob(im)).cuda()).cpu().numpy()[0];assert np.isfinite(hm).all();maps.append(hm)
            temp=dest.with_suffix('.tmp')
            with temp.open('wb') as s:np.savez_compressed(s,heatmaps=np.stack(maps),decoded_frames=np.asarray(ids))
            temp.replace(dest)
            item=dict(file=name,record_id=row['record_id'],video_id=row['video_id'],frames=ids,
                      annotation_offset=row['annotation_offset'],width=row['width'],height=row['height'],sha256=sha(dest),bytes=dest.stat().st_size)
            index.append(item);save(out/'index.json',index)
            save(out/'progress.json',dict(record=row['record_id'],last_frame=ids[-1],completed_records=len(done),chunks=len(index),seconds=time.time()-started))
        cap.release();done.append(row['record_id']);print(args.mode,len(done),len(rows),row['record_id'],flush=True)
    save(out/'complete.json',dict(records=done,frames=sum(len(r['frames']) for r in index),chunks=len(index),seconds=time.time()-started,
                                manifest_sha256=sha(HERE/args.manifest),code_sha256=sha(Path(__file__))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','full'],required=True);p.add_argument('--manifest',default='normal_jobs.json');p.add_argument('--run');main(p.parse_args())
