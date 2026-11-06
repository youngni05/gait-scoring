"""Local BODY25 forward, native frames. Reuses the verified Caffe-weight adapter."""
import os,sys,json,time,argparse,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'outputs/motionbert-smoke-20261015/deps'),str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'),str(HERE.parent/'20261031-gavd-cnn')]
import numpy as np,cv2,torch
from pose_torch import Body25
from gpu_pose_worker import blob
BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def main(limit):
    torch.set_num_threads(4);cv2.setNumThreads(2)
    network=HERE.parent/'20261031-cnn-high-gdi/research/pose_probe'
    net=Body25(network).eval()
    probe=np.load(network/'native_arrays.npz')
    t=time.time()
    with torch.inference_mode():test=net(torch.from_numpy(probe['blob'])).numpy()[0]
    error=float(np.abs(test-probe['heatmaps']).max());assert error<1e-4,error
    save(HERE/'pose-equivalence.json',dict(max_heatmap_error=error,tolerance=1e-4,seconds=time.time()-t,device='cpu',weights_sha256=sha(network/'network_weights.npz')))
    print('equivalence',error,'seconds',time.time()-t,flush=True)
    BIG.mkdir(parents=True,exist_ok=True);(BIG/'heatmaps').mkdir(exist_ok=True)
    index_path=BIG/'index.json';index=json.loads(index_path.read_text()) if index_path.exists() else []
    old={r['file']:r for r in index};started=time.time();count=0
    for row in json.loads((HERE/'inventory.json').read_text(encoding='utf8')):
        path=Path(row['file']);assert sha(path)==row['sha256'];cap=cv2.VideoCapture(str(path))
        total=row['frames'] if limit is None else min(row['frames'],limit)
        for start in range(0,total,25):
            end=min(start+25,total);name='%s_%06d_%06d.npz'%(row['id'],start,end-1)
            dest=BIG/'heatmaps'/name
            if name in old:assert sha(dest)==old[name]['sha256'];continue
            cap.set(1,start);maps=[]
            with torch.inference_mode():
                for f in range(start,end):
                    ok,im=cap.read();assert ok
                    maps.append(net(torch.from_numpy(blob(im))).numpy()[0])
            np.savez_compressed(dest,heatmaps=np.stack(maps),frames=np.arange(start,end))
            item=dict(file=name,clip=row['id'],first=start,last=end-1,width=row['width'],height=row['height'],sha256=sha(dest))
            index.append(item);save(index_path,index);count+=end-start
            progress=dict(clip=row['id'],last=end-1,frames_this_run=count,seconds=time.time()-started)
            save(HERE/'pose-progress.json',progress);print(json.dumps(progress),flush=True)
        cap.release()
    save(HERE/'pose-forward-complete.json',dict(frames=sum(r['last']-r['first']+1 for r in index),chunks=len(index),limit=limit,index_sha256=sha(index_path),seconds=time.time()-started))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);main(p.parse_args().limit)
