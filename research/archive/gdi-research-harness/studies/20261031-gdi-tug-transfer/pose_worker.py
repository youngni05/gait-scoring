"""Slurm GPU forward pass; lossless outputs decoded by official OpenPose locally."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from pose_torch import Body25
HERE=Path(__file__).resolve().parent
NETWORK=Path('/abr/RESEARCH_USER/gdi-healthy-child-pose-20261031/pose_probe')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    hashes=json.loads((HERE/'package-hashes.json').read_text())
    for name,digest in hashes['files'].items():assert sha(HERE/name)==digest,name
    for name,digest in hashes['network'].items():assert sha(NETWORK/name)==digest,name
    net=Body25(NETWORK).cuda().eval()
    # Independent original Caffe forward reference from the prior verified export.
    probe=np.load(NETWORK/'native_arrays.npz')
    with torch.inference_mode():reference=net(torch.from_numpy(probe['blob']).cuda()).cpu().numpy()[0]
    error=float(np.abs(reference-probe['heatmaps']).max());assert error<1e-4
    rows=json.loads((HERE/'input-manifest.json').read_text());index=[];started=time.time()
    (HERE/'heatmaps').mkdir(exist_ok=True)
    for row in sorted(rows,key=lambda r:(r['person']!='6',int(r['person']))):
        with np.load(HERE/'inputs'/row['file']) as z:blobs=z['blobs']
        assert blobs.shape[0]==124
        for first in range(0,124,31):
            out=[]
            with torch.inference_mode():
                for b in blobs[first:first+31]:out.append(net(torch.from_numpy(b[None]).cuda()).cpu().numpy()[0])
            maps=np.stack(out);assert np.isfinite(maps).all()
            name='OAW%02d_%03d.npz'%(int(row['person']),first);dest=HERE/'heatmaps'/name
            np.savez_compressed(dest,heatmaps=maps)
            index.append({'file':name,'person':row['person'],'first_offset':first,'count':len(maps),'sha256':sha(dest),'bytes':dest.stat().st_size})
            save(HERE/'heatmaps/index.json',index)
            save(HERE/'progress.json',{'chunks':len(index),'frames':sum(r['count'] for r in index),'seconds':time.time()-started})
            print(row['person'],first,'seconds',round(time.time()-started),flush=True)
    save(HERE/'complete.json',{'job':os.environ['SLURM_JOB_ID'],'people':12,'frames':1488,'chunks':48,'probe_max_error':error,
                              'input_manifest_sha256':sha(HERE/'input-manifest.json'),'seconds':time.time()-started,
                              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                              'index_sha256':sha(HERE/'heatmaps/index.json')})
    print('GDI_TUG_BODY25_FORWARD_COMPLETE',flush=True)
if __name__=='__main__':main()
