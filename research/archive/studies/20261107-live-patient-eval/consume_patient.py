"""LIVE patient copy of consume_pose.py. Paths, time limit, shards and fetch retries differ; decoding does not.

Receive lossless chunks, verify hashes, decode with official OpenPose 1.7.0.

Only the temporary received heatmap chunk is removed after verified JSON writes.
The source chunks remain on the school server. No source videos are deleted.
"""
from pathlib import Path
import json,hashlib,sys,os,time,subprocess,argparse
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'))
os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
REMOTE='/abr/RESEARCH_USER/gdi-live-patient-pose-20261107'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
NOWINDOW=0x08000000
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(remote,dest):
    for attempt in range(5):
        try:
            subprocess.run(SCP+[HOST+':'+REMOTE+'/'+remote,str(dest)],check=True,timeout=180,creationflags=NOWINDOW);return
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            print('fetch retry',remote,attempt,e,flush=True);time.sleep(30)
    raise RuntimeError('Fetch failed '+remote)
def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=BIG/'body25');ap.add_argument('--cache',type=Path,default=BIG/'heatmap_cache');ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);args=ap.parse_args()
    cv2.setNumThreads(2)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    cache=args.cache/str(args.shard);cache.mkdir(parents=True,exist_ok=True)
    donepath=out/('chunks_verified-%d.json'%args.shard);old=out/'chunks_verified.json'
    done=json.loads((donepath if donepath.exists() else old).read_text())
    p={'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0}
    w=op.WrapperPython();w.configure(p);w.start()
    im=np.zeros((720,1280,3),dtype=np.uint8)
    video_meta={r['file']:r for r in json.loads((HERE/'video_metadata.json').read_text())}
    started=time.time()
    if True:
        response=subprocess.run(SSH+['cat '+REMOTE+'/heatmaps/index.json'],capture_output=True,timeout=40,check=True,creationflags=NOWINDOW)
        index=json.loads(response.stdout)
        names=sorted({r['video'] for r in index});mine={v for k,v in enumerate(names) if k%args.shards==args.shard}
        index=[r for r in index if r['video'] in mine]
        keep={r['file'] for r in index};done={k:v for k,v in done.items() if k in keep};save(donepath,done)
        for row in index:
            if row['file'] in done:
                assert done[row['file']]['source_sha256']==row['sha256'];continue
            tmp=cache/row['file'];fetch('heatmaps/'+row['file'],tmp)
            assert sha(tmp)==row['sha256'],'Heatmap transfer changed'
            meta=video_meta[row['video']];im=np.zeros((meta['height'],meta['width'],3),dtype=np.uint8)
            folder=out/Path(row['video']).stem;folder.mkdir(exist_ok=True)
            hashes={}
            with np.load(str(tmp),allow_pickle=False) as z:
                maps=z['heatmaps'];assert len(maps)==row['last']-row['first']+1
                for offset,heatmap in enumerate(maps):
                    d=op.Datum();d.cvInputData=im;d.poseNetOutput=np.ascontiguousarray(heatmap)
                    w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
                    people=[] if kp is None else [{'person_id':[-1],'pose_keypoints_2d':a.flatten().tolist()} for a in kp]
                    name='%s_%012d_keypoints.json'%(Path(row['video']).stem,row['first']+offset)
                    dest=folder/name;save(dest,{'version':1.3,'people':people});hashes[name]=sha(dest)
            done[row['file']]={'source_sha256':row['sha256'],'frames':len(hashes),'json_sha256':hashes}
            save(donepath,done)
            # This is a verified disposable receive cache, not the source data.
            assert tmp.resolve().parent==cache.resolve();tmp.unlink()
            print(row['video'],row['last'],'chunks',len(done),'seconds',round(time.time()-started),flush=True)
        assert keep==set(done)
        print('SHARD_COMPLETE',args.shard,len(done),sum(v['frames'] for v in done.values()),flush=True)
    w.stop()
if __name__=='__main__':main()
