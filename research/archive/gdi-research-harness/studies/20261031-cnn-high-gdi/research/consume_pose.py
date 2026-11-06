"""Receive lossless chunks, verify hashes, decode with official OpenPose 1.7.0.

Only the temporary received heatmap chunk is removed after verified JSON writes.
The source chunks remain on the school server. No source videos are deleted.
"""
from pathlib import Path
import json,hashlib,sys,os,time,subprocess,argparse
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'))
os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
REMOTE='/abr/RESEARCH_USER/gdi-healthy-child-pose-20261031'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(remote,dest):
    subprocess.run(SCP+[HOST+':'+REMOTE+'/'+remote,str(dest)],check=True,timeout=180)
def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE/'body25');ap.add_argument('--cache',type=Path,default=HERE/'heatmap_cache');args=ap.parse_args()
    cv2.setNumThreads(2)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    cache=args.cache;cache.mkdir(parents=True,exist_ok=True)
    donepath=out/'chunks_verified.json';done=json.loads(donepath.read_text()) if donepath.exists() else {}
    p={'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0}
    w=op.WrapperPython();w.configure(p);w.start()
    im=np.zeros((720,1280,3),dtype=np.uint8)
    video_meta={r['file']:r for r in json.loads((HERE/'video_metadata.json').read_text())}
    started=time.time()
    while time.time()-started<5*3600:
        response=subprocess.run(SSH+['cat '+REMOTE+'/heatmaps/index.json'],capture_output=True,timeout=40)
        if response.returncode:
            print('index not ready',flush=True);time.sleep(20);continue
        index=json.loads(response.stdout)
        save(out/'remote_chunk_index.json',index)
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
        response=subprocess.run(SSH+['cat '+REMOTE+'/complete.json'],capture_output=True,timeout=40)
        if response.returncode==0:
            complete=json.loads(response.stdout)
            if len(done)==complete['chunks']:
                assert sum(v['frames'] for v in done.values())==complete['frames']
                save(out/'complete.json',dict(remote=complete,decoded_frames=sum(v['frames'] for v in done.values()),seconds=time.time()-started))
                print('OFFICIAL_BODY25_DECODE_COMPLETE',flush=True);break
        time.sleep(10)
    else:raise RuntimeError('Consumer time limit; resume reuses verified chunks')
    w.stop()
if __name__=='__main__':main()
