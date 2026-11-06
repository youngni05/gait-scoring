"""Receive SHA-verified GPU heatmaps on D and run official BODY25 decoding."""
from pathlib import Path
import argparse,json,hashlib,sys,os,time,subprocess
import numpy as np
import cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'))
os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
REMOTE='/abr/RESEARCH_USER/gavd-cnn-20261031'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST'
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def main(args):
    cv2.setNumThreads(2)
    out=DATA/(args.run+'_body25');out.mkdir(parents=True,exist_ok=True)
    cache=DATA/'heatmap_cache'/args.run;cache.mkdir(parents=True,exist_ok=True)
    donepath=out/'chunks_verified.json';done=json.loads(donepath.read_text()) if donepath.exists() else {}
    w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x544','render_pose':0});w.start()
    started=time.time()
    while time.time()-started<args.hours*3600:
        response=subprocess.run(SSH+['cat '+REMOTE+'/'+args.run+'/index.json'],capture_output=True,timeout=40)
        if response.returncode:
            if args.once:break
            time.sleep(20);continue
        index=json.loads(response.stdout);save(out/'remote_chunk_index.json',index)
        for row in index:
            if row['file'] in done:
                assert done[row['file']]['source_sha256']==row['sha256'];continue
            tmp=cache/row['file']
            subprocess.run(SCP+[HOST+':'+REMOTE+'/'+args.run+'/heatmaps/'+row['file'],str(tmp)],check=True,timeout=180)
            assert sha(tmp)==row['sha256'],'Heatmap transfer changed'
            im=np.zeros((row['height'],row['width'],3),dtype=np.uint8)
            folder=out/row['record_id'];folder.mkdir(exist_ok=True);hashes={}
            with np.load(str(tmp),allow_pickle=False) as z:
                assert z['decoded_frames'].tolist()==row['frames']
                assert len(z['heatmaps'])==len(row['frames'])
                for frame,heatmap in zip(row['frames'],z['heatmaps']):
                    d=op.Datum();d.cvInputData=im;d.poseNetOutput=np.ascontiguousarray(heatmap)
                    w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
                    people=[] if kp is None else [{'person_id':[-1],'pose_keypoints_2d':a.flatten().tolist()} for a in kp]
                    annotation=frame-row['annotation_offset']
                    name='%s_%012d_keypoints.json'%(row['record_id'],annotation)
                    dest=folder/name;save(dest,{'version':1.3,'people':people});hashes[name]=sha(dest)
            done[row['file']]={'source_sha256':row['sha256'],'record_id':row['record_id'],'frames':len(hashes),'json_sha256':hashes}
            save(donepath,done)
            assert tmp.resolve().parent==cache.resolve();tmp.unlink()
            print(row['record_id'],'chunks',len(done),'seconds',round(time.time()-started),flush=True)
        response=subprocess.run(SSH+['cat '+REMOTE+'/'+args.run+'/complete.json'],capture_output=True,timeout=40)
        if response.returncode==0:
            complete=json.loads(response.stdout)
            if len(done)==complete['chunks']:
                assert sum(v['frames'] for v in done.values())==complete['frames']
                save(out/'complete.json',dict(remote=complete,decoded_frames=sum(v['frames'] for v in done.values()),seconds=time.time()-started))
                print('OFFICIAL_BODY25_DECODE_COMPLETE',flush=True);break
        if args.once:break
        time.sleep(10)
    else:raise RuntimeError('Consumer time limit; resume reuses verified chunks')
    w.stop()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='pilot');p.add_argument('--hours',type=float,default=5);p.add_argument('--once',action='store_true');main(p.parse_args())
