"""Python3.7, official local OpenPose PAF decoder over hash-verified GPU maps."""
import os,sys,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np,cv2
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101')
OP=ROOT/'tmp/openpose-cpu/openpose';sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ConnectTimeout=15','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf8')
def command(args,optional=False):
    p=subprocess.run(args,capture_output=True,timeout=180)
    if p.returncode:
        if optional:return None
        raise RuntimeError(p.stderr.decode('utf8',errors='replace'))
    return p.stdout
def main():
    sub=json.loads((HERE/'submission.json').read_text());remote=sub['remote'];out=BIG/'body25';out.mkdir(exist_ok=True);cache=BIG/'received';cache.mkdir(exist_ok=True)
    receipt=cache/'receipt.json';done=json.loads(receipt.read_text()) if receipt.exists() else {}
    w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0});w.start();cv2.setNumThreads(2);started=time.time()
    while time.time()-started<1800:
        indexraw=command(SSH+['cat '+remote+'/full/index.json'],True)
        if indexraw is None:time.sleep(5);continue
        index=json.loads(indexraw);save(cache/'index.json',index)
        for row in index:
            if row['file'] in done:assert done[row['file']]['sha256']==row['sha256'];continue
            local=cache/row['file']
            if not local.exists():command(SCP+[HOST+':'+remote+'/full/heatmaps/'+row['file'],str(local)])
            assert sha(local)==row['sha256'];folder=out/row['record_id'];folder.mkdir(exist_ok=True);hashes={}
            image=np.zeros((row['height'],row['width'],3),np.uint8)
            with np.load(str(local),allow_pickle=False) as z:
                assert z['decoded_frames'].tolist()==row['frames']
                for frame,hm in zip(row['frames'],z['heatmaps']):
                    d=op.Datum();d.cvInputData=image;d.poseNetOutput=np.ascontiguousarray(hm);w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
                    people=[] if kp is None else [{'pose_keypoints_2d':a.flatten().tolist()} for a in kp]
                    dest=folder/('%s_%012d_keypoints.json'%(row['record_id'],frame));save(dest,dict(version=1.3,people=people));hashes[dest.name]=sha(dest)
            done[row['file']]=dict(sha256=row['sha256'],frames=row['frames'],json_sha256=hashes);save(receipt,done)
            print('DECODED',sum(len(r['frames']) for r in done.values()),'seconds',round(time.time()-started,1),flush=True)
        complete=command(SSH+['cat '+remote+'/full/complete.json'],True)
        if complete is not None:
            c=json.loads(complete)
            if len(done)==c['chunks']:
                assert sum(len(r['frames']) for r in done.values())==sub['expected_frames']==c['frames']
                save(HERE/'pose-complete.json',dict(remote=c,chunks=len(done),frames=c['frames'],receipt_sha256=sha(receipt),seconds=time.time()-started))
                save(HERE/'gpu-equivalence.json',json.loads(command(SSH+['cat '+remote+'/full/equivalence.json'])))
                (cache/'slurm.log').write_bytes(command(SSH+['cat '+remote+'/logs/worker-'+sub['job_id']+'.log']))
                print('COMPLETE',flush=True);break
        time.sleep(3)
    else:raise RuntimeError('Time limit: resume receipt')
    w.stop()
if __name__=='__main__':main()
