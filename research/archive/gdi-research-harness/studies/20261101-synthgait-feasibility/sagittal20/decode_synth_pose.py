"""Python3.7: SHA-verified GPU maps -> official CPU PAF decoder -> native BODY25."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time
import cv2,numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20');REMOTE='/abr/RESEARCH_USER/synthgait-pose20-20261101'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
HOST='RESEARCH_USER@PRIVATE_HOST';SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST];SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
OP=ROOT/'tmp/openpose-cpu/openpose';sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf-8');tmp.replace(p)
def command(args,optional=False,timeout=60):
    p=subprocess.run(args,capture_output=True,timeout=timeout)
    if p.returncode:
        if optional:return None
        raise RuntimeError(p.stderr.decode('utf8',errors='replace'))
    return p.stdout
def main():
    cv2.setNumThreads(2);cache=DATA/'received_maps';cache.mkdir(exist_ok=True);out=DATA/'body25';out.mkdir(exist_ok=True)
    donefile=cache/'receipt.json';done=json.loads(donefile.read_text()) if donefile.exists() else {}
    w=op.WrapperPython();w.configure({'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0,'upsampling_ratio':0});w.start()
    started=time.time()
    while time.time()-started<1800:
        raw=command(SSH+['cat '+REMOTE+'/full/index.json'],optional=True)
        if raw is None:time.sleep(4);continue
        index=json.loads(raw)
        for row in index:
            if row['file'] in done:
                assert done[row['file']]['server_sha256']==row['sha256'];continue
            local=cache/row['file']
            if not local.exists() or sha(local)!=row['sha256']:
                command(SCP+[HOST+':'+REMOTE+'/full/heatmaps/'+row['file'],str(local)],timeout=180)
            assert sha(local)==row['sha256']
            folder=out/row['record_id'];folder.mkdir(exist_ok=True);hashes={}
            image=np.zeros((row['height'],row['width'],3),np.uint8)
            with np.load(str(local),allow_pickle=False) as z:
                assert z['decoded_frames'].tolist()==row['frames']
                for frame,hm in zip(row['frames'],z['heatmaps']):
                    heatmap=np.ascontiguousarray(hm)
                    d=op.Datum();d.cvInputData=image;d.poseNetOutput=heatmap
                    w.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
                    people=[] if kp is None else [{'person_id':[-1],'pose_keypoints_2d':a.flatten().tolist()} for a in kp]
                    dest=folder/('%s_%012d_keypoints.json'%(row['record_id'],frame));save(dest,{'version':1.3,'people':people});hashes[dest.name]=sha(dest)
            done[row['file']]={'server_sha256':row['sha256'],'local_sha256':sha(local),'record_id':row['record_id'],'frames':row['frames'],'json_sha256':hashes}
            save(donefile,done);print(json.dumps({'chunks':len(done),'frames':sum(len(v['frames']) for v in done.values()),'seconds':round(time.time()-started,1)}),flush=True)
        completed=command(SSH+['cat '+REMOTE+'/full/complete.json'],optional=True)
        if completed is not None:
            complete=json.loads(completed)
            if len(done)==complete['chunks']:
                assert complete['frames']==1231 and len(complete['records'])==20
                assert sum(len(v['frames']) for v in done.values())==1231
                save(cache/'server-complete.json',complete)
                equivalence=command(SSH+['cat '+REMOTE+'/full/equivalence.json']);checks=json.loads(equivalence)
                assert len(checks)==3 and max(r['max_error'] for r in checks)<1e-4
                save(HERE/'gpu-heatmap-equivalence.json',checks)
                manifest=json.loads((HERE/'download-manifest.json').read_text(encoding='utf-8'));reports=[]
                for r in manifest['records']:
                    folder=out/r['sample_id'];files=sorted(folder.glob('*_keypoints.json'));arrays=[];counts=[]
                    for i,p in enumerate(files):
                        assert int(p.stem.split('_')[-2])==i
                        people=json.loads(p.read_text())['people'];counts.append(len(people));arr=np.full((25,3),np.nan)
                        if len(people)==1:arr=np.asarray(people[0]['pose_keypoints_2d']).reshape(25,3)
                        arrays.append(arr)
                    raw=np.stack(arrays);np.savez_compressed(str(out/(r['sample_id']+'.npz')),raw=raw,frame=np.arange(len(raw)),time_seconds=np.arange(len(raw))/16,detected_people=np.asarray(counts))
                    reports.append({'sample_id':r['sample_id'],'frames':len(raw),'unique_person_frames':counts.count(1),'empty_frames':counts.count(0),'ambiguous_frames':sum(c>1 for c in counts),'raw_path':str(out/(r['sample_id']+'.npz'))})
                report={'complete':True,'records':reports,'frames':1231,'chunks':len(done),'server_local_sha256_matched':True,
                    'receipt':str(donefile),'heatmap_max_error':max(r['max_error'] for r in checks),'heatmap_tolerance':1e-4,
                    'model':'official BODY25 Caffe weights forwarded by verified PyTorch adapter; official OpenPose CPU PAF decoding',
                    'net_resolution':'-1x368','decoder_upsampling_ratio':0,'optimizer_steps':0,'seconds':time.time()-started}
                save(HERE/'pose-verification.json',report);print('BODY25_20_COMPLETE',flush=True);break
        time.sleep(4)
    else:raise RuntimeError('Decoder time limit; verified chunk receipt supports resume')
    w.stop()
if __name__=='__main__':main()
