"""Resume-safe retrieval and official CPU decoding; remote float32 data retained."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import cv2,numpy as np
ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'));os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op
REMOTE='/abr/RESEARCH_USER/gdi-tug-transfer-20261031'
HOST='RESEARCH_USER@PRIVATE_HOST'
AUTH=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ForwardAgent=no','-i','C:/Users/jun/.ssh/id_ed25519_gdi_abrm02.pub']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe','-p','10000']+AUTH+[HOST]
SCP=['C:/Windows/System32/OpenSSH/scp.exe','-P','10000']+AUTH
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def main():
    cv2.setNumThreads(2)
    folder=OUT/'body25';folder.mkdir(exist_ok=True);cache=OUT/'received-heatmaps';cache.mkdir(exist_ok=True)
    donepath=OUT/'decoded-chunks.json';done=json.loads(donepath.read_text()) if donepath.exists() else {}
    metadata={r['person']:r for r in json.loads((OUT/'input-manifest.json').read_text())}
    wrapper=op.WrapperPython();wrapper.configure({'model_folder':str(OP/'models')+'/', 'body':2,'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0});wrapper.start()
    result=subprocess.run(SSH+['cat '+REMOTE+'/heatmaps/index.json'],capture_output=True,timeout=45,check=True)
    index=json.loads(result.stdout);save(OUT/'remote-index.json',index)
    for row in index:
        if row['file'] in done:
            assert done[row['file']]['source_sha256']==row['sha256'];continue
        path=cache/row['file']
        if not path.exists():subprocess.run(SCP+[HOST+':'+REMOTE+'/heatmaps/'+row['file'],str(path)],check=True,timeout=240)
        assert sha(path)==row['sha256']
        meta=metadata[row['person']];im=np.zeros((meta['height'],meta['width'],3),np.uint8)
        dest=folder/('OAW%02d'%int(row['person']));dest.mkdir(exist_ok=True);hashes={}
        with np.load(str(path)) as z:
            assert len(z['heatmaps'])==row['count']
            for offset,heatmap in enumerate(z['heatmaps']):
                d=op.Datum();d.cvInputData=im;d.poseNetOutput=np.ascontiguousarray(heatmap)
                wrapper.emplaceAndPop(op.VectorDatum([d]));kp=d.poseKeypoints
                frame=meta['first']+row['first_offset']+offset
                p=dest/('%06d.json'%frame);save(p,{'frame':frame,'people':[] if kp is None else kp.tolist()});hashes[p.name]=sha(p)
        done[row['file']]={'source_sha256':row['sha256'],'json_sha256':hashes,'frames':len(hashes)};save(donepath,done)
        print('decoded',row['person'],row['first_offset'],'chunks',len(done),flush=True)
    wrapper.stop()
    result=subprocess.run(SSH+['cat '+REMOTE+'/complete.json'],capture_output=True,timeout=45)
    if result.returncode==0:
        completed=json.loads(result.stdout);save(OUT/'remote-complete.json',completed)
        assert completed['input_manifest_sha256']==sha(OUT/'input-manifest.json')
        if len(done)==completed['chunks']:
            assert sum(r['frames'] for r in done.values())==1488
            differences=[]
            for frame in range(250,255):
                original=json.loads((OUT/'pilot_pose'/('%06d.json'%frame)).read_text())['people']
                decoded=json.loads((folder/'OAW06'/('%06d.json'%frame)).read_text())['people']
                a=np.asarray(original);b=np.asarray(decoded);assert a.shape==b.shape
                delta=float(np.abs(a-b).max());assert delta<.01,(frame,delta)
                differences.append(delta)
            save(OUT/'decode-verification.json',{'passed':True,'frames':1488,'people':12,'official_cpu_reference_max_differences':differences,'tolerance':.01,
                                                'scope':'Frame and model equivalence only; participant tracking still requires visual audit'})
            print('BODY25_DECODE_VERIFIED',flush=True)
if __name__=='__main__':main()
