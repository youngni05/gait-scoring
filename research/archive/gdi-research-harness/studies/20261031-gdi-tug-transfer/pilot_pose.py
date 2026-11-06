"""Official CPU BODY25 on one fixed walking interval; no score-based selection."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import cv2
import numpy as np

ROOT=Path('D:/Projects/D_P')
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
OP=ROOT/'tmp/openpose-cpu/openpose'
sys.path.append(str(OP/'bin/python/openpose/Release'))
os.environ['PATH']=str(OP/'bin')+';'+os.environ['PATH']
import pyopenpose as op

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=124);args=ap.parse_args()
    assert 1<=args.count<=124
    source=ROOT/'outputs/gait-public-access-20261015/Toronto/OAW06-bottom.mp4'
    dest=OUT/'pilot_pose';dest.mkdir(parents=True,exist_ok=True)
    audit=json.loads((OUT/'source-audit.json').read_text())
    assert sha(source)==audit['video']['sha256']
    params={'model_folder':str(OP/'models')+'/', 'model_pose':'BODY_25','net_resolution':'-1x368','render_pose':0}
    cv2.setNumThreads(2)
    w=op.WrapperPython();w.configure(params);w.start()
    cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,250)
    assert cap.get(cv2.CAP_PROP_POS_FRAMES)==250
    started=time.time()
    for index in range(250,250+args.count):
        ok,im=cap.read();assert ok
        path=dest/('%06d.json'%index)
        if path.exists(): continue
        datum=op.Datum();datum.cvInputData=im
        w.emplaceAndPop(op.VectorDatum([datum]))
        kp=datum.poseKeypoints
        people=[] if kp is None else kp.tolist()
        path.write_text(json.dumps({'frame':index,'people':people}),encoding='utf8')
        if index in (250,280,310,340,373):
            canvas=im.copy()
            for person,a in enumerate(people):
                a=np.asarray(a)
                for x,y,c in a:
                    if c>.1:cv2.circle(canvas,(round(x),round(y)),6,(0,255,0),-1)
                x,y,c=a[8]
                if c>.1:cv2.putText(canvas,str(person),(round(x),round(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            cv2.imwrite(str(dest/('%06d.jpg'%index)),canvas)
        if (index-249)%10==0:print(index,'seconds',round(time.time()-started,2),flush=True)
    cap.release();w.stop()
    files=sorted(dest.glob('*.json'))
    result={'source_sha256':audit['video']['sha256'],'start_frame':250,'requested_frames':args.count,'fps':30,
            'selection':'Fixed 250..373 interval from visual source review before any predictions; frontal approach, not sagittal.',
            'params':params,'elapsed_seconds':time.time()-started,'json_sha256':{p.name:sha(p) for p in files},
            'weight_sha256':sha(OP/'models/pose/body_25/pose_iter_584000.caffemodel')}
    (OUT/'pilot-pose-receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'frames':len(files),'seconds':result['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
