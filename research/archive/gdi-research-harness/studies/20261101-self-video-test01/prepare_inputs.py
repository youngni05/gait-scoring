"""Apply unchanged native BODY25 preprocessing; show failures without replacing clips."""
import sys,json,hashlib,argparse,warnings
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BIG=Path('D:/gait-research-data/D_P/self-video-test01-20261101')
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path[:0]=[str(BUNDLE),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'),str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')]
import numpy as np,cv2
from PIL import Image,ImageDraw
from preprocess import extract_windows
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main(partial=False):
    xs=[];windows=[];quality=[];inventory=read(HERE/'inventory.json');provenance={}
    for row in inventory:
        clip=row['id'];n=row['processed_frames'];files=sorted((BIG/'body25'/clip).glob('*keypoints.json'))
        if len(files)!=n:
            if partial:quality.append(dict(number=row['number'],id=clip,status='pending',frames_ready=len(files),expected=n));continue
            raise ValueError(('incomplete_pose',clip,len(files),n))
        raw=np.full((n,75),np.nan);counts=[];all_conf=[];hashes={}
        for i,p in enumerate(files):
            assert int(p.stem.split('_')[-2])==i
            people=read(p)['people'];counts.append(len(people));hashes[p.name]=sha(p)
            if len(people)==1:
                a=np.asarray(people[0]['pose_keypoints_2d'],dtype=np.float64)
                assert a.shape==(75,) and np.isfinite(a).all() and (a[2::3]>=0).all()
                raw[i]=a;all_conf.extend(a[2::3].tolist())
        np.savez_compressed(BIG/(clip+'-raw.npz'),raw=raw,counts=counts)
        q=dict(number=row['number'],id=clip,frames=n,one_person_frames=counts.count(1),empty_frames=counts.count(0),ambiguous_frames=sum(v>1 for v in counts),confidence_over1_values=sum(v>1 for v in all_conf),maximum_confidence=max(all_conf) if all_conf else None)
        try:
            extracted=extract_windows(raw)
            for x,start,missing in zip(extracted['x'],extracted['window_start'],extracted['window_missing_fraction']):
                xs.append(x);windows.append(dict(id=clip,number=row['number'],start=int(start),end=int(start)+123,missing_fraction=float(missing)))
            q.update(status='prepared',valid_windows=len(extracted['x']),starts=extracted['window_start'].tolist(),rejected=extracted['rejected'],processed_after_tail_trim=extracted['processed_frames'])
        except ValueError as e:q.update(status='no_eligible_input',valid_windows=0,reason=str(e))
        quality.append(q);provenance[clip]=hashes
        cap=cv2.VideoCapture(row['file']);canvas=Image.new('RGB',(1800,790));draw=ImageDraw.Draw(canvas)
        frames=np.rint(np.linspace(0,min(n-1,495),12)).astype(int)
        for j,f in enumerate(frames):
            cap.set(1,int(f));ok,im=cap.read();assert ok
            if counts[f]==1:
                kp=raw[f].reshape(25,3)
                for side,col in [([12,13,14,19],(0,255,0)),([9,10,11,22],(255,0,255))]:
                    for a,b in zip(side[:-1],side[1:]):
                        if kp[a,2]>0 and kp[b,2]>0:cv2.line(im,tuple(kp[a,:2].astype(int)),tuple(kp[b,:2].astype(int)),col,5)
                    for a in side:
                        if kp[a,2]>0:cv2.circle(im,tuple(kp[a,:2].astype(int)),6,col,-1)
            img=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));img.thumbnail((296,350));x=j%6*300;y=j//6*395
            canvas.paste(img,(x,y+25));draw.text((x,y),'%s frame%d n=%d'%(clip,f,counts[f]),fill='white')
        cap.release();canvas.save(HERE/(clip+'-pose-review.jpg'));print(json.dumps(q),flush=True)
    if xs:np.savez_compressed(BIG/'features.npz',x=np.stack(xs))
    for name,value in [('windows.json',windows),('quality.json',quality),('input-provenance.json',provenance)]:
        (HERE/name).write_text(json.dumps(value,indent=2),encoding='utf8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
