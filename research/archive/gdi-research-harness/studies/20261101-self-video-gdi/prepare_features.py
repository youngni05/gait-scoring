"""Native BODY25 -> frozen features, with score-blind interval review assets."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BIG=Path('D:/gait-research-data/D_P/self-video-gdi-20261101')
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path[:0]=[str(BUNDLE),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'),str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')]
import numpy as np,cv2
from PIL import Image,ImageDraw
from preprocess import extract_windows,source_functions,LEFT,RIGHT,REQUIRED
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());inventory=json.loads((HERE/'inventory.json').read_text(encoding='utf8'))
    exception=json.loads((HERE/'confidence-exception.json').read_text(encoding='utf8'))
    proof=json.loads((HERE/'official-confidence-proof.json').read_text());assert proof['official_max_confidence']>1 and 'User explicitly' in exception['authorization']
    xs=[];metadata=[];quality=[];raws={}
    for row in inventory:
        clip=row['id'];files=sorted((BIG/'body25'/clip).glob('*keypoints.json'));assert len(files)==row['frames']
        raw=np.full((len(files),75),np.nan);counts=[];hashes={};allkp=[]
        for i,p in enumerate(files):
            assert int(p.stem.split('_')[-2])==i
            people=json.loads(p.read_text())['people'];counts.append(len(people));allkp.append(people);hashes[p.name]=sha(p)
            if len(people)==1:
                a=np.asarray(people[0]['pose_keypoints_2d']);assert a.shape==(75,) and np.isfinite(a).all()
                assert (a[2::3]>=0).all(),('negative_official_confidence',clip,i)
                raw[i]=a
        raws[clip]=raw;np.savez_compressed(BIG/(clip+'-raw.npz'),raw=raw,counts=counts)
        e=extract_windows(raw);ns=source_functions();processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
        primary=[]
        for x,start,miss in zip(e['x'],e['window_start'],e['window_missing_fraction']):
            metadata.append(dict(clip=clip,kind='primary',start=int(start),end=int(start)+123,missing=float(miss)));xs.append(x);primary.append(int(start))
        straight=[]
        for start in protocol['straight_starts'][clip]:
            miss=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[start:start+124]),REQUIRED))
            x=np.stack([processed[start:start+124,LEFT],processed[start:start+124,RIGHT]]).astype(np.float32)
            accepted=miss<=.25 and np.isfinite(x).all()
            straight.append(dict(start=start,end=start+123,missing=miss,accepted=bool(accepted)))
            if accepted:xs.append(x);metadata.append(dict(clip=clip,kind='straight',start=start,end=start+123,missing=miss))
        q=dict(clip=clip,frames=len(raw),one_person_frames=counts.count(1),empty_frames=counts.count(0),ambiguous_frames=sum(n>1 for n in counts),primary_starts=primary,primary_rejected=e['rejected'],straight=straight,max_confidence=float(np.nanmax(raw[:,2::3])),json_hashes=hashes)
        quality.append(q)
        # Same fixed 124-frame intervals: seven overlaid frames and nine RGB input frames.
        cap=cv2.VideoCapture(row['file']);canvas=Image.new('RGB',(2100,1080));draw=ImageDraw.Draw(canvas)
        for j,start in enumerate(protocol['straight_starts'][clip]):
            sampled=np.rint(np.linspace(start,start+123,9)).astype(int);rgb=[]
            for k,f in enumerate(sampled):
                cap.set(1,int(f));ok,im=cap.read();assert ok
                frame=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);img=Image.fromarray(frame);img.thumbnail((288,512));rgb.append(np.asarray(img))
            np.save(BIG/('%s-straight%d-rgb.npy'%(clip,j)),np.stack(rgb))
            for k,f in enumerate(np.rint(np.linspace(start,start+123,7)).astype(int)):
                cap.set(1,int(f));ok,im=cap.read();assert ok
                if counts[f]==1:
                    pts=raw[f].reshape(25,3)
                    for side,col in [([12,13,14,19],(0,255,0)),([9,10,11,22],(255,0,255))]:
                        for a,b in zip(side[:-1],side[1:]):
                            if pts[a,2]>0 and pts[b,2]>0:cv2.line(im,tuple(pts[a,:2].astype(int)),tuple(pts[b,:2].astype(int)),col,5)
                        for a in side:
                            if pts[a,2]>0:cv2.circle(im,tuple(pts[a,:2].astype(int)),6,col,-1)
                img=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));img.thumbnail((296,320));x=k*300;y=j*360;canvas.paste(img,(x,y+25));draw.text((x,y),'%s %d %.2fs'%(clip,f,f/30),fill='white')
        canvas.save(HERE/(clip+'-pose-review.jpg'));cap.release()
    np.savez_compressed(BIG/'features.npz',x=np.stack(xs))
    (HERE/'windows.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    (HERE/'quality.json').write_text(json.dumps(quality,indent=2),encoding='utf8')
    print(json.dumps([{k:v for k,v in r.items() if k!='json_hashes'} for r in quality],indent=2))
if __name__=='__main__':main()
