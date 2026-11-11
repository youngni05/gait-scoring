"""Blind target tracking and candidate windows for LIVE patient videos. No GDI model runs here."""
import json,sys
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
POSE=BIG/'body25';VIDEOS=BIG/'videos';OUT=BIG/'qa';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def smooth(a):
    pad=np.pad(a,(4,4),mode='edge');return np.median(np.stack([pad[k:k+len(a)] for k in range(9)]),axis=0)
def select(people,height):
    """Same rule as child_prepare.select: exactly one lower-floor person with visible legs."""
    eligible=[]
    for i,p in enumerate(people):
        a=np.asarray(p['pose_keypoints_2d']).reshape(25,3);hips=a[[8,9,12]];ankles=a[[11,14]]
        hips=hips[hips[:,2]>.1];ankles=ankles[ankles[:,2]>.1]
        if not len(hips) or not len(ankles):continue
        if ankles[:,1].mean()-hips[:,1].mean()<.18*height:continue
        if ankles[:,1].mean()<.65*height:continue
        eligible.append(i)
    return (eligible[0] if len(eligible)==1 else -1),eligible
meta=[r for r in read(HERE/'video_metadata.json') if r['fps']<=40];results=[]
for m in meta:
    stem=Path(m['file']).stem;folder=POSE/stem;paths=sorted(folder.glob('*_keypoints.json'))
    if len(paths)!=m['frames']:continue
    raw=np.full((m['frames'],25,3),np.nan);picks=[];amb=0
    for f,p in enumerate(paths):
        people=read(p)['people'];sel,el=select(people,m['height']);amb+=len(el)>1;picks.append(sel)
        if sel>=0:raw[f]=np.asarray(people[sel]['pose_keypoints_2d']).reshape(25,3)
    valid=np.isfinite(raw[:,8,0])&(raw[:,8,2]>.1);obs=np.flatnonzero(valid)
    row=dict(record=stem,group=m['group'],person=m['person'],frames=m['frames'],width=m['width'],height=m['height'],valid_fraction=float(valid.mean()),ambiguous_frames=amb,candidates=[])
    if len(obs)>=124:
        hip=smooth(np.interp(np.arange(len(raw)),obs,raw[obs,8,0]))
        thigh=np.linalg.norm(raw[:,[9,12],:2]-raw[:,[10,13],:2],axis=2)
        scale=np.nanmedian(thigh[(raw[:,[9,12],2]>.1)&(raw[:,[10,13],2]>.1)])
        moving=(np.abs(np.gradient(hip))>max(1,.015*scale))&valid
        coverage=np.convolve(moving.astype(float),np.ones(124)/124,'valid')
        observed=np.convolve(valid.astype(float),np.ones(124)/124,'valid');rank=coverage.copy()
        for ci in range(2):
            start=int(np.argmax(rank))
            if rank[start]<0:break
            c=dict(start=start,end=start+124,moving_fraction=float(coverage[start]),observed_fraction=float(observed[start]),person_indices=picks[start:start+124])
            row['candidates'].append(c);rank[max(0,start-123):start+124]=-1
        wanted={int(f) for c in row['candidates'] if c['moving_fraction']>=.8 for f in np.rint(np.linspace(c['start'],c['end']-1,17))}
        if wanted:
            cap=cv2.VideoCapture(str(VIDEOS/m['file']));snaps={}
            for f in range(max(wanted)+1):
                ok,im=cap.read();assert ok,(stem,f)
                if f in wanted:snaps[f]=im
            cap.release()
            for ci,c in enumerate(row['candidates']):
                if c['moving_fraction']<.8:continue
                tw,th=(288,162) if m['width']>m['height'] else (108,192)
                ncol=4 if tw==288 else 9
                nrow=(17+ncol-1)//ncol
                canvas=np.full((30+nrow*(th+3),ncol*tw,3),255,np.uint8)
                cv2.putText(canvas,'%s c%d s%d mov%.2f obs%.2f'%(stem,ci,c['start'],c['moving_fraction'],c['observed_fraction']),(4,20),0,.5,(0,0,0),1)
                for k,f in enumerate(np.rint(np.linspace(c['start'],c['end']-1,17)).astype(int)):
                    im=snaps[int(f)].copy()
                    for j in [8,9,10,11,12,13,14,19,22]:
                        if raw[f,j,2]>.1:cv2.circle(im,tuple(raw[f,j,:2].astype(int)),6,(0,255,0),-1)
                    r,col=divmod(k,ncol);y=30+r*(th+3);canvas[y:y+th,col*tw:(col+1)*tw]=cv2.resize(im,(tw,th))
                cv2.imwrite(str(OUT/(stem+'-c%d.jpg'%ci)),canvas,[1,75])
    results.append(row);print(stem,'valid %.2f amb %d'%(row['valid_fraction'],amb),[(c['start'],round(c['moving_fraction'],2),round(c['observed_fraction'],2)) for c in row['candidates']],flush=True)
(HERE/'candidates.json').write_text(json.dumps(results,indent=1)+'\n',encoding='utf-8')
print('COMPLETE',len(results),'of',len(meta))
