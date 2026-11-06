"""Review all source clips without scores; preserve original selection files."""
import csv,json
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
POSE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def smooth(a):
    pad=np.pad(a,(4,4),mode='edge')
    return np.median(np.stack([pad[k:k+len(a)] for k in range(9)]),axis=0)
info={Path(r['file']).stem:r for r in read(SOURCE/'td_download.json')['videos']}
out=HERE/'qa';out.mkdir(parents=True,exist_ok=True);results=[]
for meta in read(SOURCE/'video_metadata.json'):
    stem=Path(meta['file']).stem;folder=POSE/stem
    original={int(r['filename'].split('_')[-2]):int(r['person_index']) for r in csv.DictReader((folder/'selection.csv').open())}
    raw=np.full((meta['frames'],25,3),np.nan);picks=[]
    for f in range(meta['frames']):
        people=read(folder/('%s_%012d_keypoints.json'%(stem,f)))['people'];sel=original[f]
        if stem.startswith(('S03-','S07-')):
            # Source images establish child in foreground; no original tracks overwritten.
            # S07's sole foreground child was rejected by the old height threshold.
            candidates=[]
            for k,p in enumerate(people):
                a=np.asarray(p['pose_keypoints_2d']).reshape(25,3);feet=a[[11,14]];feet=feet[feet[:,2]>.1]
                if len(feet) and a[8,2]>.1 and feet[:,1].mean()>.65*meta['height']:
                    candidates.append((float(feet[:,1].mean()),k))
            candidates.sort(reverse=True);sel=candidates[0][1] if candidates else -1
            if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.04*meta['height']:sel=-1
        picks.append(sel)
        if sel>=0:raw[f]=np.asarray(people[sel]['pose_keypoints_2d']).reshape(25,3)
    valid=np.isfinite(raw[:,8,0])&(raw[:,8,2]>.1);obs=np.flatnonzero(valid)
    row=dict(record=stem,person=info[stem]['person'],age=info[stem]['age_years'],native_fps=meta['fps'],frames=meta['frames'],candidates=[],tracking_revised=stem.startswith(('S03-','S07-')))
    if len(obs):
        hip=smooth(np.interp(np.arange(len(raw)),obs,raw[obs,8,0]))
        thigh=np.linalg.norm(raw[:,[9,12],:2]-raw[:,[10,13],:2],axis=2)
        scale=np.nanmedian(thigh[(raw[:,[9,12],2]>.1)&(raw[:,[10,13],2]>.1)])
        moving=(np.abs(np.gradient(hip))>max(1,.015*scale))&valid
        coverage=np.convolve(moving.astype(float),np.ones(124)/124,'valid')
        observed=np.convolve(valid.astype(float),np.ones(124)/124,'valid');rank=coverage.copy()
        for ci in range(2):
            start=int(np.argmax(rank))
            if rank[start]<0:break
            cand=dict(start=start,end=start+124,moving_fraction=float(coverage[start]),observed_fraction=float(observed[start]),person_indices=picks[start:start+124])
            row['candidates'].append(cand);rank[max(0,start-123):start+124]=-1
        wanted={f for c in row['candidates'] if c['moving_fraction']>=.8 for f in np.rint(np.linspace(c['start'],c['end']-1,17)).astype(int)}
        if wanted:
            cap=cv2.VideoCapture(str(SOURCE/'td_sagittal'/meta['file']));snaps={}
            for f in range(max(wanted)+1):
                ok,im=cap.read();assert ok,(stem,f)
                if f in wanted:snaps[f]=im
            cap.release()
            for ci,c in enumerate(row['candidates']):
                if c['moving_fraction']<.8:continue
                canvas=np.full((880,1152,3),255,np.uint8)
                cv2.putText(canvas,'%s candidate%d start%d moving%.3f'%(stem,ci,c['start'],c['moving_fraction']),(8,22),0,.5,(0,0,0),1)
                for k,f in enumerate(np.rint(np.linspace(c['start'],c['end']-1,17)).astype(int)):
                    im=snaps[f].copy()
                    for j in [8,9,10,11,12,13,14,19,22]:
                        if raw[f,j,2]>.1:cv2.circle(im,tuple(raw[f,j,:2].astype(int)),5,(0,255,0),-1)
                    r,col=divmod(k,4);canvas[55+r*165:55+r*165+162,col*288:(col+1)*288]=cv2.resize(im,(288,162))
                    cv2.putText(canvas,'f%d p%d'%(f,picks[f]),(col*288+3,55+r*165+15),0,.43,(0,0,255),1)
                cv2.imwrite(str(out/(stem+'-c%d.jpg'%ci)),canvas)
    results.append(row);print(stem,[(c['start'],round(c['moving_fraction'],3)) for c in row['candidates']],flush=True)
(HERE/'live-candidates.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print('COMPLETE',len(results))
