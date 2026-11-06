"""Locate candidate walking passages without consulting any GDI predictions."""
import csv,hashlib,json
from pathlib import Path
import cv2,numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[1]/'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
POSE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def smooth9(a):
    a=np.pad(a,(4,4),mode='edge')
    return np.median(np.stack([a[k:k+len(a)-8] for k in range(9)]),axis=0)
meta={Path(r['file']).stem:r for r in read(SOURCE/'video_metadata.json')}
info={Path(r['file']).stem:r for r in read(SOURCE/'td_download.json')['videos']}
retained=set(read(SOURCE/'target_final_review.json')['retained_records'])
out=HERE/'qa';out.mkdir(parents=True,exist_ok=True);result=[]
for stem in sorted(retained):
    m=meta[stem];folder=POSE/stem
    picks={int(r['filename'].split('_')[-2]):int(r['person_index']) for r in csv.DictReader((folder/'selection.csv').open())}
    raw=np.full((m['frames'],25,3),np.nan)
    for f,sel in picks.items():
        if sel>=0:raw[f]=np.asarray(read(folder/('%s_%012d_keypoints.json'%(stem,f)))['people'][sel]['pose_keypoints_2d']).reshape(25,3)
    valid=np.isfinite(raw[:,8,0])&(raw[:,8,2]>.1);obs=np.flatnonzero(valid)
    hip=smooth9(np.interp(np.arange(len(raw)),obs,raw[obs,8,0]))
    thigh=np.linalg.norm(raw[:,[9,12],:2]-raw[:,[10,13],:2],axis=2)
    scale=np.nanmedian(thigh[(raw[:,[9,12],2]>.1)&(raw[:,[10,13],2]>.1)])
    moving=(np.abs(np.gradient(hip))>max(1,.015*scale))&valid
    coverage=np.convolve(moving.astype(float),np.ones(124)/124,'valid')
    observed=np.convolve(valid.astype(float),np.ones(124)/124,'valid')
    start=int(np.argmax(coverage));row={'record':stem,'person':info[stem]['person'],'age':info[stem]['age_years'],
        'best_start':start,'best_end_exclusive':start+124,'best_moving_fraction':float(coverage[start]),
        'observed_fraction':float(observed[start]),'native_fps':m['fps'],
        'review_status':'pending','selection_note':'Maximum movement proxy over whole original clip; candidate finder only, not validated walking classifier.'}
    result.append(row)
    cap=cv2.VideoCapture(str(SOURCE/'td_sagittal'/(stem+'.MOV')))
    frames=np.unique(np.rint(np.linspace(start,start+123,17)).astype(int))
    canvas=np.full((5*165+55,4*288,3),255,np.uint8)
    cv2.putText(canvas,stem+' start=%d moving=%.3f observed=%.3f'%(start,coverage[start],observed[start]),(8,22),0,.5,(0,0,0),1)
    # Decode chronologically: random MOV seeking can return a different frame.
    snapshots={}
    for frame in range(int(frames[-1])+1):
        ok,im=cap.read();assert ok,(stem,frame)
        if frame in frames:snapshots[frame]=im
    for k,f in enumerate(frames):
        im=snapshots[f]
        for j in [8,9,10,11,12,13,14,19,22]:
            if raw[f,j,2]>.1:cv2.circle(im,tuple(raw[f,j,:2].astype(int)),5,(0,255,0),-1)
        im=cv2.resize(im,(288,162));r,c=divmod(k,4);canvas[55+r*165:55+r*165+162,c*288:(c+1)*288]=im
        cv2.putText(canvas,'f%d moving%d'%(f,moving[f]),(c*288+3,55+r*165+15),0,.43,(0,0,255),1)
    cap.release();cv2.imwrite(str(out/(stem+'.jpg')),canvas)
(HERE/'segment-candidates.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'records':len(result),'people':len(set(r['person'] for r in result)),
 'candidate_fraction_at_least_80pct':[{k:r[k] for k in ['record','age','best_start','best_moving_fraction','observed_fraction']} for r in result if r['best_moving_fraction']>=.8]},indent=2))
