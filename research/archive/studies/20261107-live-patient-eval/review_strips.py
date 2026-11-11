"""Compact 17-frame strips of each candidate for visual review. Scores are not available here."""
import json,sys
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/live-patient-eval-20261107')
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True);which=int(sys.argv[2])
qc=[d for d in json.loads((HERE/'candidates-qc.json').read_text(encoding='utf-8')) if d['passes']]
rows={r['record']:r for r in json.loads((HERE/'candidates.json').read_text(encoding='utf-8'))}
meta={Path(r['file']).stem:r for r in json.loads((HERE/'video_metadata.json').read_text())}
items=[]
for rec in sorted({d['record'] for d in qc}):
    c=[d for d in qc if d['record']==rec]
    if which==0:items.append(c[0])
    elif len(c)>1:items.append(c[1])
TW,TH=88,170;strips=[]
for d in items:
    rec=d['record'];c=rows[rec]['candidates'][d['candidate']];cap=cv2.VideoCapture(str(BIG/'videos'/meta[rec]['file']))
    fr=[int(f) for f in np.rint(np.linspace(c['start'],c['end']-1,17))];snap={}
    for f in range(max(fr)+1):
        ok,im=cap.read()
        if f in fr:snap[f]=im
    cap.release()
    strip=np.full((TH+16,17*TW,3),255,np.uint8)
    cv2.putText(strip,'%s c%d %s mov%.2f miss%.2f th%.0f'%(rec[:24],d['candidate'],d['group'],d['moving'],d['missing'],d['thigh']),(2,12),0,.42,(0,0,200),1)
    last=None
    for k,f in enumerate(fr):
        sel=c['person_indices'][f-c['start']];im=snap[f];hip=None
        if sel>=0:
            p=json.loads((BIG/'body25'/rec/('%s_%012d_keypoints.json'%(rec,f))).read_text())['people'][sel];a=np.asarray(p['pose_keypoints_2d']).reshape(25,3)
            for j in [8,9,10,11,12,13,14,1]:
                if a[j,2]>.1:cv2.circle(im,tuple(a[j,:2].astype(int)),5,(0,255,0),-1)
            hip=int(a[8,0]) if a[8,2]>.1 else None
        last=hip if hip is not None else last
        cx=last if last is not None else 640;x0=int(np.clip(cx-155,0,im.shape[1]-310))
        crop=im[50:650,x0:x0+310] if im.shape[0]>=650 else im[:,x0:x0+310]
        strip[16:16+TH,k*TW:(k+1)*TW-1]=cv2.resize(crop,(TW-1,TH))
    strips.append(strip);print(rec,flush=True)
for i in range(0,len(strips),6):
    cv2.imwrite(str(OUT/('sheet%d-%02d.jpg'%(which,i//6))),np.vstack(strips[i:i+6]),[1,82])
print(len(strips))
