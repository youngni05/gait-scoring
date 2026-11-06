"""Render actual chronological frames from the final accepted new passages."""
import json
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parents[1]/'gdi-research-harness/studies/20261031-cnn-high-gdi/research'
POSE=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-cnn-high-gdi/research/body25')
for row in json.loads((HERE/'membership.json').read_text())['child']:
    if row['person'] not in ['LIVE-TD:S03','LIVE-TD:S20']:continue
    stem=row['record'];frames=np.rint(np.linspace(row['start'],row['end']-1,17)).astype(int);cap=cv2.VideoCapture(str(SOURCE/'td_sagittal'/(stem+'.MOV')));snaps={}
    for f in range(row['end']):
        ok,im=cap.read();assert ok
        if f in frames:snaps[f]=im
    cap.release();canvas=np.full((880,1152,3),255,np.uint8)
    cv2.putText(canvas,stem+' FINAL %d:%d'%(row['start'],row['end']),(8,22),0,.55,(0,0,0),1)
    for k,f in enumerate(frames):
        im=snaps[f];sel=row['person_indices'][f-row['start']]
        if sel>=0:
            obj=json.loads((POSE/stem/('%s_%012d_keypoints.json'%(stem,f))).read_text());a=np.array(obj['people'][sel]['pose_keypoints_2d']).reshape(25,3)
            for j in [8,9,10,11,12,13,14,19,22]:
                if a[j,2]>.1:cv2.circle(im,tuple(a[j,:2].astype(int)),5,(0,255,0),-1)
        r,c=divmod(k,4);canvas[55+r*165:55+r*165+162,c*288:(c+1)*288]=cv2.resize(im,(288,162));cv2.putText(canvas,'f%d target%d'%(f,sel),(c*288+3,55+r*165+15),0,.43,(0,0,255),1)
    cv2.imwrite(str(HERE/'qa'/(stem+'-final.jpg')),canvas)
print('Final added passages rendered.')
