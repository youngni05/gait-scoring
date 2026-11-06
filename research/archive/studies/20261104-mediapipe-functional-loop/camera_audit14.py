"""Score-independent native-camera inventory/contact sheets for all fourteen people."""
import sys,json,hashlib
from pathlib import Path
DEP=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps');sys.path.insert(0,str(DEP))
import cv2,numpy as np
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop')
manifest=json.loads((H/'all-bottom-videos-manifest.json').read_text());assert manifest['whole_archive_md5_verified'] and len(manifest['rows'])==14;cv2.setNumThreads(2);contact={q:[] for q in [.2,.5,.8]};records=[]
for row in manifest['rows']:
    cap=cv2.VideoCapture(row['file']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));assert n>0 and fps>0 and W>0 and h>0
    records.append(dict(person_id=row['person_id'],frames=n,fps=fps,width=W,height=h,seconds=n/fps,source_sha256=row['sha256']))
    for fraction in contact:
        i=int(n*fraction);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,b=cap.read();assert ok;b=cv2.resize(b,(160,round(h*160/W)));cv2.rectangle(b,(round(.28*b.shape[1]),0),(round(.85*b.shape[1]),b.shape[0]-1),(0,255,0),1);cv2.putText(b,row['person_id'],(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,0,255),1);contact[fraction].append(b)
    cap.release()
for fraction,images in contact.items():
    assert len(set(q.shape for q in images))==1;grid=np.vstack([np.hstack(images[:7]),np.hstack(images[7:])]);cv2.imwrite(str(H/f'camera14-{int(fraction*100)}percent.jpg'),grid)
(H/'camera14-metadata.json').write_text(json.dumps(dict(status='native_metadata_verified_pending_visual_target_review',people=14,records=records,contact_frames='Each video20/50/80percent; no clinical score used or displayed. Green lines show proposed central target corridor, not a clinical measurement.',clinical_fit_started=False),indent=2),encoding='utf8');print('14camera metadata/contact sheets prepared')
