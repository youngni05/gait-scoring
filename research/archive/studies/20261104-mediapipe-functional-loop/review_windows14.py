"""Score-blind first operational 4s-window contact sheets, before any TUG fit."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import cv2,numpy as np
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop'); B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
cv2.setNumThreads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=np.load(B/'bottom14-inputs.npz'); videos={r['person_id']:r for r in json.loads((H/'all-bottom-videos-manifest.json').read_text())['rows']}; rows=[];records=[]
for pid in sorted(set(d['person_ids'])):
    win=np.flatnonzero(d['person_ids']==pid)[0]; start=float(d['window_start_ms'][win]); pose=np.load(B/'bottom14'/f'{pid}.npz'); cap=cv2.VideoCapture(videos[pid]['file']); frames=[];sources=[]
    for offset in [0,1000,2000,3000]:
        t=start+offset; index=int(np.argmin(np.abs(pose['source_ms']-t))); cap.set(cv2.CAP_PROP_POS_FRAMES,index);okay,im=cap.read(); assert okay; h,w=im.shape[:2]
        xy=pose['landmarks'][index];v=pose['valid'][index]
        for l,r in [(11,12),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,31),(28,32)]:
            if v[l] and v[r]:cv2.line(im,tuple((xy[l,:2]*[w,h]).astype(int)),tuple((xy[r,:2]*[w,h]).astype(int)),(0,255,0),3)
        im=cv2.resize(im,(200,356));cv2.putText(im,f'{pid} {t/1000:.1f}s',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,0,255),1); frames.append(im);sources.append(dict(frame=index,presentation_ms=float(pose['source_ms'][index]),provider=int(pose['provider'][index])))
    cap.release(); rows.append(np.hstack(frames));records.append(dict(person_id=pid,selected_first_operational_window_ms=start,samples=sources))
for part,lo in enumerate(range(0,len(rows),4)):cv2.imwrite(str(H/f'first-windows-review-{part}.jpg'),np.vstack(rows[lo:lo+4]))
(H/'first-windows-review-manifest.json').write_text(json.dumps(dict(status='pending_visual_review',source_input_sha256=sha(B/'bottom14-inputs.npz'),score_labels_read=False,records=records,limits='Four samples illustrate target identity and displacement only; cannot prove every frame or exclude all turns/standing. Review source time progression before gait claims.'),indent=2),encoding='utf8')
print('first window score-blind contacts',len(rows))
