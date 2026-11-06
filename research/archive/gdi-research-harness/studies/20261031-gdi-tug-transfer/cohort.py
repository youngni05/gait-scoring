"""Select and visually audit tracks before exporting any model input or score."""
from pathlib import Path
import argparse,hashlib,json
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((OUT/'decode-verification.json').read_text())['passed']
    intervals=json.loads((OUT/'intervals.json').read_text());rows=json.loads((OUT/'input-manifest.json').read_text())
    originals={r['person']:r for r in json.loads((OUT/'acquisition.json').read_text())['videos']}
    decoded=json.loads((OUT/'decoded-chunks.json').read_text());hashes={}
    for r in decoded.values():hashes.update(r['json_sha256'])
    out=OUT/'tracks';out.mkdir(exist_ok=True);review=[]
    for row in rows:
        person=row['person'];folder=OUT/'body25'/('OAW%02d'%int(person));raw=[];selections=[];previous=None;flags=[]
        for frame in range(row['first'],row['first']+124):
            p=folder/('%06d.json'%frame)
            # Per-person and per-chunk key, because source frame numbers repeat.
            chunk=next(r for name,r in decoded.items() if name.startswith('OAW%02d_'%int(person)) and p.name in r['json_sha256'])
            assert sha(p)==chunk['json_sha256'][p.name]
            people=json.loads(p.read_text())['people'];candidates=[]
            for i,arr in enumerate(people):
                a=np.asarray(arr)
                if a[8,2]>.1 and max(a[11,2],a[14,2])>.1:
                    candidates.append((i,a))
            if not candidates:raw.append(np.full(75,np.nan));selections.append(-1);flags.append(frame);continue
            if previous is None:
                select=min if intervals['target_initial_side'][person]=='left' else max
                i,a=select(candidates,key=lambda item:item[1][8,0])
            else:i,a=min(candidates,key=lambda item:np.linalg.norm(item[1][8,:2]-previous))
            if previous is not None and np.linalg.norm(a[8,:2]-previous)>.1*row['height']:flags.append(frame)
            previous=a[8,:2];raw.append(a.reshape(-1));selections.append(i)
        raw=np.stack(raw);dest=out/('OAW%02d.npz'%int(person));np.savez_compressed(str(dest),raw=raw,selected_index=selections)
        cap=cv2.VideoCapture(originals[person]['path']);tiles=[]
        for offset in (0,31,62,93,123):
            frame=row['first']+offset;cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
            a=raw[offset].reshape(25,3)
            for x,y,c in a:
                if np.isfinite(c) and c>.1:cv2.circle(im,(round(x),round(y)),7,(0,255,0),-1)
            cv2.putText(im,'selected %d'%selections[offset],(30,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
            tiles.append(cv2.resize(im,(216,384)))
        cap.release();cv2.imwrite(str(OUT/('track-%02d.jpg'%int(person))),np.hstack(tiles))
        review.append({'person':person,'track_file':dest.name,'sha256':sha(dest),'flags':flags,'missing_frames':selections.count(-1)})
    (OUT/'track-review.json').write_text(json.dumps(review,indent=2))
    print('Tracks need visual QA of all twelve track images before feature export',flush=True)
if __name__=='__main__':main()
