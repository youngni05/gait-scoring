"""Read-only source audit; all generated artifacts are explicitly stored on D:."""
from pathlib import Path
import argparse, csv, hashlib, io, json, zipfile
import cv2
import numpy as np

ROOT = Path('D:/Projects/D_P')
OUT = Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
SOURCE = ROOT/'outputs/gait-public-access-20261015/Toronto'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert OUT.resolve().drive.upper()=='D:'
    archive=SOURCE/'PoseTracking.zip'
    audit={'pose_archive_sha256':sha(archive),'source_video':str((SOURCE/'OAW06-bottom.mp4').resolve()),'pose_formats':{}}
    with zipfile.ZipFile(archive) as z:
        for method in ('OpenPose','Alphapose'):
            names=sorted(n for n in z.namelist() if '/'+method+'/' in n and n.endswith('.csv'))
            headers={}
            for n in names:
                header=next(csv.reader(io.StringIO(z.read(n).decode())))
                key=','.join(header);headers[key]=headers.get(key,0)+1
            audit['pose_formats'][method]={'files':len(names),'headers':headers,'has_foot_points':any('Toe' in h or 'Heel' in h for h in headers)}
    p=SOURCE/'OAW06-bottom.mp4'
    cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    audit['video']={'sha256':sha(p),'frames':n,'fps':fps,'width':cap.get(3),'height':cap.get(4),'rotation_metadata':cap.get(48)}
    tiles=[]
    for i in np.linspace(0,n-1,12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read()
        assert ok
        h,w=im.shape[:2];s=min(280/w,330/h);small=cv2.resize(im,(round(w*s),round(h*s)))
        tile=np.full((365,300,3),245,np.uint8);tile[:small.shape[0],:small.shape[1]]=small
        cv2.putText(tile,'frame %d / %.2fs'%(i,i/fps),(5,350),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        tiles.append(tile)
    cap.release()
    cv2.imwrite(str(OUT/'pilot-contact.jpg'),np.vstack([np.hstack(tiles[j:j+4]) for j in (0,4,8)]))
    (OUT/'source-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(json.dumps(audit['video'],indent=2))

if __name__=='__main__':main()
