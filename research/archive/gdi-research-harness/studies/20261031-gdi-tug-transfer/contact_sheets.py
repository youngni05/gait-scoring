"""Review camera and walking intervals without loading clinical labels."""
from pathlib import Path
import argparse,cv2,json,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--person');args=ap.parse_args()
OUT=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261031-gdi-tug-transfer')
for receipt in sorted((OUT/'acquisition').glob('*.json')):
    row=json.loads(receipt.read_text())
    if args.person and row['person']!=args.person:continue
    dest=OUT/('contact-%02d%s.jpg'%(int(row['person']),'-full' if args.full else ''))
    if dest.exists():continue
    cap=cv2.VideoCapture(row['path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    tiles=[]
    # Inspect first 25 seconds with enough temporal detail to choose first pass.
    indices=np.linspace(0,n-1 if args.full else min(n-1,int(25*fps)),12).astype(int)
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
        h,w=im.shape[:2];s=min(280/w,330/h);small=cv2.resize(im,(round(w*s),round(h*s)))
        tile=np.full((365,300,3),245,np.uint8);tile[:small.shape[0],:small.shape[1]]=small
        cv2.putText(tile,'%s f%d %.2fs'%(row['person'],i,i/fps),(5,350),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);tiles.append(tile)
    cap.release();cv2.imwrite(str(dest),np.vstack([np.hstack(tiles[j:j+4]) for j in (0,4,8)]))
    print(row['person'],n,fps,flush=True)
