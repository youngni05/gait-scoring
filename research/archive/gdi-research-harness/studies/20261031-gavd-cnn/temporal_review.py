"""Eight chronological views per complete annotation segment, before CNN scoring."""
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig',newline='')));meta=json.loads((DATA/'video_inventory.json').read_text());out=HERE/'qa'/a.tag;out.mkdir(parents=True,exist_ok=True);index=[]
for i,r in enumerate(rows):
    if i%4==0:canvas=np.full((4*175,8*200,3),245,np.uint8)
    y=(i%4)*175;cv2.putText(canvas,'%02d %s %s %s'%(i+1,r['video_id'],r['view'],r['gait_pattern']),(4,y+18),0,.5,(0,0,0),1)
    boxes=json.loads((DATA/'annotations_by_sequence'/(r['record_id']+'.json')).read_text());m=meta[r['video_id']];vi=json.loads(r['annotation_video_info']);cap=cv2.VideoCapture(m['path'])
    frames=[]
    for j,pos in enumerate(np.linspace(0,len(boxes)-1,8).astype(int)):
        b=boxes[pos];frame=b['frame'];frames.append(frame);cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read()
        if not ok:continue
        bb=b['bbox'];sx=im.shape[1]/vi['width'];sy=im.shape[0]/vi['height'];cv2.rectangle(im,(int(bb['left']*sx),int(bb['top']*sy)),(int((bb['left']+bb['width'])*sx),int((bb['top']+bb['height'])*sy)),(0,255,0),1)
        scale=min(200/im.shape[1],128/im.shape[0]);im=cv2.resize(im,(int(im.shape[1]*scale),int(im.shape[0]*scale)));dx=j*200+(200-im.shape[1])//2;dy=y+25+(128-im.shape[0])//2;canvas[dy:dy+im.shape[0],dx:dx+im.shape[1]]=im;cv2.putText(canvas,str(frame),(j*200+4,y+169),0,.4,(0,0,0),1)
    cap.release();index.append(dict(index=i+1,record_id=r['record_id'],video_id=r['video_id'],frames=frames,page=i//4+1))
    if i%4==3 or i==len(rows)-1:cv2.imwrite(str(out/('sheet_%02d.jpg'%(i//4+1))),canvas)
(out/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps(dict(records=len(rows),pages=math.ceil(len(rows)/4))))
