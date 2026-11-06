"""Visualize abrupt joint motion for anatomical-side inspection, without scores."""
import argparse,csv,json,math,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pose_qc import select,REQUIRED,EDGES
HERE=Path(__file__).resolve().parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
p=argparse.ArgumentParser();p.add_argument('--group',choices=['normal','abnormal'],required=True);args=p.parse_args()
rows={r['record_id']:r for r in csv.DictReader((HERE/('cause_selected.csv')).open(encoding='utf-8-sig'))};quality=list(csv.DictReader((HERE/'qa'/('cause_highres')/'quality.csv').open(encoding='utf-8-sig')));quality=[r for r in quality if float(r['max_joint_jump_bbox_height'])>.25];meta=json.loads((DATA/'video_inventory.json').read_text());out=HERE/'qa'/('cause_highres_jumps');out.mkdir(parents=True,exist_ok=True);index=[]
for i,q in enumerate(quality):
    r=rows[q['record_id']];m=meta[r['video_id']];vi=json.loads(r['annotation_video_info']);boxes={b['frame']:b['bbox'] for b in json.loads((DATA/'annotations_by_sequence'/(r['record_id']+'.json')).read_text())};at=int(q['jump_at_frame']);cap=cv2.VideoCapture(m['path'])
    if i%4==0:canvas=np.full((4*260,3*360,3),245,np.uint8)
    y=(i%4)*260;cv2.putText(canvas,'%02d %s %s jump %.3f'%(i+1,r['video_id'],r['view'],float(q['max_joint_jump_bbox_height'])),(4,y+18),0,.5,(0,0,0),1)
    for j,f in enumerate([at-1,at,at+1]):
        file=DATA/('cause_highres_body25')/r['record_id']/('%s_%012d_keypoints.json'%(r['record_id'],f))
        if f not in boxes or not file.exists():continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,f-1);ok,im=cap.read()
        if not ok:continue
        b=boxes[f];box=[b['left']*m['width']/vi['width'],b['top']*m['height']/vi['height'],b['width']*m['width']/vi['width'],b['height']*m['height']/vi['height']];idx,a,_=select(json.loads(file.read_text())['people'],box)
        bx,by,bw,bh=box;x0=max(0,int(bx-.15*bw));x1=min(im.shape[1],int(bx+1.15*bw));y0=max(0,int(by-.05*bh));y1=min(im.shape[0],int(by+1.05*bh));crop=im[y0:y1,x0:x1].copy();scale=min(360/crop.shape[1],210/crop.shape[0]);crop=cv2.resize(crop,(int(crop.shape[1]*scale),int(crop.shape[0]*scale)))
        if a is not None:
            for k in REQUIRED:
                if a[k,2]<.1:continue
                xy=tuple(((a[k,:2]-[x0,y0])*scale).astype(int));color=(0,0,255) if k in [9,10,11,22] else (180,100,0);cv2.circle(crop,xy,3,color,-1);cv2.putText(crop,str(k),xy,0,.38,color,1)
        dx=j*360+(360-crop.shape[1])//2;dy=y+25;canvas[dy:dy+crop.shape[0],dx:dx+crop.shape[1]]=crop;cv2.putText(canvas,'frame '+str(f),(j*360+4,y+252),0,.42,(0,0,0),1)
    cap.release();index.append(dict(record_id=r['record_id'],video_id=r['video_id'],at_frame=at,index=i+1,page=i//4+1))
    if i%4==3 or i==len(quality)-1:cv2.imwrite(str(out/('sheet_%02d.jpg'%(i//4+1))),canvas)
(out/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps(dict(flagged=len(index),pages=math.ceil(len(index)/4))))
