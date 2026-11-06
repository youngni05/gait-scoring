"""Inspect maximum cross-resolution disagreement independently of model scores."""
import csv,json
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
old={r['record_id']:r for r in rows(STUDY/'reviewed_shift0.csv')};new=rows(HERE/'reviewed_highres.csv');meta=json.loads((DATA/'video_inventory.json').read_text());required=[9,10,11,12,13,14,19,22];canvas=np.full((4*310,3*370,3),245,np.uint8);index=[]
for ri,r in enumerate(new):
    k=r['record_id'];b=old[k];s0={a['filename']:int(a['person_index']) for a in rows(Path(b['selection_csv']))};s1={a['filename']:int(a['person_index']) for a in rows(Path(r['selection_csv']))};best=(-1,None,None,None,None)
    for name,ix in s0.items():
        iy=s1[name]
        if min(ix,iy)<0:continue
        a=np.array(json.loads((Path(b['json_dir'])/name).read_text())['people'][ix]['pose_keypoints_2d']).reshape(25,3);c=np.array(json.loads((Path(r['json_dir'])/name).read_text())['people'][iy]['pose_keypoints_2d']).reshape(25,3)
        for joint in required:
            if min(a[joint,2],c[joint,2])<.1:continue
            distance=float(np.linalg.norm(a[joint,:2]-c[joint,:2]))
            if distance>best[0]:best=(distance,name,joint,a,c)
    distance,name,joint,a,c=best;frame=int(name.split('_')[-2]);m=meta[r['video_id']];vi=json.loads(r['annotation_video_info']);boxes={a['frame']:a['bbox'] for a in json.loads((DATA/'annotations_by_sequence'/(k+'.json')).read_text())};box=boxes[frame];x=box['left']*m['width']/vi['width'];y=box['top']*m['height']/vi['height'];w=box['width']*m['width']/vi['width'];h=box['height']*m['height']/vi['height'];x0=max(0,int(x-.2*w));x1=min(m['width'],int(x+1.2*w));y0=max(0,int(y-.05*h));y1=min(m['height'],int(y+1.08*h))
    cap=cv2.VideoCapture(m['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read();cap.release();assert ok
    crop=im[y0:y1,x0:x1];scale=min(370/crop.shape[1],250/crop.shape[0]);crop=cv2.resize(crop,(int(crop.shape[1]*scale),int(crop.shape[0]*scale)))
    cv2.putText(canvas,r['video_id']+' frame%d joint%d diff%.1fpx'%(frame,joint,distance),(5,ri*310+18),0,.5,(0,0,0),1)
    for ci,pose in enumerate([None,a,c]):
        tile=crop.copy()
        if pose is not None:
            for j in required:
                if pose[j,2]<.1:continue
                xy=tuple(((pose[j,:2]-[x0,y0])*scale).astype(int));color=(0,0,255) if j in [9,10,11,22] else (180,100,0);cv2.circle(tile,xy,3,color,-1);cv2.putText(tile,str(j),xy,0,.4,color,1)
        dx=ci*370+(370-tile.shape[1])//2;dy=ri*310+25;canvas[dy:dy+tile.shape[0],dx:dx+tile.shape[1]]=tile;cv2.putText(canvas,['Original','BODY25 368','BODY25 544'][ci],(ci*370+5,ri*310+300),0,.5,(0,0,0),1)
    index.append(dict(record_id=k,video_id=r['video_id'],frame=frame,joint=joint,distance_pixels=distance))
cv2.imwrite(str(HERE/'pose_disagreements.jpg'),canvas);(HERE/'pose_disagreements.json').write_text(json.dumps(index,indent=2));print(index)
