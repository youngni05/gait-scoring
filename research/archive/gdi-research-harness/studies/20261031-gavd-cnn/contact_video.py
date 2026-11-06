"""Python 3.7 / existing OpenCV: annotation-bbox review sheets, no pose inference."""
import argparse,csv,json,math
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')

def main(args):
    with open(str(args.manifest or HERE/'review_template.csv'),encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if args.label:rows=[r for r in rows if r['label']==args.label]
    meta=json.loads((DATA/'video_inventory.json').read_text())
    out=HERE/'qa'/args.tag;out.mkdir(parents=True,exist_ok=True)
    index=[];canvas=None
    for i,r in enumerate(rows):
        if i%18==0:canvas=np.full((6*220,3*640,3),245,dtype=np.uint8)
        x=(i%3)*640;y=((i%18)//3)*220
        cv2.putText(canvas,'%03d %s %s'%(i+1,r['video_id'],r['view']),(x+4,y+18),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,0,0),1,cv2.LINE_AA)
        boxes=json.loads((DATA/'annotations_by_sequence'/(r['record_id']+'.json')).read_text())
        vi=json.loads(r['annotation_video_info']);cap=cv2.VideoCapture(meta[r['video_id']]['path'])
        success=[]
        for j,pos in enumerate([len(boxes)//4,3*len(boxes)//4]):
            b=boxes[pos];decoded=b['frame']+args.offset
            cap.set(cv2.CAP_PROP_POS_FRAMES,decoded);ok,im=cap.read();success.append(bool(ok))
            if not ok:continue
            height,width=im.shape[:2];bb=b['bbox'];sx=width/vi['width'];sy=height/vi['height']
            cv2.rectangle(im,(int(bb['left']*sx),int(bb['top']*sy)),(int((bb['left']+bb['width'])*sx),int((bb['top']+bb['height'])*sy)),(0,255,0),2)
            scale=min(320/width,176/height);thumb=cv2.resize(im,(int(width*scale),int(height*scale)))
            dy=y+25+(176-thumb.shape[0])//2;dx=x+320*j+(320-thumb.shape[1])//2
            canvas[dy:dy+thumb.shape[0],dx:dx+thumb.shape[1]]=thumb
            cv2.putText(canvas,'ann %s dec %s'%(b['frame'],decoded),(x+320*j+4,y+215),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA)
        cap.release()
        index.append(dict(index=i+1,seq=r['record_id'],video_id=r['video_id'],view=r['view'],fps=meta[r['video_id']]['fps'],
                          annotation_offset=args.offset,sampled_frames_read=success,page=i//18+1))
        if i%18==17 or i==len(rows)-1:cv2.imwrite(str(out/('sheet_%02d.jpg'%(i//18+1))),canvas)
    (out/'index.json').write_text(json.dumps(index,indent=2))
    print(json.dumps(dict(sequences=len(rows),pages=int(math.ceil(len(rows)/18)),output=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--label');p.add_argument('--tag',default='normal_video');p.add_argument('--offset',type=int,default=-1)
    main(p.parse_args())
