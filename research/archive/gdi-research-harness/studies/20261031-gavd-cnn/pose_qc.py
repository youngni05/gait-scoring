"""Score-blind annotation-box target selection and BODY25 review sheets (Python 3.7)."""
from pathlib import Path
import argparse,csv,json,re
import cv2,numpy as np
HERE=Path(__file__).resolve().parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
REQUIRED=[9,10,11,12,13,14,19,22]
EDGES=[(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),(14,19),(11,22)]
def select(people,box):
    candidates=[];x,y,bw,bh=box
    for i,p in enumerate(people):
        a=np.asarray(p['pose_keypoints_2d']).reshape(25,3);v=a[:,2]>=.1
        if v.sum()<6:continue
        inside=(a[:,0]>=x)&(a[:,0]<=x+bw)&(a[:,1]>=y)&(a[:,1]<=y+bh)
        hip=a[[8,9,12]];hv=hip[:,2]>=.1
        if not hv.any():continue
        hc=hip[hv,:2].mean(0)
        if not (x<=hc[0]<=x+bw and y<=hc[1]<=y+bh):continue
        lo=a[v,:2].min(0);hi=a[v,:2].max(0)
        inter=np.maximum(0,np.minimum(hi,[x+bw,y+bh])-np.maximum(lo,[x,y])).prod()
        union=(hi-lo).prod()+bw*bh-inter;iou=inter/union if union>0 else 0
        fraction=float(inside[v].mean())
        if fraction<.7 or iou<.2:continue
        candidates.append((.6*iou+.4*fraction,i,a))
    candidates.sort(key=lambda z:z[0],reverse=True)
    if not candidates:return -1,None,'no_bbox_match'
    if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.1:return -1,None,'ambiguous_bbox_match'
    return candidates[0][1],candidates[0][2],'selected'
def writecsv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(args):
    with open(str(args.manifest or HERE/'review_template.csv'),encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if args.label:rows=[r for r in rows if r['label']==args.label]
    meta=json.loads((DATA/'video_inventory.json').read_text());out=HERE/'qa'/args.run;out.mkdir(parents=True,exist_ok=True)
    summaries=[];index=[];canvas=None
    for i,r in enumerate(rows):
        folder=DATA/(args.run+'_body25')/r['record_id'];paths=sorted(folder.glob('*_keypoints.json'))
        if not paths:continue
        boxes={b['frame']:b['bbox'] for b in json.loads((DATA/'annotations_by_sequence'/(r['record_id']+'.json')).read_text())}
        vi=json.loads(r['annotation_video_info']);m=meta[r['video_id']];sx=m['width']/vi['width'];sy=m['height']/vi['height']
        selections=[];conf=[];nzero=[];selected={};reasons={}
        for p in paths:
            frame=int(re.search(r'_(\d+)_keypoints',p.name).group(1));b=boxes.get(frame)
            if b is None:continue
            box=[b['left']*sx,b['top']*sy,b['width']*sx,b['height']*sy]
            person,a,reason=select(json.loads(p.read_text())['people'],box)
            selections.append(dict(filename=p.name,person_index=person));reasons[reason]=reasons.get(reason,0)+1
            conf.append(a[REQUIRED,2] if a is not None else np.zeros(8))
            nzero.append((a[REQUIRED,:2]==0).any(1) if a is not None else np.ones(8,dtype=bool))
            selected[frame]=(person,a,box)
        writecsv(folder/'selection.csv',selections)
        c=np.asarray(conf);z=np.asarray(nzero);jumps=[]
        for frame,(person,a,box) in sorted(selected.items()):
            previous=selected.get(frame-1)
            if a is None or previous is None or previous[1] is None:continue
            b=previous[1];valid=(a[REQUIRED,2]>=.1)&(b[REQUIRED,2]>=.1)
            if valid.any():jumps.append((float(np.linalg.norm(a[REQUIRED,:2][valid]-b[REQUIRED,:2][valid],axis=1).max()/box[3]),frame))
        largest=max(jumps) if jumps else (0,None)
        summaries.append(dict(record_id=r['record_id'],video_id=r['video_id'],view=r['view'],label=r['label'],frames=len(paths),selected=sum(s['person_index']>=0 for s in selections),unmatched=reasons.get('no_bbox_match',0),ambiguous=reasons.get('ambiguous_bbox_match',0),max_required_zero_fraction=float(z.mean(0).max()),mean_required_confidence=float(c.mean()),low_confidence_fraction=float((c<.1).mean()),mean_toe_confidence=float(c[:,[6,7]].mean()),bbox_height_pixels=float(np.median([v[2][3] for v in selected.values()])),max_joint_jump_bbox_height=largest[0],jump_at_frame=largest[1]))
        idx=len(summaries)-1
        if idx%12==0:canvas=np.full((4*235,3*640,3),245,dtype=np.uint8)
        x=(idx%3)*640;y=((idx%12)//3)*235
        cv2.putText(canvas,'%03d %s %s'%(i+1,r['video_id'],r['view']),(x+4,y+18),0,.46,(0,0,0),1,cv2.LINE_AA)
        frames=sorted(selected);sample=[frames[len(frames)//4],frames[3*len(frames)//4]]
        cap=cv2.VideoCapture(m['path'])
        for j,frame in enumerate(sample):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,im=cap.read()
            if not ok:continue
            person,a,b=selected[frame];bx,by,bw,bh=b
            cv2.rectangle(im,(int(bx),int(by)),(int(bx+bw),int(by+bh)),(0,255,0),1)
            if a is not None:
                for l,k in EDGES:
                    if min(a[l,2],a[k,2])>=.1:cv2.line(im,tuple(a[l,:2].astype(int)),tuple(a[k,:2].astype(int)),(255,0,255),2)
                for k in REQUIRED:
                    if a[k,2]>=.1:cv2.circle(im,tuple(a[k,:2].astype(int)),3,(0,0,255) if k in [9,10,11,22] else (255,255,0),-1)
            scale=min(320/im.shape[1],180/im.shape[0]);im=cv2.resize(im,(int(im.shape[1]*scale),int(im.shape[0]*scale)))
            dx=x+j*320+(320-im.shape[1])//2;dy=y+25+(180-im.shape[0])//2;canvas[dy:dy+im.shape[0],dx:dx+im.shape[1]]=im
            cv2.putText(canvas,'frame %s person %s'%(frame,person),(x+j*320+3,y+225),0,.43,(0,0,0),1)
        cap.release();index.append(dict(index=i+1,record_id=r['record_id'],page=idx//12+1))
        if idx%12==11:cv2.imwrite(str(out/('sheet_%02d.jpg'%(idx//12+1))),canvas)
    if canvas is not None:cv2.imwrite(str(out/('sheet_%02d.jpg'%((len(summaries)-1)//12+1))),canvas)
    writecsv(out/'quality.csv',summaries);(out/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps(dict(records=len(summaries),frames=sum(r['frames'] for r in summaries),output=str(out))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='pilot');p.add_argument('--manifest');p.add_argument('--label');main(p.parse_args())
