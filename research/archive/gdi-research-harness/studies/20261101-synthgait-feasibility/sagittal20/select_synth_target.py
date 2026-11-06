"""Explicit audited selection for the one video with background mannequins."""
from pathlib import Path
import csv,json,hashlib
import cv2,numpy as np
ROOT=Path('D:/Projects/D_P');HERE=ROOT/'gdi-research-harness/studies/20261101-synthgait-feasibility/sagittal20'
DATA=Path('D:/gait-research-data/D_P/synthgait19k-review/sagittal20');ID='vid_00009_saggital'
def main():
    cv2.setNumThreads(2);out=DATA/'body25_selected';out.mkdir(exist_ok=True);raw=[];counts=[];rows=[];panels=[]
    cap=cv2.VideoCapture(str(DATA/(ID+'.mp4')))
    for frame,p in enumerate(sorted((DATA/'body25'/ID).glob('*_keypoints.json'))):
        people=json.loads(p.read_text())['people'];candidates=[]
        for i,person in enumerate(people):
            kp=np.array(person['pose_keypoints_2d']).reshape(25,3);xy=kp[kp[:,2]>.1,:2]
            if len(xy)<3:continue
            a=xy.min(axis=0);b=xy.max(axis=0);candidates.append((float(b[1]-a[1]),i,kp,a,b))
        candidates.sort(key=lambda c:c[0],reverse=True);height,i,kp,a,b=candidates[0];gap=height-candidates[1][0]
        raw.append(kp);counts.append(len(people));rows.append(dict(filename=p.name,person_index=i,frame=frame,
            bbox_height=height,height_margin_to_next=gap,source_json_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        ok,im=cap.read();assert ok
        cv2.rectangle(im,tuple(a.astype(int)),tuple(b.astype(int)),(0,255,0),2)
        cv2.putText(im,'frame %d target %d'%(frame,i),(5,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
        panels.append(cv2.resize(im,(208,120)))
    cap.release();assert len(raw)==81
    selected=out/(ID+'.npz');np.savez_compressed(str(selected),raw=np.stack(raw),frame=np.arange(81),time_seconds=np.arange(81)/16.,detected_people=np.array(counts))
    csvpath=HERE/'target-selection.csv'
    with csvpath.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atlas=np.concatenate([np.concatenate(panels[i:i+9],axis=1) for i in range(0,81,9)],axis=0)
    cv2.imwrite(str(DATA/'target_selection_81frames.png'),atlas)
    # Keep original unique-only failure evidence and raw JSON intact.
    pose=json.loads((HERE/'pose-verification.json').read_text(encoding='utf-8'))
    for row in pose['records']:
        if row['sample_id']==ID:
            row['raw_path']=str(selected);row['explicitly_selected_frames']=81;row['selection_csv']=str(csvpath)
    pose['selection_note']='Only vid_00009: foreground walker vs background mannequins; explicit per-frame CSV, original unique-only output preserved.'
    (HERE/'pose-verification.selected.json').write_text(json.dumps(pose,indent=2),encoding='utf-8')
    centers=np.array([k[[9,12],:2].mean(axis=0) for k in raw])
    report={'sample_id':ID,'frames':81,'selected_path':str(selected),'selection_csv':str(csvpath),
        'selection_sha256':hashlib.sha256(csvpath.read_bytes()).hexdigest(),'min_height_margin_pixels':min(r['height_margin_to_next'] for r in rows),
        'max_hip_center_step_pixels':float(np.linalg.norm(np.diff(centers,axis=0),axis=1).max()),
        'rule':'Tallest >0.1-confidence pose bounding box, after inspecting frames0/20/40/60/80; inspect all81-frame atlas before accepting.',
        'atlas':str(DATA/'target_selection_81frames.png'),'visually_reviewed_all_frames':False,'original_unique_only_outputs_preserved':True}
    (HERE/'target-selection-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
