"""Render source images with selected BODY25 targets for visual quality review."""
from pathlib import Path
import json,csv,argparse
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
PAIRS=[(1,8),(8,9),(9,10),(10,11),(8,12),(12,13),(13,14),(11,22),(14,19)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quality',default='quality.json');ap.add_argument('--prefix',default='target');args=ap.parse_args()
    rows=json.loads((HERE/args.quality).read_text());out=HERE/'qa';out.mkdir(exist_ok=True)
    for page in range(0,len(rows),6):
        canvas=np.full((6*170,5*320,3),255,np.uint8)
        for line,row in enumerate(rows[page:page+6]):
            stem=row['record'];folder=HERE/'body25'/stem
            picks={int(r['filename'].split('_')[-2]):int(r['person_index']) for r in csv.DictReader((folder/'selection.csv').open())}
            visible=[f for f,s in picks.items() if s>=0 and f<500]
            best=max(row['scheduled_windows'],key=lambda w:w['movement_fraction'],default={'start':0})['start']+62
            frames=[0,min(124,row['frames']-1),min(best,row['frames']-1),visible[-1] if visible else min(248,row['frames']-1),row['multiple_eligible_frames'][0] if row['multiple_eligible_frames'] else min(372,row['frames']-1)]
            cap=cv2.VideoCapture(str(HERE/'td_sagittal'/(stem+'.MOV')))
            for col,frame in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
                if not ok:raise ValueError((stem,frame))
                obj=json.loads((folder/('%s_%012d_keypoints.json'%(stem,frame))).read_text())
                for index,person in enumerate(obj['people']):
                    a=np.array(person['pose_keypoints_2d']).reshape(25,3);color=(0,255,0) if index==picks[frame] else (0,0,255)
                    for first,last in PAIRS:
                        if min(a[first,2],a[last,2])>.1:cv2.line(im,tuple(a[first,:2].astype(int)),tuple(a[last,:2].astype(int)),color,3)
                im=cv2.resize(im,(320,150));canvas[line*170+20:line*170+170,col*320:(col+1)*320]=im
                cv2.putText(canvas,'%s %s f%d sel%d'%(row['person'],stem.split('-')[-2],frame,picks[frame]),(col*320+3,line*170+15),0,.45,(0,0,0),1)
            cap.release()
        cv2.imwrite(str(out/('%s_%02d.jpg'%(args.prefix,page))),canvas)
    print('review_pages',len(range(0,len(rows),6)))
if __name__=='__main__':main()
