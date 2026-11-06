"""Scientific source-frame comparison; no pose retouching or synthetic content."""
import sys,json
from pathlib import Path
sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
import cv2
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
frame=1484;sheet=Image.new('RGB',(1000,400),'white');draw=ImageDraw.Draw(sheet)
for i,cam in enumerate(['PG1','PG2']):
    p=Path('D:/gait-research-data/MoVi/raw')/f'F_{cam}_Subject_1_L.avi'
    cap=cv2.VideoCapture(str(p));cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,img=cap.read();cap.release();assert ok
    im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im=im.resize((500,375));sheet.paste(im,(i*500,0))
    draw.text((i*500+8,380),f'MoVi F | Subject 1 | {cam} | source frame {frame}',fill='black')
sheet.save(HERE/'qa/movi-camera-example.jpg',quality=90)
(HERE/'visual-review.json').write_text(json.dumps({'review_date':'2026-11-05','reviewed_contact_sheets':[f'qa/movi-contact-{i}.jpg' for i in range(1,5)],'representative_frames_seen':87,'review_scope':'One representative frame per available F/PG1 subject, not all-frame occlusion or complete gait-cycle validation','selection_changed_by_visual_review':False,'example_source_frames':{'person':1,'cameras':['PG1','PG2'],'one_based_frame':1484}},indent=2)+'\n')
