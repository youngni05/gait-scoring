import sys,json
from pathlib import Path
S=Path(__file__).parent; sys.path.insert(0,str(S/'deps'))
import cv2
from PIL import Image,ImageDraw
meta=[]
for cam in ['03','06']:
 cap=cv2.VideoCapture(str(S/f'camera{cam}.mp4')); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=cap.get(cv2.CAP_PROP_FRAME_WIDTH);h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
 sheet=Image.new('RGB',(1000,690),'white'); draw=ImageDraw.Draw(sheet)
 for j in range(12):
  k=round(j*(n-1)/11);cap.set(cv2.CAP_PROP_POS_FRAMES,k); ok,bgr=cap.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((245,195)); x=(j%4)*250;y=(j//4)*230;sheet.paste(im,(x,y+25));draw.text((x+3,y+3),f'{cam} frame {k}, {k/fps:.2f}s',fill='black')
 sheet.save(S/f'contact{cam}.jpg');cap.release();meta.append(dict(camera=cam,fps=fps,frames=n,width=w,height=h,duration=n/fps))
(S/'video_metadata.json').write_text(json.dumps(meta,indent=2));print(meta)
