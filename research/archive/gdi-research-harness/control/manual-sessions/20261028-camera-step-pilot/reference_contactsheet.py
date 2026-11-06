import sys
from pathlib import Path
S=Path(__file__).parent;sys.path.insert(0,str(S/'deps'))
import cv2
from PIL import Image,ImageDraw
cap=cv2.VideoCapture(str(S/'camera03.mp4'))
for part,start in enumerate([70,150,230]):
 sheet=Image.new('RGB',(1280,1000),'white');draw=ImageDraw.Draw(sheet)
 for j,k in enumerate(range(start,start+80,5)):
  cap.set(cv2.CAP_PROP_POS_FRAMES,k);ok,bgr=cap.read()
  if not ok:continue
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((140,230,540,520));im.thumbnail((315,220));x=j%4*320;y=j//4*250;sheet.paste(im,(x,y+25));draw.text((x+5,y+5),f'frame{k} t={k/50:.2f}',fill='black')
 sheet.save(S/f'contact_reference{part}.jpg')
cap.release()
