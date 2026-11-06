import json, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import cv2
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent
rows=[]
for j,p in enumerate(sorted((ROOT/'영상자료/검증용').glob('*.mp4'))):
    c=cv2.VideoCapture(str(p)); n=int(c.get(7)); fps=c.get(5)
    rows.append(dict(id='clip%d'%j,file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fps=fps,frames=n,width=int(c.get(3)),height=int(c.get(4)),seconds=n/fps))
    canvas=Image.new('RGB',(1600,1000));draw=ImageDraw.Draw(canvas)
    for k in range(15):
        frame=round((n-1)*k/14);c.set(1,frame);ok,im=c.read();assert ok
        im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im.thumbnail((318,300))
        x=(k%5)*320;y=(k//5)*333
        canvas.paste(im,(x,y+25));draw.text((x,y),'%s frame %d / %.2fs'%(rows[-1]['id'],frame,frame/fps),fill='white')
    canvas.save(HERE/('contact%d.jpg'%j));c.release()
(HERE/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(rows,ensure_ascii=False))
