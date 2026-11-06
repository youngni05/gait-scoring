"""Inventory Test01 by filename number, with no visual label inference."""
import sys,json,re,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import cv2,numpy as np
from PIL import Image,ImageDraw
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=list((ROOT/'영상자료/검증용/Test01').glob('*.mp4'))
    files.sort(key=lambda p:int(re.search(r'\((\d+)\)',p.stem).group(1)))
    rows=[]
    for p in files:
        number=int(re.search(r'\((\d+)\)',p.stem).group(1));clip='v%02d'%number
        cap=cv2.VideoCapture(str(p));assert cap.isOpened();n=int(cap.get(7));fps=cap.get(5)
        count=min(n,1499);canvas=Image.new('RGB',(1280,750));draw=ImageDraw.Draw(canvas)
        for j,f in enumerate(np.rint(np.linspace(0,count-1,8)).astype(int)):
            cap.set(1,int(f));ok,im=cap.read();assert ok,(clip,int(f))
            img=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));img.thumbnail((315,340));x=j%4*320;y=j//4*375
            canvas.paste(img,(x,y+25));draw.text((x,y),'%s frame %d %.2fs'%(clip,f,f/fps),fill='white')
        canvas.save(HERE/(clip+'-contact.jpg'))
        row=dict(number=number,id=clip,file=str(p),filename=p.name,bytes=p.stat().st_size,sha256=sha(p),fps=fps,frames=n,width=int(cap.get(3)),height=int(cap.get(4)),seconds=n/fps,processed_frames=count,label=None)
        cap.release();rows.append(row);print(json.dumps({k:row[k] for k in ['number','fps','frames','width','height','seconds','processed_frames']}),flush=True)
    assert len(rows)==13 and [r['number'] for r in rows]==list(range(1,14))
    (HERE/'inventory.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf8')
if __name__=='__main__':main()
