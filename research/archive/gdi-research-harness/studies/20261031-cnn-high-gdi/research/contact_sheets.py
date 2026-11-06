"""Source-video visual checks independent of predicted GDI scores."""
from pathlib import Path
import cv2,numpy as np
HERE=Path(__file__).resolve().parent
paths=sorted((HERE/'td_sagittal').glob('*.MOV'))
out=HERE/'qa';out.mkdir(exist_ok=True)
for page in range(0,len(paths),6):
    canvas=np.full((6*150,5*256,3),255,np.uint8)
    for row,path in enumerate(paths[page:page+6]):
        cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for col,frame in enumerate([0,min(n-1,124),min(n-1,248),min(n-1,372),min(n-1,496)]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read()
            if not ok:continue
            im=cv2.resize(im,(256,144));cv2.putText(im,path.name.split('-')[0]+' '+path.name.split('-')[-2]+' f'+str(frame),(4,15),0,.45,(0,0,255),1)
            canvas[row*150:row*150+144,col*256:(col+1)*256]=im
        cap.release()
    cv2.imwrite(str(out/('source_%02d.jpg'%page)),canvas)
print('pages',len(list(out.glob('source*.jpg'))))
