"""Reuse Full's frozen preprocessing; additionally compare Lite against Full."""
from common import *
import cv2
from PIL import Image,ImageDraw

def overlay(row,lite,full,valid):
    starts=read(OLD/'protocol.json')['straight_starts'][row['id']]
    ids=[int(f) for s in starts for f in np.rint(np.linspace(s,s+123,7)).astype(int)]
    canvas=Image.new('RGB',(1960,1590),(20,20,20));draw=ImageDraw.Draw(canvas);cap=cv2.VideoCapture(row['file'])
    for ix,f in enumerate(ids):
        cap.set(1,f);ok,bgr=cap.read();assert ok
        image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));image.thumbnail((276,490));pen=ImageDraw.Draw(image)
        sx=image.width/row['width'];sy=image.height/row['height']
        for a,color in [(full[f].reshape(25,3),(240,70,210)),(lite[f].reshape(25,3),(40,210,255))]:
            for chain in [[12,13,14,19],[9,10,11,22]]:
                for j,k in zip(chain[:-1],chain[1:]):
                    if valid(a[j]) and valid(a[k]):pen.line((a[j,0]*sx,a[j,1]*sy,a[k,0]*sx,a[k,1]*sy),fill=color,width=2)
                for j in chain:
                    if valid(a[j]):
                        x,y=a[j,0]*sx,a[j,1]*sy;pen.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        x=ix%7*280;y=ix//7*530;canvas.paste(image,(x,y+35))
        draw.text((x+3,y+3),f'{row["id"]} frame {f}',fill='white');draw.text((x+3,y+18),'Full magenta / Lite cyan',fill='white')
    cap.release();canvas.save(HERE/(row['id']+'-full-lite.jpg'),quality=92)
    return ids

def main():
    worker=module('reuse_full_preparation',FULL/'prepare_compare.py')
    # Imported common points to the isolated Lite study; source bundle stays frozen.
    assert worker.HERE==HERE and worker.BIG==BIG
    worker.main()
    rows=[];frames=[]
    for row in read(OLD/'inventory.json'):
        clip=row['id'];lite=np.load(BIG/(clip+'-pose.npz'));full=np.load(FULL_BIG/(clip+'-pose.npz'))
        assert np.array_equal(lite['source_ms'],full['source_ms']) and np.array_equal(lite['task_ms'],full['task_ms'])
        c=worker.compare(lite['raw'],full['raw'],lite['landmarks'])
        # compare() names its first input mediapipe and reference input openpose;
        # clarify the roles to avoid treating Full as ground truth.
        rows.append(dict(clip=clip,reference='MediaPipe Full',candidate='MediaPipe Lite',metrics=c))
        frames.append(dict(clip=clip,frames=overlay(row,lite['raw'],full['raw'],worker.valid)))
    save(HERE/'full-lite-pose.json',dict(reference='Full',candidate='Lite',legacy_metric_keys={'openpose':'Full','mediapipe':'Lite'},
        scale='mean Full hip-knee-ankle chain length',ground_truth=False,clips=rows,review_frames=frames))
    print([(r['clip'],r['metrics']['pooled_pixel_distance']) for r in rows],flush=True)

if __name__=='__main__':main()
