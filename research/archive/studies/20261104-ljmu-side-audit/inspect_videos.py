import concurrent.futures, gzip, json, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=Path('D:/gait-research-data/D_P/ljmu-side-audit-20261104')
sys.path.insert(0,str(ROOT/'tmp/ljmu-video-python-deps'))
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'))
sys.path.insert(0,str(ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'))
import av
from preprocess import extract_windows

def inspect(p):
    out=HERE/'checks'/f'{p.stem}.json'
    if out.exists():
        cached=json.loads(out.read_text())
        pose_available=(DATA/'poses'/f'{p.stem}_json.gz').exists() or (p.stem=='P8_E2T6_90_ALT' and (DATA/'poses'/'P8_E2T6_90_json.gz').exists())
        if cached['pose_status']!='missing' or not pose_available:return cached
    d={'name':str(p.relative_to(DATA/'files')).replace('\\','/'),'id':p.stem,
       'person':int(re.match(r'P(\d+)_',p.stem).group(1)),
       'experiment':int(re.search(r'_E(\d+)',p.stem).group(1)),
       'decode_ok':False,'pose_status':'missing','selected_start':None}
    pose=DATA/'poses'/f'{p.stem}_json.gz'
    if not pose.exists() and p.stem=='P8_E2T6_90_ALT': pose=DATA/'poses'/'P8_E2T6_90_json.gz'
    try:
        c=av.open(str(p));s=c.streams.video[0]
        d.update(fps=float(s.average_rate),fps_fraction=str(s.average_rate),width=s.width,height=s.height,header_frames=s.frames)
        raw=None
        if pose.exists():
            rows=json.load(gzip.open(pose,'rt'))
            d['pose_file']=pose.name
            d['pose_frames']=len(rows)
            d['frame_ids_contiguous']=[x['frame'] for x in rows]==list(range(len(rows)))
            raw=np.full((len(rows),75),np.nan)
            conf_errors=0;multi=0
            for i,r in enumerate(rows):
                pp=r['data']['people']
                if len(pp)==1:
                    a=np.asarray(pp[0]['pose_keypoints_2d'])
                    if a.shape!=(75,) or not np.isfinite(a).all(): raise ValueError('Invalid BODY25 shape/value')
                    conf_errors+=int(((a[2::3]<0)|(a[2::3]>1)).any())
                    raw[i]=a
                elif len(pp)>1: multi+=1
            d.update(confidence_error_frames=conf_errors,multiple_detection_frames=multi)
            d['pose_status']='confidence_out_of_range' if conf_errors else 'valid'
            if not conf_errors:
                try:
                    x=extract_windows(raw)
                    d['baseline_windows']=[int(v) for v in x['window_start']]
                    d['baseline_missing']=[float(v) for v in x['window_missing_fraction']]
                    # Retain the frozen preprocessing schedule, then prefer the
                    # candidate with most observed in-frame lower-body coordinates.
                    quality=[]
                    for k in d['baseline_windows']:
                        a=raw[k:k+124].reshape(124,25,3)
                        pts=a[:,[1,9,10,11,12,13,14,19,22],:2]
                        valid=np.isfinite(pts).all(axis=(1,2))&(pts[:,:,0]>8).all(axis=1)&(pts[:,:,0]<s.width-8).all(axis=1)&(pts[:,:,1]>8).all(axis=1)&(pts[:,:,1]<s.height-8).all(axis=1)
                        hip=a[:,8,0]
                        good=np.flatnonzero(np.isfinite(hip)&(hip>0))
                        travel=float(abs(np.median(hip[good[-10:]])-np.median(hip[good[:10]]))) if len(good)>=20 else 0
                        quality.append((float(valid.mean()),travel,k))
                    quality.sort(reverse=True)
                    frac,travel,k=quality[0]
                    d.update(selected_start=k,observed_inframe_fraction=frac,pelvis_travel_px=travel)
                except (ValueError,IndexError) as ex: d['pose_status']='preprocess_excluded';d['pose_reason']=str(ex)
        n=s.frames
        mid=(d['selected_start']+62) if d['selected_start'] is not None else n//2
        want=set([max(0,int(n*q)) for q in [.15,.5,.85]]+[mid])
        frames={};ts=[]
        for i,f in enumerate(c.decode(video=0)):
            ts.append(f.time)
            if i in want: frames[i]=f.to_image()
        c.close()
        d.update(decoded_frames=len(ts),duration_seconds=(ts[-1]-ts[0]+1/d['fps']) if ts else 0,
                 monotonic_timestamps=all(b>a for a,b in zip(ts,ts[1:])),decode_ok=True)
        if len(ts)>1:d['timestamp_step_min']=min(np.diff(ts));d['timestamp_step_max']=max(np.diff(ts))
        if raw is not None:d['video_pose_count_match']=len(raw)==len(ts)
        thumb=frames.get(mid,frames.get(n//2))
        if thumb is not None:
            if raw is not None and mid<len(raw):
                pts=raw[mid].reshape(25,3)
                pts=pts[np.isfinite(pts).all(axis=1)&(pts[:,0]>0)&(pts[:,1]>0)]
                if len(pts)>6:
                    x0,y0=pts[:,:2].min(axis=0)-40;x1,y1=pts[:,:2].max(axis=0)+40
                    thumb=thumb.crop((max(0,x0),max(0,y0),min(s.width,x1),min(s.height,y1)))
            thumb.thumbnail((192,170))
            im=Image.new('RGB',(200,200),'white');im.paste(thumb,((200-thumb.width)//2,0));dr=ImageDraw.Draw(im)
            dr.text((2,174),p.stem,fill='black')
            dr.text((2,187),f'{len(ts)}f {d["fps"]:.2f}fps',fill='black')
            (HERE/'thumbs').mkdir(exist_ok=True)
            im.save(HERE/'thumbs'/f'{p.stem}.jpg',quality=85)
        # Full-scene temporal contact strip for detailed review and examples.
        strip=Image.new('RGB',(960,205),'white');dr=ImageDraw.Draw(strip)
        for j,q in enumerate([.15,.5,.85]):
            k=int(n*q)
            if k in frames:
                strip.paste(frames[k].resize((320,180)),(j*320,25))
                dr.text((j*320+5,5),f'{p.stem}  frame {k}',fill='black')
        (HERE/'strips').mkdir(exist_ok=True)
        strip.save(HERE/'strips'/f'{p.stem}.jpg',quality=85)
    except Exception as ex: d['error']=str(ex)
    (HERE/'checks').mkdir(exist_ok=True)
    out.write_text(json.dumps(d,indent=2),encoding='utf-8')
    return d

def main():
    ps=sorted((DATA/'files').rglob('*.mp4'))
    # Wait for matching pose acquisition rather than caching incomplete metadata.
    ps=[p for p in ps if (DATA/'poses'/f'{p.stem}_json.gz').exists() or p.stem in ['P56_E2T3_90','P8_E2T6_90_ALT']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,d in enumerate(pool.map(inspect,ps)):
            if (i+1)%50==0: print(f'inspected {i+1}/{len(ps)}',flush=True)
    print(f'pass complete {len(ps)}',flush=True)

if __name__=='__main__': main()
