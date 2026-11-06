"""Frozen feature preparation, landmark disagreement, and score-blind overlays."""
import warnings
from common import *
import cv2
from PIL import Image, ImageDraw

PREP = module('original_feature_preparation', OLD/'prepare_features.py')

def distribution(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return dict(n=int(a.size), median=float(np.median(a)), p90=float(np.percentile(a,90)), mean=float(a.mean())) if a.size else dict(n=0)

def valid(points):
    return np.isfinite(points[...,:2]).all(-1) & (points[...,:2]!=0).all(-1)

def compare(raw, op, landmarks):
    mp = raw.reshape(-1,25,3); op = op.reshape(-1,25,3)
    lengths = []
    for hip,knee,ankle in [(12,13,14),(9,10,11)]:
        v = np.linalg.norm(op[:,hip,:2]-op[:,knee,:2],axis=1)+np.linalg.norm(op[:,knee,:2]-op[:,ankle,:2],axis=1)
        v[~valid(op[:,[hip,knee,ankle]]).all(1)] = np.nan
        lengths.append(v)
    scale = (lengths[0]+lengths[1])/2
    result = {}; all_dist=[]; all_norm=[]
    for name,j in JOINTS.items():
        good=valid(op[:,j]) & valid(mp[:,j])
        distance=np.linalg.norm(op[:,j,:2]-mp[:,j,:2],axis=1)
        normal=good & np.isfinite(scale) & (scale>0)
        visibility=landmarks[:,MAP[j],3]
        result[name]=dict(pixel_distance=distribution(distance[good]),leg_length_fraction=distribution(distance[normal]/scale[normal]),
            openpose_missing_frames=int((~valid(op[:,j])).sum()),mediapipe_missing_frames=int((~valid(mp[:,j])).sum()),
            mediapipe_visibility_below_half_frames=int((visibility<.5).sum()),visibility=distribution(visibility))
        all_dist.extend(distance[good]);all_norm.extend(distance[normal]/scale[normal])
    left=[12,13,14]; right=[9,10,11]
    good=valid(op[:,left+right]).all(1)&valid(mp[:,left+right]).all(1)
    same=np.linalg.norm(op[:,left+right,:2]-mp[:,left+right,:2],axis=2).mean(1)
    swap=np.linalg.norm(op[:,left+right,:2]-mp[:,right+left,:2],axis=2).mean(1)
    return dict(joints=result,pooled_pixel_distance=distribution(all_dist),pooled_leg_length_fraction=distribution(all_norm),
                left_right_diagnostic=dict(jointly_observed_frames=int(good.sum()),swapped_correspondence_closer_frames=int(((swap<same)&good).sum()),
                    interpretation='Alternative correspondence distance only; neither estimator is ground truth; no actual swap applied'))

def overlay(row, raw, op):
    starts=read(OLD/'protocol.json')['straight_starts'][row['id']]
    frames=[int(f) for s in starts for f in np.rint(np.linspace(s,s+123,7)).astype(int)]
    canvas=Image.new('RGB',(7*280,3*530),(20,20,20));draw=ImageDraw.Draw(canvas)
    cap=cv2.VideoCapture(row['file'])
    for ix,f in enumerate(frames):
        cap.set(1,f);ok,bgr=cap.read();assert ok
        img=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));img.thumbnail((276,490))
        pen=ImageDraw.Draw(img);sx=img.width/row['width'];sy=img.height/row['height']
        for a,color in [(op[f].reshape(25,3),(50,255,80)),(raw[f].reshape(25,3),(255,70,220))]:
            for chain in [[12,13,14,19],[9,10,11,22]]:
                for j,k in zip(chain[:-1],chain[1:]):
                    if valid(a[j]) and valid(a[k]):pen.line((a[j,0]*sx,a[j,1]*sy,a[k,0]*sx,a[k,1]*sy),fill=color,width=2)
                for j in chain:
                    if valid(a[j]):
                        x,y=a[j,0]*sx,a[j,1]*sy;pen.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
        x=(ix%7)*280;y=(ix//7)*530
        canvas.paste(img,(x,y+35));draw.text((x+3,y+4),f'{row["id"]} frame {f}',fill='white')
        draw.text((x+3,y+18),'OP green / MP magenta',fill='white')
    cap.release();canvas.save(HERE/(row['id']+'-comparison.jpg'),quality=92)
    return frames

def main():
    ns=PREP.source_functions();xs=[];windows=[];quality=[];comparisons=[];reviews=[];baseline_error=0.
    requested=read(OLD/'windows.json');previous=np.load(OLD_BIG/'features.npz')['x']
    for row in read(OLD/'inventory.json'):
        clip=row['id'];z=np.load(BIG/(clip+'-pose.npz'));raw=z['raw'];op=np.load(OLD_BIG/(clip+'-raw.npz'))['raw']
        assert raw.shape==op.shape==(row['frames'],75)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            processed=ns['process_video_and_add_cols'](raw,swap_orientation=False)
            prior=ns['process_video_and_add_cols'](op,swap_orientation=False)
        accepted=[];rejected=[]
        for i,w in enumerate(requested):
            if w['clip']!=clip:continue
            start=w['start'];end=w['end']+1
            check=np.stack([prior[start:end,PREP.LEFT],prior[start:end,PREP.RIGHT]]).astype(np.float32)
            baseline_error=max(baseline_error,float(np.abs(check-previous[i]).max()))
            missing=float(ns['max_pct_nan_or_zero_given_cols'](ns['drop_confidence_cols'](raw[start:end]),PREP.REQUIRED))
            x=np.stack([processed[start:end,PREP.LEFT],processed[start:end,PREP.RIGHT]]).astype(np.float32)
            record=dict(clip=clip,kind=w['kind'],start=start,end=end-1,missing=missing)
            if missing<=.25 and np.isfinite(x).all():xs.append(x);windows.append(record);accepted.append(record)
            else:rejected.append(dict(**record,nonfinite_features=not bool(np.isfinite(x).all())))
        quality.append(dict(clip=clip,frames=row['frames'],accepted=accepted,rejected=rejected))
        comparisons.append(dict(clip=clip,**compare(raw,op,z['landmarks'])))
        reviews.append(dict(clip=clip,frames=overlay(row,raw,op)))
    assert baseline_error==0, baseline_error
    np.savez_compressed(BIG/'features.npz',x=np.stack(xs))
    save(HERE/'windows.json',windows);save(HERE/'quality.json',quality)
    save(HERE/'pose-comparison.json',dict(clips=comparisons,units='source-image pixels; normalized by mean OpenPose hip-knee-ankle chain length',ground_truth=False))
    save(HERE/'preparation.json',dict(baseline_feature_reconstruction_max=baseline_error,requested_windows=len(requested),accepted_windows=len(windows),
        review_frames=reviews,features_sha256=sha(BIG/'features.npz')))
    print('FEATURES',len(windows),'/',len(requested),'baseline_replay',baseline_error,flush=True)
    for c in comparisons:print(c['clip'],c['pooled_pixel_distance'],c['pooled_leg_length_fraction'],flush=True)

if __name__=='__main__':main()
