"""Create identical-time RGB and numerical inputs without loading clinical labels."""
import json
import sys
import time
from pathlib import Path
from bootstrap import HERE,ROOT,BIG,sha
CAMERA=ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot'
sys.path.insert(0,str(CAMERA/'deps'))
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image,ImageDraw

PROMPTS={
 'focused':'Describe only the visible movement of the main participant in the center of this video. '
 'Report how the feet advance, whether steps alternate, any visible pause or turn, arm movement, trunk movement, '
 'and any walking aid actually being used. Separate visible evidence from what cannot be seen. '
 'Do not describe room decorations, clothing, age, sex, diagnosis, disability, risk, test scores or numerical speed. '
 'Do not invent subtle movements. A person beside the walking path may be an observer. Use at most five short sentences.',
 'generic':'Describe the actions of the main person in the center of this video in at most five short sentences. '
 'Describe visible actions only, without age, sex, diagnosis, scores, risk, clothing or room decoration.'}

def save(n,v):(HERE/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def select_window(path,fps):
    cap=cv2.VideoCapture(str(path));duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    times=np.arange(min(30,int(duration)-1)+1);quality=[];positions=[]
    opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(CAMERA/'pose_landmarker_full.task')),
          running_mode=vision.RunningMode.IMAGE,num_poses=3)
    with vision.PoseLandmarker.create_from_options(opts) as detector:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(float(t)*fps));good,bgr=cap.read();assert good
            h,w=bgr.shape[:2];rgb=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(round(w*768/h),768))
            result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
            candidates=[p for p in result.pose_landmarks if min(p[23].visibility,p[24].visibility)>=.5]
            if candidates:
                p=min(candidates,key=lambda p:abs((p[23].x+p[24].x)/2-.5))
                quality.append(min(p[j].visibility for j in [11,12,23,24,25,26,27,28])>=.5)
                positions.append([(p[23].x+p[24].x)/2,(p[23].y+p[24].y)/2])
            else:quality.append(False);positions.append([np.nan,np.nan])
    cap.release();scores=[]
    for start in range(len(times)-5):
        pos=np.array(positions[start:start+6]);movement=float(np.ptp(pos[np.isfinite(pos).all(1)],axis=0).max()) if np.isfinite(pos).all(1).sum()>1 else 0.
        scores.append(sum(quality[start:start+6])+(1 if movement>.03 else 0))
    assert scores
    first=int(np.argmax(scores))
    return float(times[first]),{'coarse_seconds':times.tolist(),'full_body_visible':quality,'selected_score':scores[first],
                               'rule':'earliest maximum over6 one-second checks: visible8joints count plus1 if hip movement range>.03; no clinical labels or captions'}

def main():
    receipt=json.loads((HERE/'acquisition.json').read_text())
    protocol={'declared_before_captions_and_predictions':True,'people':12,
      'source':'Toronto bottom-camera raw walking videos; same12 people as prior pose study, noGDI labels here',
      'window':'4.1seconds from earliest maximum visibility/motion candidate in first30seconds;6 one-second checks, eightjoints visible>=.5 count plus1 for hiprange>.03;124target timestamps at30Hz mapped to nearestnative frame; no label/caption selection; repeatedframes possible below30fps',
      'preflight_revision':'Fixed10s window caused many missingframes in initial5video decoding check; visibility-based selection fixed before any caption or target evaluation.',
      'numeric':'New MediaPipe extraction on EXACT same RGB window;17COCO joints XY/conf; per-clip valid bounding-box normalization; mean/std/mean absolute adjacent change=153features. Same summary strategy as prior study but NOT identical old Alphapose inputs.',
      'target_selection':'Among detected poses, closest hip midpoint to horizontal image center; reject hip invisibility; no disease/score input',
      'caption_model':receipt['model'],'model_revision':receipt['revision'],'model_execution':'localCPU; pretrained weights frozen; no raw video sent to externalAPI',
      'caption_input':'9 equally spaced actualRGB frames per4.1s clip, chronological interleaved-image prompt with timestamps; no derived numerical features or clinical labels supplied; not full-frame-rate video',
      'caption_variants':['focused','generic','focused with first frame repeated9times (static control)'],
      'prompts':PROMPTS,'caption_decoding':'greedy do_sampleFalse max_new_tokens160; no prompt tuning from test performance',
      'Jev':'1.14; text captions only; movement questions plus support/pause/turn/clearance; up to60requests and200000inputtokens',
      'predictions':'existing Toronto5fold people splits; fixedRidge alpha100,RF128 leaf2 seed20261031; same training/test people for every arm; missing/scaling trainonly',
      'comparisons':['numeric','Jev numeric summary','Jev caption only','numeric+Jev numeric summary','numeric+Jev focused caption','numeric+Jev generic caption','numeric+Jev static caption','numeric+shuffled Jev focused caption'],
      'shuffle':'50seeded permutations of caption features within training and test partitions separately; labels never used; diagnostic negative control',
      'metrics':['personMAE','RMSE','conditional paired bootstrap95','perperson errors','caption evidence audit'],
      'interpretation':'12person exploratory study; require paired prediction benefit AND visually supported observations, not merely longer prose; no universalVLM conclusion',
      'storage':str(BIG),'GPU_training':False}
    if (HERE/'protocol.json').exists() and (HERE/'captions.jsonl').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    else:save('protocol.json',protocol)
    rows=[];feature_rows=[]
    indices=[0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]
    for item in receipt['videos']:
        pid=item['person'];path=Path(item['path']);assert sha(path)==item['sha256']
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS);assert 10<=fps<=120
        start_seconds,selection=select_window(path,fps)
        native_indices=np.rint((start_seconds+np.arange(124)/30)*fps).astype(int)
        frames=[];poses=[];raw_landmarks=[];pose_counts=[];last_index=-1;bgr=None
        opts=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(CAMERA/'pose_landmarker_full.task')),
          running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
        started=time.monotonic()
        with vision.PoseLandmarker.create_from_options(opts) as detector:
            for i in range(124):
                if native_indices[i]!=last_index:
                    if native_indices[i]!=last_index+1:cap.set(cv2.CAP_PROP_POS_FRAMES,int(native_indices[i]))
                    good,bgr=cap.read();assert good,f'short_video_{pid}'
                    last_index=native_indices[i]
                h,w=bgr.shape[:2];rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                # Same resized view for numerical extraction and captioning.
                rgb=cv2.resize(rgb,(round(w*768/h),768));frames.append(rgb)
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),round(i*1000/30))
                candidates=[p for p in result.pose_landmarks if min(p[23].visibility,p[24].visibility)>=.5]
                pose_counts.append(len(candidates))
                if candidates:
                    p=min(candidates,key=lambda p:abs((p[23].x+p[24].x)/2-.5))
                    arr=np.array([[p[j].x,p[j].y,p[j].visibility] for j in indices])
                    arr[arr[:,2]<.5]=0;poses.append(arr)
                    raw_landmarks.append([[q.x,q.y,q.visibility] for q in p])
                else:poses.append(np.zeros((17,3)));raw_landmarks.append(None)
        cap.release();frames=np.stack(frames);x=np.stack(poses)
        valid=x[:,:,2]>0
        xy=x[:,:,:2]
        if valid.any():
            low=xy[valid].min(0);high=xy[valid].max(0);scale=max(high-low)
            assert scale>0
            xy[valid]=(xy[valid]-(low+high)/2)/scale
        # A complete detection failure is retained with zero confidence, never
        # silently excluded or replaced by a different person's observations.
        flat=x.reshape(124,51)
        features=np.concatenate([flat.mean(0),flat.std(0),np.abs(np.diff(flat,axis=0)).mean(0)])
        feature_rows.append(features)
        clip=BIG/'clips'/f'person{pid:02d}.mp4'
        writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),30,(frames.shape[2],frames.shape[1]))
        assert writer.isOpened()
        for frame in frames:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
        writer.release()
        sampled=np.linspace(0,123,9,dtype=int)
        np.save(BIG/'frames'/f'person{pid:02d}.npy',frames[sampled],allow_pickle=False)
        np.save(BIG/'frames'/f'pose{pid:02d}.npy',x,allow_pickle=False)
        sheet=Image.new('RGB',(900,840),'white');draw=ImageDraw.Draw(sheet)
        for j,k in enumerate(sampled):
            im=Image.fromarray(frames[k]);im.thumbnail((295,250));cx=(j%3)*300;cy=(j//3)*280
            sheet.paste(im,(cx,cy+25));draw.text((cx+5,cy+4),f'{k/30:.2f}s',fill='black')
        sheet.save(HERE/f'contact{pid:02d}.jpg')
        row={'person':pid,'video_sha256':item['sha256'],'clip_path':str(clip),'clip_sha256':sha(clip),
             'start_frame':int(native_indices[0]),'last_frame':int(native_indices[-1]),'fps':fps,'sampled_frames':sampled.tolist(),
             'native_frame_indices':native_indices.tolist(),
             'start_seconds':start_seconds,'selection':selection,
             'valid_joint_fraction':float(valid.mean()),'frames_multiple_candidates':sum(n>1 for n in pose_counts),
             'missing_frames':sum(p is None for p in raw_landmarks),'elapsed_seconds':time.monotonic()-started}
        rows.append(row);save('video_preparation.json',rows)
        print(json.dumps({k:row[k] for k in ['person','start_seconds','valid_joint_fraction','missing_frames','elapsed_seconds']}),flush=True)
    # Small feature/quality summary stays in C; clips and frame arrays stay in D.
    save('numeric_features.json',{'people':[r['person'] for r in rows],'X':np.stack(feature_rows).tolist(),
                                'clinical_labels_read':False,'protocol_sha256':sha(HERE/'protocol.json')})

if __name__=='__main__':main()
