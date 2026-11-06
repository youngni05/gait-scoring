"""Standing-frame versus walking-frame height scaling, paired frozen gait windows."""
import os,sys,json,hashlib,csv,time
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
H=Path(__file__).resolve().parent
OLD=H.parent/'20261105-planar-measurement-validation'
B=Path('D:/gait-research-data/D_P/standing-height-calibration-20261105')
ROOT=Path('D:/gait-research-data/MoVi')
DEP=Path('D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps')
MODEL=DEP.parent.parent/'20261028-movi-totalcapture-acquisition/heavy/pose_landmarker_heavy.task'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def primary():return [r for r in read(OLD/'input-audit.json')['records'] if r['group']=='primary2s']

def freeze():
    assert not(H/'protocol.json').exists()
    rows=primary();assert len(rows)==28 and len(set(r['subject'] for r in rows))==14
    sources={str(p):sha(p) for p in [OLD/'input-audit.json',OLD/'measurements.json',OLD/'results.json',OLD/'verification.json',MODEL,H/'standing-frame60-contactsheet.jpg']}
    for r in rows:
        p=ROOT/'raw'/f"F_{r['camera']}_Subject_{r['subject']}_L.avi";sources[str(p)]=sha(p)
        p=ROOT/'reference'/f"F_v3d_Subject_{r['subject']}.mat";sources[str(p)]=sha(p)
    save(H/'protocol.json',{'status':'frozen_before_standing_inference_and_outcomes','code_sha256':sha(__file__),'sources':sources,
      'question':'Does replacing walking landmark height with an independently recorded standing landmark height reduce speed and step/stride length error in identical people and gait windows?',
      'data':'Same 14 people, PG1 and PG2, 28 fixed primary 2s straight gait windows from previous verified study; age18..33. All failures retained.',
      'standing_frames':'Video one-based31..90 (2sec);1..30 tracker warmup. Fixed for every person/view before inference; frame60 of all28 views visually reviewed: participant standing, usually T-pose. Initial calibration occurs at a different position from some walking windows; do not use reference depth to fix input.',
      'inference':'Same existing MediaPipe Heavy task, VIDEO CPU,3poses, confidence .5/.5/.5;800x600 30Hz native RGB; choose largest 2D pose bbox as old extractor. Save all33 image landmarks and counts. No 3D predictor inputs, no reference-based fitting.',
      'baseline':'Provider height / median per-frame vertical range of all33 walking landmarks, exactly previous results.',
      'standing':'Provider height / median per-frame vertical range of all33 standing landmarks31..90. Reuse same definition to isolate standing versus walking, not a top-of-head reconstruction. One constant per person/camera.',
      'standing_quality':'All60frames finite, pose count>0, positive span; median hip23/24 visibility>=.5, each heel29/30 visibility>=.5. Failure unmeasurable for standing condition; no fallback or threshold relaxation.',
      'isolation':'Hold all walking coordinates, quality decisions, event times, step availability and reference values fixed. Scale previous speed,step length,stride length by standing_scale/walking_scale. No change in time/asymmetry; no re-detection or scale-dependent gate change. This isolates unit conversion, not a newly tuned end-to-end extractor.',
      'metrics':'For each view/metric compare only identical available person/window pairs; also report all registered people and unavailable reasons. MAE,bias,p95AE; paired change in abs error standing-baseline and10000 person bootstrap95CI. CI entirely<0 supports reduction; crossing0 inconclusive; report all directions.',
      'scope':'Standing-pose calibration, not calibrated floor-plane measurement; head-top/body-span mismatch remains. Height and camera-depth/location sensitivity remain; not phone/elderly/repeatability/clinical validation.',
      'resources':'28x90=2520 new CPU pose frames,28x60=1680 calibration frames; no training/GPU/download/upload. Small summaries/code in project; arrays under D:/gait-research-data/D_P/standing-height-calibration-20261105, expected<10MB.'})
    print('FROZEN',len(rows),'views',flush=True)

def infer():
    sys.path.insert(0,str(DEP))
    import cv2,numpy as np,mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    protocol=read(H/'protocol.json');assert protocol['code_sha256']==sha(__file__);assert protocol['sources'][str(MODEL)]==sha(MODEL)
    B.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2)
    for n,r in enumerate(primary(),1):
        sid=r['subject'];camera=r['camera'];key=f's{sid:02d}_{camera}';dest=B/(key+'.npz');receipt=B/(key+'.json');video=ROOT/'raw'/f'F_{camera}_Subject_{sid}_L.avi'
        assert protocol['sources'][str(video)]==sha(video)
        if dest.exists() and receipt.exists():
            q=read(receipt);assert q['array_sha256']==sha(dest) and q['code_sha256']==sha(__file__);print('CACHED',n,key,flush=True);continue
        t=time.monotonic();cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        width,height,fps=(cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FPS));assert (width,height)==(800,600) and abs(fps-30)<1e-6
        lm=np.full((90,33,5),np.nan);counts=[]
        options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL),delegate=python.BaseOptions.Delegate.CPU),running_mode=vision.RunningMode.VIDEO,num_poses=3,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
        with vision.PoseLandmarker.create_from_options(options) as detector:
            for i in range(90):
                ok,frame=cap.read();assert ok,(key,i)
                prediction=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),round(i*1000/30));poses=prediction.pose_landmarks;counts.append(len(poses))
                if poses:
                    index=max(range(len(poses)),key=lambda k:np.ptp([p.x for p in poses[k]])*np.ptp([p.y for p in poses[k]]));lm[i]=[[p.x,p.y,p.z,p.visibility,p.presence] for p in poses[index]]
        cap.release();np.savez_compressed(dest,video_frame_one_based=np.arange(1,91),landmarks=lm,pose_counts=counts)
        save(receipt,{'key':key,'subject':sid,'camera':camera,'video':str(video),'video_sha256':protocol['sources'][str(video)],'array':str(dest),'array_sha256':sha(dest),'code_sha256':sha(__file__),'model_sha256':sha(MODEL),'seconds':time.monotonic()-t,'versions':{'python':sys.version,'mediapipe':mp.__version__,'opencv':cv2.__version__,'numpy':np.__version__}})
        print('DONE',n,key,round(time.monotonic()-t,2),flush=True)
    save(H/'inference-completion.json',{'views':28,'people':14,'frames':2520,'code_sha256':sha(__file__),'arrays':{str(p):sha(p) for p in B.glob('*.npz')},'receipts':{str(p):sha(p) for p in B.glob('*.json')}})

def analyze():
    import numpy as np
    protocol=read(H/'protocol.json');assert protocol['code_sha256']==sha(__file__)
    complete=read(H/'inference-completion.json');assert complete['code_sha256']==sha(__file__)
    old={r['key']:r for r in read(OLD/'measurements.json')['records']};calibrations=[];pairs=[]
    for row in primary():
        sid=row['subject'];camera=row['camera'];path=B/f's{sid:02d}_{camera}.npz';assert sha(path)==complete['arrays'][str(path)];z=np.load(path);np.testing.assert_array_equal(z['video_frame_one_based'],np.arange(1,91));lm=z['landmarks'][30:90];span=np.ptp(lm[:,:,1]*600,axis=1)
        good=bool(np.isfinite(lm).all() and (z['pose_counts'][30:90]>0).all() and (span>0).all() and np.median(lm[:,[23,24],3])>=.5 and np.median(lm[:,29,3])>=.5 and np.median(lm[:,30,3])>=.5)
        r=old[row['key']];d=np.load(row['array']);height=float(d['height']);newscale=height/float(np.median(span)) if good else None;ratio=newscale/r['scale_metres_per_pixel'] if good else None
        calibration={'key':row['key'],'subject':sid,'camera':camera,'standing_quality':good,'reason':None if good else 'standing_pose_or_heel_visibility','height':height,'standing_span_pixels':float(np.median(span)) if np.isfinite(span).all() else None,'standing_span_cv':float(span.std()/span.mean()) if np.isfinite(span).all() else None,'standing_scale':newscale,'walking_scale':r['scale_metres_per_pixel'],'scale_ratio':ratio,'median_heel_visibility':[float(np.median(lm[:,j,3])) if np.isfinite(lm).all() else None for j in [29,30]],'standing_hip_xy':lm[:,[23,24],:2].mean((0,1)).tolist() if np.isfinite(lm).all() else None,'walking_hip_xy':d['lm'][:,[23,24],:2].mean((0,1)).tolist()};calibrations.append(calibration)
        for metric in ['speed','step_length','stride_length']:
            baseline=r['prediction'][metric];truth=r['reference'][metric];new=baseline*ratio if good and baseline is not None else None
            pairs.append({'key':row['key'],'subject':sid,'camera':camera,'metric':metric,'reference':truth,'baseline':baseline,'standing':new,'paired_available':new is not None and truth is not None,'standing_quality':good})
    summaries={}
    for camera in ['PG1','PG2']:
        summaries[camera]={}
        for metric in ['speed','step_length','stride_length']:
            rr=[p for p in pairs if p['camera']==camera and p['metric']==metric];matched=[p for p in rr if p['paired_available']];n=len(matched)
            a=np.array([p['baseline']-p['reference'] for p in matched]);b=np.array([p['standing']-p['reference'] for p in matched]);delta=abs(b)-abs(a);rng=np.random.default_rng(20261105);ci=np.quantile(delta[rng.integers(0,n,(10000,n))].mean(1),[.025,.975]).tolist() if n>=2 else None
            summaries[camera][metric]={'registered_people':len(rr),'baseline_available_people':sum(p['baseline'] is not None and p['reference'] is not None for p in rr),'paired_people':n,'paired_ids':[p['subject'] for p in matched],'baseline_MAE':float(abs(a).mean()) if n else None,'standing_MAE':float(abs(b).mean()) if n else None,'MAE_change_standing_minus_baseline':float(delta.mean()) if n else None,'paired_change_CI95':ci,'baseline_bias':float(a.mean()) if n else None,'standing_bias':float(b.mean()) if n else None,'baseline_p95AE':float(np.quantile(abs(a),.95)) if n else None,'standing_p95AE':float(np.quantile(abs(b),.95)) if n else None,'improved_people':int((delta<0).sum()),'worse_people':int((delta>0).sum()),'interpretation':'reduction_supported' if ci and ci[1]<0 else 'increase_supported' if ci and ci[0]>0 else 'inconclusive'}
    save(H/'comparison.json',{'summary':summaries,'calibrations':calibrations,'pairs':pairs,'protocol_sha256':sha(H/'protocol.json'),'inference_sha256':sha(H/'inference-completion.json'),'code_sha256':sha(__file__)})
    with (H/'paired-measurements.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':{'freeze':freeze,'infer':infer,'analyze':analyze}[sys.argv[1]]()
