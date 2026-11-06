"""Select paired MoVi walking clips by visibility, before RF inference."""
import sys, json, hashlib, tarfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps'))
import cv2
import numpy as np

SOURCE = Path('D:/gait-research-data/MoVi')
PRIOR = SOURCE/'experiments/walking-expansion'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    rows=json.loads((PRIOR/'results.json').read_text())['results']
    candidates=[]
    for row in rows:
        if row['camera']!='PG1' or row['variant']!='heavy' or row['evaluation_frames']<150: continue
        subject=row['subject']; masks=[]
        for cam in ['PG1','PG2']:
            z=np.load(PRIOR/('subject-%02d'%subject)/'heavy'/(cam+'_800_comparison.npz'))
            a=z['reference_xy']
            masks.append(((a[:,:,0]>=0)&(a[:,:,0]<800)&(a[:,:,1]>=0)&(a[:,:,1]<600)&~z['reference_gap']).all(1))
        both=masks[0]&masks[1]
        starts=[k for k in range(len(both)-149) if both[k:k+150].all()]
        candidates.append(dict(subject=subject,eligible=bool(starts),start_one_based=row['video_frames'][0]+starts[0] if starts else None))
    selected=[x for x in sorted(candidates,key=lambda x:x['subject']) if x['eligible']][:5]
    assert len(selected)==5
    protocol=dict(dataset='MoVi F synchronized PG1/PG2',doi='10.5683/SP2/JRHDRN',
        selection='First five ascending IDs with 150 consecutive frames where all 12 reference major joints are visible in both cameras; earliest eligible interval. Reference is used only for clip eligibility, never pose or RF input.',
        candidates=candidates,selected=selected,frames=150,fps=30,duration_seconds=5,
        estimator='HMR2.0b official image backbone and SMPL rotation head only, no mesh or SMPL body model. Different estimator from CARE-PD WHAM.',
        roi='Existing MediaPipe Heavy RGB detection, all 33 landmark extents, padded 10 percent on each side; no reference-coordinate input.',
        conditions=[dict(name='clean',sigma=0.,seed=0)]+[dict(name='jitter%d_s%d'%(int(s*100),seed),sigma=s,seed=seed) for s in [.02,.05] for seed in [11,22,33]],
        perturbation='Independent per-frame center x/y shifts with Gaussian SD sigma times baseline box height; isotropic scale exp(Gaussian SD sigma); Gaussian draws clipped at +/-3 SD. Synthetic detector-box noise, not observed detection error.',
        rf='Frozen motion3d 128-feature models, all 6 folds x 5 seeds. No refitting, calibration, smoothing, or score-based clip selection.',
        scope='Paired view and detector-box sensitivity only. No gait score labels, clinical accuracy or healthy score assumption. Five healthy source participants are not PD external validation.',
        numerical_checks='SO(3) orthogonality and determinant errors <=1e-5; finite features; exact count; stored input/code hashes; RF replay <=1e-12. No clinical stability pass threshold specified.')
    (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
    manifest=[];tiles=[]
    for item in selected:
        subject=item['subject'];start=item['start_one_based']
        for cam in ['PG1','PG2']:
            key='s%02d_%s'%(subject,cam);folder=HERE/'clips'/key;folder.mkdir(parents=True,exist_ok=True)
            video=SOURCE/'raw'/('F_%s_Subject_%d_L.avi'%(cam,subject))
            old=next(x for x in rows if x['subject']==subject and x['camera']==cam and x['variant']=='heavy')
            assert digest(video)==old['video_sha256']
            posepath=PRIOR/('subject-%02d'%subject)/'heavy'/(cam+'_800.npz')
            assert digest(posepath)==old['prediction_sha256']
            z=np.load(posepath);ids=np.arange(start,start+150);where=np.searchsorted(z['video_frame_one_based'],ids)
            assert np.array_equal(z['video_frame_one_based'][where],ids)
            lm=z['landmarks'][where];assert np.isfinite(lm).all() and (z['pose_counts'][where]>0).all()
            xy=lm[:,:,:2]*[800,600];lo=xy.min(1);hi=xy.max(1);pad=(hi-lo)*.1;boxes=np.c_[lo-pad,hi+pad].astype('float32')
            np.savez_compressed(folder/'boxes.npz',boxes=boxes,frames_one_based=ids,landmark_visibility=lm[:,:,3])
            cap=cv2.VideoCapture(str(video));assert abs(cap.get(cv2.CAP_PROP_FPS)-30)<1e-6
            assert (int(cap.get(3)),int(cap.get(4)))==(800,600)
            cap.set(cv2.CAP_PROP_POS_FRAMES,start-1);frame_hashes=[]
            for k in range(150):
                ok,im=cap.read();assert ok
                path=folder/('%03d.jpg'%k);assert cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,98])
                frame_hashes.append(digest(path))
                if k==75:
                    a=im.copy();b=boxes[k].astype(int);cv2.rectangle(a,tuple(b[:2]),tuple(b[2:]),(0,255,0),2)
                    cv2.putText(a,key,(20,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2);tiles.append(cv2.resize(a,(400,300)))
            cap.release()
            manifest.append(dict(key=key,subject=subject,camera=cam,source=str(video),source_sha256=old['video_sha256'],pose_sha256=old['prediction_sha256'],frames_one_based=[int(ids[0]),int(ids[-1])],frame_sha256=frame_hashes,boxes_sha256=digest(folder/'boxes.npz')))
    cv2.imwrite(str(HERE/'clip-review.jpg'),np.concatenate([np.concatenate(tiles[k:k+2],axis=1) for k in range(0,len(tiles),2)],axis=0))
    (HERE/'clips.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(HERE/'clips.tar.gz','w:gz') as tar:
        for p in sorted((HERE/'clips').rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(HERE)).replace('\\','/'))
        for n in ['protocol.json','clips.json']:tar.add(HERE/n,arcname=n)
    print(json.dumps(dict(selected=selected,clips=len(manifest),frames=1500,archive_bytes=(HERE/'clips.tar.gz').stat().st_size,sha256=digest(HERE/'clips.tar.gz'))))

if __name__=='__main__':main()
