"""Freeze geometry-selected paired clips before any score prediction."""
import sys,json,tarfile,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;VIDEO=HERE.parent;BASE=VIDEO.parent
sys.path.insert(0,str(VIDEO));import prepare_clips as old
import numpy as np
cv2=old.cv2
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    audit=json.loads((HERE/'availability.json').read_text()); selected=[r for r in audit['people'] if r['windows']['2']['20']['count']]
    assert len(selected)==14
    protocol=dict(selection=audit['criteria'],duration_seconds=2,frames=60,fps=30,primary_heading_degrees=20,
        selection_rule='All people with eligible 2-second 20-degree straight window. Earliest eligible straight interval; maximum heading-span fully visible 2-second turn window in same provider walking interval as paired control. No scores used.',
        subjects=[r['subject'] for r in selected],conditions=[dict(name='clean',sigma=0.,seed=0)],
        geometry_conditions=['straight','turn'],expected_clips=56,expected_frames=3360,
        rf='Frozen motion3d 128-feature 6 folds x 5 seeds; same model bytes, no refit or calibration. Original 698-feature RF secondary reference.',
        care='Original four straight-walking cohorts; use annotation, no new frame-level geometry claim. 50 frames at 25Hz: first 2 sec primary, centered 2 sec sensitivity. Compare full vs short only on same eligible records and people; held-out fold predictions only. 5-sec first-window comparison on >=125-frame common subset.',
        outcomes='Paired-view absolute and signed differences, score band [0.5,1.5), between-person spread, seedwise effects. CARE-PD person-equal MAE, bias and >=2 errors overall and true scores 2/3,3. No clinical pass threshold, no MoVi accuracy or assumed healthy score.',
        previous_control='Old 5-second clean windows only for the intersection of original five subjects and current primary 14; do not compare changed samples as if paired.',
        roi='Same existing MediaPipe RGB landmarks and 10 percent padding as previous inference; reference used solely for eligibility.',
        sources=dict(audit=old.digest(HERE/'availability.json'),prepare=old.digest(Path(__file__))))
    save(HERE/'protocol.json',protocol)
    source_rows=json.loads((old.PRIOR/'results.json').read_text())['results']; manifest=[];tiles=[]
    for item in selected:
        subject=item['subject']
        for condition in ['straight','turn']:
            window=item['windows']['2']['20']['earliest'] if condition=='straight' else item['windows']['2']['max_turn']
            assert window['heading_p90_span']>60 if condition=='turn' else window['heading_p90_span']<=20
            start=window['start']
            for cam in ['PG1','PG2']:
                key='s%02d_%s_%s'%(subject,cam,condition);folder=HERE/'clips'/key;folder.mkdir(parents=True,exist_ok=True)
                video=old.SOURCE/'raw'/('F_%s_Subject_%d_L.avi'%(cam,subject))
                row=next(r for r in source_rows if r['subject']==subject and r['camera']==cam and r['variant']=='heavy' and r['input_width']==800)
                assert old.digest(video)==row['video_sha256']
                posepath=old.PRIOR/('subject-%02d'%subject)/'heavy'/(cam+'_800.npz');assert old.digest(posepath)==row['prediction_sha256']
                z=np.load(posepath);ids=np.arange(start,start+60);where=np.searchsorted(z['video_frame_one_based'],ids)
                assert np.array_equal(z['video_frame_one_based'][where],ids)
                lm=z['landmarks'][where];assert np.isfinite(lm).all() and (z['pose_counts'][where]>0).all()
                xy=lm[:,:,:2]*[800,600];lo=xy.min(1);hi=xy.max(1);pad=(hi-lo)*.1;boxes=np.c_[lo-pad,hi+pad].astype('float32')
                np.savez_compressed(folder/'boxes.npz',boxes=boxes,frames_one_based=ids,landmark_visibility=lm[:,:,3])
                cap=cv2.VideoCapture(str(video));assert abs(cap.get(5)-30)<1e-6;cap.set(cv2.CAP_PROP_POS_FRAMES,start-1);hashes=[]
                for k in range(60):
                    ok,im=cap.read();assert ok
                    p=folder/('%03d.jpg'%k);assert cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,98]);hashes.append(old.digest(p))
                    if k==30:
                        b=boxes[k].astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(0,255,0),2);cv2.putText(im,key,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,0),2);tiles.append(cv2.resize(im,(400,300)))
                cap.release()
                manifest.append(dict(key=key,subject=subject,camera=cam,condition=condition,geometry=window,source=str(video),source_sha256=row['video_sha256'],pose_sha256=row['prediction_sha256'],frames_one_based=[int(ids[0]),int(ids[-1])],frame_sha256=hashes,boxes_sha256=old.digest(folder/'boxes.npz')))
    save(HERE/'clips.json',manifest)
    # Four columns: straight PG1/PG2, turn PG1/PG2. Split pages for legibility.
    for page,start in enumerate(range(0,len(tiles),16)):
        subset=tiles[start:start+16];cv2.imwrite(str(HERE/('review-%d.jpg'%page)),np.concatenate([np.concatenate(subset[k:k+4],1) for k in range(0,len(subset),4)],0))
    with tarfile.open(HERE/'bundle.tar.gz','w:gz') as tar:
        for p in sorted((HERE/'clips').rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(HERE).as_posix())
        for name in ['protocol.json','clips.json','infer.py','submit.slurm']:tar.add(HERE/name,arcname=name)
    print(json.dumps(dict(people=len(selected),clips=len(manifest),frames=3360,bytes=(HERE/'bundle.tar.gz').stat().st_size,sha256=old.digest(HERE/'bundle.tar.gz'))))
if __name__=='__main__':main()
