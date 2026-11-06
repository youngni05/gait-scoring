"""Select visible foreground children; audit standing vs walking before scores.

The frozen first-500-frame window schedule remains unchanged. Movement coverage
is a diagnostic, not a newly validated gait detector or a clinical threshold.
"""
from pathlib import Path
import json,csv,sys,argparse,hashlib
import numpy as np
from scipy.ndimage import median_filter
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
BUNDLE=ROOT/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
sys.path.insert(0,str(BUNDLE))
from preprocess import extract_windows
from prepare import prepare

def select(people,height):
    eligible=[]
    for index,person in enumerate(people):
        a=np.asarray(person['pose_keypoints_2d']).reshape(25,3)
        hips=a[[8,9,12]];ankles=a[[11,14]]
        hips=hips[hips[:,2]>.1];ankles=ankles[ankles[:,2]>.1]
        if not len(hips) or not len(ankles):continue
        if ankles[:,1].mean()-hips[:,1].mean()<.18*height:continue
        # Foreground walking floor, below the bed/background staff feet. Set
        # from source-video inspection, before GDI inference; QA still required.
        if ankles[:,1].mean()<.65*height:continue
        eligible.append(index)
    return eligible[0] if len(eligible)==1 else -1,eligible

def process(partial=False):
    exclusions=json.loads((HERE/'target_exclusions.json').read_text())
    metadata={Path(r['file']).stem:r for r in json.loads((HERE/'video_metadata.json').read_text())}
    people={Path(r['file']).stem:r for r in json.loads((HERE/'td_download.json').read_text())['videos']}
    records=[];quality=[]
    for stem,meta in sorted(metadata.items()):
        folder=HERE/'body25'/stem;paths=sorted(folder.glob('*_keypoints.json'))
        if len(paths)!=meta['frames']:
            if partial:continue
            raise ValueError('Incomplete record '+stem)
        assert [int(p.stem.split('_')[-2]) for p in paths]==list(range(meta['frames']))
        raw=np.full((len(paths),75),np.nan);selections=[];ambiguous=[]
        for f,path in enumerate(paths):
            obj=json.loads(path.read_text());sel,eligible=select(obj['people'],meta['height'])
            selections.append((path.name,sel))
            if len(eligible)>1:ambiguous.append(f)
            if sel>=0:raw[f]=obj['people'][sel]['pose_keypoints_2d']
        with (folder/'selection.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['filename','person_index']);writer.writerows(selections)
        a=raw.reshape(-1,25,3);valid=np.isfinite(a[:,8,0])&(a[:,8,2]>.1)
        hip=a[:,8,0].copy();hip[~valid]=np.nan
        observed=np.flatnonzero(valid)
        movement=np.zeros(len(raw),bool)
        if len(observed):
            smooth=median_filter(np.interp(np.arange(len(raw)),observed,hip[observed]),size=9)
            thigh=np.linalg.norm(a[:,[9,12],:2]-a[:,[10,13],:2],axis=2)
            scale=np.nanmedian(thigh[(a[:,[9,12],2]>.1)&(a[:,[10,13],2]>.1)])
            movement=(np.abs(np.gradient(smooth))>max(1,.015*scale))&valid
        diagnostics=[];failure=None
        try:
            extracted=extract_windows(raw)
            for k in extracted['window_start']:
                diagnostics.append(dict(start=int(k),movement_fraction=float(movement[k:k+124].mean()),selected_fraction=float(np.isfinite(raw[k:k+124,0]).mean())))
        except ValueError as e:failure=str(e)
        if stem in exclusions:
            failure=exclusions[stem]
            diagnostics=[]
        run=best=0
        for v in movement:
            run=run+1 if v else 0;best=max(best,run)
        q=dict(record=stem,person=people[stem]['person'],age=people[stem]['age_years'],frames=len(raw),selected_frames=int(np.isfinite(raw[:,0]).sum()),multiple_eligible_frames=ambiguous,scheduled_windows=diagnostics,longest_contiguous_movement_frames=best,preprocessing_failure=failure)
        quality.append(q)
        if failure is None:
            records.append(dict(person_id=people[stem]['person'],record_id=stem,json_dir='body25/'+stem,selection_csv='body25/'+stem+'/selection.csv',fps=meta['fps'],fps_basis='decoded source MOV average rate; native frames, no resampling',source_dataset='LIVE-GaitNeuroKids v1.1 TD sagittal videos'))
    suffix='_partial' if partial else ''
    (HERE/('quality'+suffix+'.json')).write_text(json.dumps(quality,indent=2))
    manifest=dict(schema='body25_records_v2',coordinate_convention='OpenPose_BODY25_pixel_xy',time_policy='native_frames_exploratory',records=records)
    path=HERE/('manifest'+suffix+'.json');path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(complete_videos=len(quality),eligible_videos=len(records),people=len(set(r['person_id'] for r in records)),windows=sum(len(q['scheduled_windows']) for q in quality),windows_with_80pct_movement=sum(w['movement_fraction']>=.8 for q in quality for w in q['scheduled_windows'])),indent=2))
    return path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();process(args.partial)
