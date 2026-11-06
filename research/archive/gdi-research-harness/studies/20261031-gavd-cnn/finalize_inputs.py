"""Freeze reviewed full BODY25 inputs before scores; retain original frame count."""
from pathlib import Path
import argparse,csv,hashlib,json,re,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pose_qc import select
HERE=Path(__file__).resolve().parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
OUTDOOR={'hSIYGZhRGd4','KnvYRnTA3XQ','-0wbleNgAwg','3FXUw98rrUY','PAeh4qBwsUk','ucZdenbi65k','b2n0USprXmQ','qeG-H6lnlL4','338R10fA_wU','50wy7QkeCSw','BGMTJ8NJAxA'}
MOVING={'KnvYRnTA3XQ','N9tJ7I4ls6I','0cQOoqhD9Mo','50wy7QkeCSw','BGMTJ8NJAxA'}
ACTORS={'WpARylM4UYU':'visual-normal-gym-boy','_-Ubl8iD2B0':'visual-normal-gym-boy',
        'TgkxrrhnvlM':'visual-mission-gait-striped-shirt','7jJVj2ZNk_E':'visual-mission-gait-striped-shirt','0CdEPm-VYwA':'visual-mission-gait-striped-shirt'}
def readcsv(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def writecsv(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(args):
    meta=json.loads((DATA/'video_inventory.json').read_text());reviews=[];audits=[];condition=[];destroot=DATA/('reviewed_shift%d'%args.shift)
    if destroot.exists():raise ValueError('Do not overwrite frozen reviewed inputs')
    destroot.mkdir(parents=True)
    for group in ['normal','abnormal']:
        source=DATA/(group+'_full_body25');complete=json.loads((source/'complete.json').read_text())
        jobs=json.loads((HERE/(group+'_full_jobs.json')).read_text())['records']
        assert complete['decoded_frames']==sum(j['decoded_last']-j['decoded_first']+1 for j in jobs)
        temporal={r['record_id']:r for r in json.loads((HERE/(group+'_temporal_audit.json')).read_text())}
        quality={r['record_id']:r for r in readcsv(HERE/'qa'/(group+'_full')/'quality.csv')}
        for r in readcsv(HERE/(group+'_selected.csv')):
            key=r['record_id'];m=meta[r['video_id']];q=quality[key];reason=''
            if temporal[key]['temporal_review']=='exclude':reason=temporal[key]['reason']
            elif float(q['mean_toe_confidence'])<.1:reason='full_mean_toe_confidence_below_pilot_gate'
            elif float(q['max_joint_jump_bbox_height'])>.5:reason='large_joint_jump_requires_manual_review'
            first=int(r['annotation_frame_first']);count=min(1499,int(r['annotation_frame_last'])-first+1)
            paths=[source/key/('%s_%012d_keypoints.json'%(key,f+args.shift)) for f in range(first,first+count)]
            if not all(p.is_file() for p in paths):reason='neighbour_frame_unavailable_for_sensitivity' if args.shift else 'missing_primary_frame'
            audits.append(dict(record_id=key,video_id=r['video_id'],label=r['label'],include=not bool(reason),reason=reason or 'reviewed_before_CNN; model_window_quality_gate_still_applies'))
            if reason:continue
            boxes={b['frame']:b['bbox'] for b in json.loads((DATA/'annotations_by_sequence'/(key+'.json')).read_text())};vi=json.loads(r['annotation_video_info']);sx=m['width']/vi['width'];sy=m['height']/vi['height']
            folder=destroot/key;folder.mkdir();selections=[];hashes={}
            for f,src in zip(range(first,first+count),paths):
                obj=json.loads(src.read_text());b=boxes[f];box=[b['left']*sx,b['top']*sy,b['width']*sx,b['height']*sy];idx,_,_=select(obj['people'],box)
                name='%s_%012d_keypoints.json'%(key,f);dest=folder/name;dest.write_bytes(src.read_bytes());selections.append(dict(filename=name,person_index=idx));hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
            writecsv(folder/'selection.csv',selections);(folder/'provenance.json').write_text(json.dumps(dict(source=str(source/key),shift=args.shift,sha256=hashes),indent=2))
            r.update(json_dir=str(folder),selection_csv=str(folder/'selection.csv'),fps=m['fps'],fps_basis='OpenCV decoded original MP4; native frames; SHA256 '+m['sha256'],annotation_to_decoded_frame_offset=-1+args.shift,frame_mapping_reviewed='yes',target_and_anatomical_sides_reviewed='yes',walking_and_view_reviewed='yes',label_reviewed='yes',include='yes',review_notes='Eight chronological video samples and BODY25 overlays; bbox target on every frame. Original anatomical BODY25 channels retained; not motion-capture verified. Exact source 0/1 annotation convention remains uncertain; +1-frame sensitivity separately. No score-dependent selection.')
            if r['video_id'] in ACTORS:r.update(cluster_id=ACTORS[r['video_id']],identity_basis='conservative_cross_video_visual_same_actor_group_not_verified_person_ID')
            reviews.append(r)
            condition.append(dict(record_id=key,video_id=r['video_id'],cluster_id=r['cluster_id'],label=r['label'],view=r['view'],fps=m['fps'],fps_group=round(m['fps'],3),environment='outdoor' if r['video_id'] in OUTDOOR else 'indoor',camera='moving' if r['video_id'] in MOVING else 'approximately_fixed',height=m['height'],width=m['width'],bbox_height_pixels=q['bbox_height_pixels'],mean_toe_confidence=q['mean_toe_confidence'],max_required_zero_fraction=q['max_required_zero_fraction']))
    writecsv(HERE/('reviewed_shift%d.csv'%args.shift),reviews);writecsv(HERE/('final_selection_shift%d.csv'%args.shift),audits);writecsv(HERE/('conditions_shift%d.csv'%args.shift),condition)
    print(json.dumps(dict(shift=args.shift,records=len(reviews),sources=len({r['video_id'] for r in reviews}),labels={v:sum(r['label']==v for r in reviews) for v in ['Normal Gait','Abnormal Gait']})))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shift',type=int,choices=[0,1],default=0);main(p.parse_args())
