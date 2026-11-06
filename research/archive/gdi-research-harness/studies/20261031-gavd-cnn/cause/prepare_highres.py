"""Keep the same four records, annotation frame bounds and bbox target-selection rule."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent;sys.path.insert(0,str(STUDY))
from pose_qc import select
DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
source=DATA/'cause_highres_body25';complete=json.loads((source/'complete.json').read_text());assert complete['decoded_frames']==772
done=json.loads((source/'chunks_verified.json').read_text());verified=0
for chunk in done.values():
    for name,h in chunk['json_sha256'].items():
        assert hashlib.sha256((source/chunk['record_id']/name).read_bytes()).hexdigest()==h;verified+=1
assert verified==772
inventory=json.loads((DATA/'video_inventory.json').read_text());rows=list(csv.DictReader((STUDY/'cause_selected.csv').open(encoding='utf-8-sig')));root=DATA/'cause/highres_reviewed';root.mkdir(exist_ok=False)
for r in rows:
    key=r['record_id'];folder=root/key;folder.mkdir();m=inventory[r['video_id']];vi=json.loads(r['annotation_video_info']);boxes={b['frame']:b['bbox'] for b in json.loads((DATA/'annotations_by_sequence'/(key+'.json')).read_text())};selection=[]
    for f in range(int(r['annotation_frame_first']),int(r['annotation_frame_last'])+1):
        name='%s_%012d_keypoints.json'%(key,f);src=source/key/name;obj=json.loads(src.read_text());b=boxes[f];box=[b['left']*m['width']/vi['width'],b['top']*m['height']/vi['height'],b['width']*m['width']/vi['width'],b['height']*m['height']/vi['height']];idx,_,_=select(obj['people'],box)
        (folder/name).write_bytes(src.read_bytes());selection.append(dict(filename=name,person_index=idx))
    with (folder/'selection.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['filename','person_index']);w.writeheader();w.writerows(selection)
    r.update(json_dir=str(folder),selection_csv=str(folder/'selection.csv'),review_notes=r['review_notes']+' Diagnostic same-frame BODY25 height544; original cohort retained regardless of score; not anatomical ground truth.')
with (HERE/'reviewed_highres.csv').open('w',encoding='utf-8-sig',newline='') as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(HERE/'highres_receive_verification.json').write_text(json.dumps(dict(verified_json=verified,chunks=len(done),records=len(rows),source_complete=complete),indent=2));print('Four same-record high-resolution inputs prepared')
