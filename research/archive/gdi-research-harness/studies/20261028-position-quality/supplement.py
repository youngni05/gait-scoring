import csv,json
import numpy as np
from collections import defaultdict
from analyze_quality import HERE,ROOT,OLD,PREV,DATA,OFFSETS,load,sha,save

names=['pelvis','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','spine','thorax','nose','head','left_shoulder','left_elbow','left_wrist','right_shoulder','right_elbow','right_wrist']
def spread(path):
    groups=defaultdict(list)
    with path.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['method']=='mean3':groups[r['patient_id']].append((float(r['prediction_L'])+float(r['prediction_R']))/2)
    return float(np.std([np.mean(v) for v in groups.values()],ddof=1))

def main():
    with np.load(OLD/'raw_features.npz',allow_pickle=False) as f:
        mask=f['split']=='validation';ids=f['patient_ids'][mask];videos=f['video_ids'][mask]
    grouped={p:defaultdict(list) for p in OFFSETS};missing=boundary=0
    sources=load(HERE/'sources.json')
    for person,video in zip(ids,videos):
        path=DATA/(str(video)+'.npz');assert sha(path)==sources[path.relative_to(ROOT).as_posix()]
        with np.load(path,allow_pickle=False) as z:q=z['input']
        for pos,start in OFFSETS.items():
            segment=q[start:start+124];m=segment[:,:,2]==0
            grouped[pos][str(person)].append(m.mean(axis=0))
            missing+=int(m.sum());boundary+=int((np.any(np.abs(segment[:,:,:2])>=.99999,axis=2)&m).sum())
    rates={pos:dict(zip(names,np.mean([np.mean(v,axis=0) for v in records.values()],axis=0).tolist())) for pos,records in grouped.items()}
    code=ROOT/'outputs/motionbert-smoke-20261015/vendor/lib/model/DSTformer.py'
    result=dict(joint_missing_person_mean=rates,missing_joint_observations=missing,
        missing_with_boundary_xy=boundary,missing_boundary_fraction=boundary/missing,
        overlapping_windows_count_cells_repeatedly=True,
        person_mean_prediction_sd_original=spread(PREV/'predictions.csv'),
        person_mean_prediction_sd_mixed=spread(HERE/'mixed_predictions.csv'),
        encoder_source_sha256=sha(code),
        source_observation='forward uses a Linear(dim_in, dim_feat) input and no explicit confidence==0 token mask; a trained network may still learn to use confidence',
        actual_occlusion_confirmed=False)
    save('supplement.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
