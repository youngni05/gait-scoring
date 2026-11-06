"""Verify retained original child inputs and record bounded visual recovery."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'20261105-child100-train-only';BIG=Path('D:/gait-research-data/D_P')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(OLD/'child-membership.json')['accepted'];new=read(HERE/'membership.json')['child'];index={r['record']:i for i,r in enumerate(new)}
with np.load(BIG/'child100-train-only-20261105/data.npz') as z:ox=z['x'][11261:]
with np.load(BIG/'matched-normal100-20261105/child/data.npz') as z:nx=z['x'][11261:]
for i,r in enumerate(old):
    n=new[index[r['record']]];assert n['video_sha256']==r['video_sha256'] and n['pose_sequence_sha256']==r['pose_sequence_sha256']
    assert n['start']==r['best_start'] and np.array_equal(ox[i],nx[index[r['record']]])
added=[r for r in new if r['record'] not in {v['record'] for v in old}]
assert len(added)==2
candidates=read(HERE/'live-candidates.json');reasons={}
for r in candidates:
    if r['record'] in index:reason='accepted'
    elif r['record'] in read(HERE/'live-review.json')['explicit_rejections']:reason=read(HERE/'live-review.json')['explicit_rejections'][r['record']]
    elif max((c['moving_fraction'] for c in r['candidates']),default=0)<.8:reason='No scanned124-frame candidate reaches unchanged80% movement screen'
    else:reason='Discontinuous or invalid candidate rejected in visual review'
    reasons[r['record']]=reason
out=dict(passed=True,scanned_source_clips=len(candidates),scanned_people=len({r['person'] for r in candidates}),
    accepted_clips=len(new),accepted_people=len({r['person'] for r in new}),original10_feature_arrays_exactly_retained=True,
    additions=[dict(record=r['record'],person=r['person'],age=r['age'],start=r['start'],end=r['end'],missing_fraction=r['missing_fraction'],moving_fraction=r['moving_fraction'],final_contact_sheet_sha256=sha(HERE/'qa'/(r['record']+'-final.jpg'))) for r in added],
    review='Final17 chronological source-frame contact sheets for both additions were visually inspected. Target identity and straight continuous walking checked; sparse visual QA is not framewise pose ground truth. All46 clips rescanned using stored poses; no new pose model inference.',
    exclusions=reasons,remaining_limit='Retain124 native frames and25% coordinate missingness. Some excluded clips may support a separately designed shorter-window study; do not pad/stretch/join them into this experiment.')
(HERE/'live-recheck.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in out.items() if k not in ['exclusions','review','remaining_limit']},indent=2))
