"""Independently check saved metric tables, folds and external aggregation."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
g=read(HERE/'source-gate-results.json');v=read(HERE/'external-live-results.json');protocol=sha(HERE/'protocol.json');assert g['protocol_sha256']==v['protocol_sha256']==protocol
with np.load('D:/gait-research-data/D_P/ljmu100-mixed-20261104/data.npz') as z:ids=z['person_id'];y=z['y'];fold=z['fold'];src=z['source']
for m in g['models']:
 assert m['train_people']==np.unique(ids[fold!=m['fold']]).tolist();assert m['test_people']==np.unique(ids[fold==m['fold']]).tolist()
for r in g['rows']:
 pred=np.array(g['predictions'][r['family']][r['method']]);mask=np.repeat((src==('LJMU' if r['group']=='LJMU_assumed100' else 'Stanford'))[:,None],2,1)
 if r['group'].endswith('ge90'):mask&=y>=90
 if r['group'].endswith('ge100'):mask&=y>=100
 if r['group'].endswith('lt70'):mask&=y<70
 ri,si=np.nonzero(mask);people,n=np.unique(ids[ri],return_counts=True);count=dict(zip(people,n));w=np.array([1/count[q]/len(people) for q in ids[ri]]);error=pred[ri,si]-y[ri,si]
 values={'mae':np.abs(error)@w,'shortfall':np.maximum(-error,0)@w,'bias':error@w,'under_fraction':(error<0)@w,'overprediction':np.maximum(error,0)@w,'mean_prediction':pred[ri,si]@w}
 assert r['people']==len(people)
 for k,val in values.items():assert abs(r[k]-val)<1e-10,(r,k)
original=read(ROOT/'studies/20261104-ljmu100-mixed/comparison.json')
for family in ['CNN','STT']:
 r=next(r for r in original['rows'] if r['model']==family+'_refit' and r['source']=='Stanford')
 for method,key in [('baseline','before_MAE'),('mixed','after_MAE')]:assert abs(next(x['mae'] for x in g['rows'] if x['family']==family and x['group']=='Stanford_all' and x['method']==method)-r[key])<1e-12
with np.load('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105/data.npz') as z:child_ids=z['person_id'][1073:]
assert len(child_ids)==12 and len(np.unique(child_ids))==9
for r in v['rows']:
 values=np.array(v['predictions'][r['family']+'__'+r['method']]);assert values.shape==(5,12,2)
 weights=np.array([1/(9*(child_ids==q).sum()*2*5) for q in child_ids]);mean=float((values*weights[None,:,None]).sum());assert abs(mean-r['mean_prediction'])<1e-10
 for person,expected in r['per_person'].items():assert abs(values[:,child_ids==person].mean()-expected)<1e-10
for row in v['provenance']:assert sha(Path(row['checkpoint']))==row['sha256'] and row['heldout_three_record_replay_error']<.001
out=dict(passed=True,source_rows=len(g['rows']),external_rows=len(v['rows']),frozen_checkpoints=len(v['provenance']),max_checkpoint_replay_error=max(r['heldout_three_record_replay_error'] for r in v['provenance']),source_results_sha256=sha(HERE/'source-gate-results.json'),external_results_sha256=sha(HERE/'external-live-results.json'),protocol_sha256=protocol)
(HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
