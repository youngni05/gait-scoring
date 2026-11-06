"""CPU plan preflight using existing immutable data/components; no optimization."""
from pathlib import Path
import datetime,hashlib,json,os,sys
import numpy as np
H=Path(__file__).resolve().parent
cfg=json.loads((H/'protocol.json').read_text(encoding='utf-8'))
source=H.parent/cfg['source_study']
sys.path.insert(0,str(source))
os.environ['GDI_DATA_PATH']=cfg['data_path']
import torch
import run as r

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def state_hash(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

assert sha(source/'manifest.json')==cfg['source_manifest_sha256']
assert sha(cfg['data_path'])==cfg['data_sha256']
assert cfg['new_seed']!=cfg['old_seed']
d=r.data();people=d['person_id'];fold=d['fold']
assert len(np.unique(people))==692 and len(people)==1061
boundaries=[]
for f in range(5):
    for j in range(5):
        if j==f:continue
        tr,va,te=r.split(d,f,j)
        groups=[set(people[m]) for m in [tr,va,te]]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        assert set.union(*groups)==set(people)
        boundaries.append(dict(outer=f,inner=j,train=len(groups[0]),valid=len(groups[1]),test=len(groups[2])))

fit_keys=[(f,j) for f in range(5) for j in [*range(f),*range(f+1,5),None]]
def fit_seed(base,f,j):return base+f*100+(j if j is not None else 50)
old={fit_seed(cfg['old_seed'],f,j) for f,j in fit_keys}
new={fit_seed(cfg['new_seed'],f,j) for f,j in fit_keys}
assert len(old)==len(new)==25 and old.isdisjoint(new)
states=[]
for f,j in fit_keys:
    seed=fit_seed(cfg['new_seed'],f,j)
    r.setup(seed,'cpu');mse=r.GdiCNN(r.config()['dropout']);a=state_hash(mse)
    r.setup(seed,'cpu');rank=r.GdiCNN(r.config()['dropout']);b=state_hash(rank)
    assert a==b
    r.setup(fit_seed(cfg['old_seed'],f,j),'cpu');prior=r.GdiCNN(r.config()['dropout'])
    assert a!=state_hash(prior)
    states.append(dict(outer=f,inner=j,fit_seed=seed,paired_initial_state_sha256=a))

tr,va,te=r.split(d,0,1);x=d['x'][tr];ids=people[tr]
assert len(np.unique(ids))==416 and len(np.unique(people[va]))==138
samples=[]
for epoch in range(3):
    seed=fit_seed(cfg['new_seed'],0,1)+epoch
    a=r.sample_epoch(x,ids,seed);b=r.sample_epoch(x,ids,seed)
    np.testing.assert_array_equal(a,b)
    sampled_ids=ids[x.owner[a[:,0]]]
    p,counts=np.unique(sampled_ids,return_counts=True)
    assert set(p)==set(ids) and len(np.unique(counts))==1
    assert not set(p)&set(people[va]) and not set(p)&set(people[te])
    assert np.isin(a[:,1],[0,1]).all()
    assert not np.array_equal(a,r.sample_epoch(x,ids,fit_seed(cfg['old_seed'],0,1)+epoch))
    samples.append(dict(epoch=epoch+1,count=len(a),per_person=int(counts[0]),sha256=r.sample_digest(a)))

tasks=[tuple(x) for rows in cfg['tasks'].values() for x in rows]
assert len(tasks)==len(set(tasks))==10
assert set(tasks)=={(a,f) for a in cfg['arms'] for f in range(5)}
assert sorted(map(len,cfg['tasks'].values()))==[3,3,4]
assert cfg['ensemble']['between_seeds_weights']==[0.5,0.5]
proof=dict(passed=True,checked_at=datetime.datetime.now().astimezone().isoformat(),
    protocol_sha256=sha(H/'protocol.json'),data_sha256=sha(cfg['data_path']),
    source_manifest_sha256=sha(source/'manifest.json'),source_component_sha256={n:sha(source/n) for n in ['run.py','model.py','balanced.py','rank_aux.py']},
    person_boundaries=boundaries,paired_initializations=states,pilot_training_samples=samples,
    unique_tasks=10,planned_new_fits=50,planned_outer_predictions=10,
    input_cohort_verified=True,new_and_old_fit_seeds_disjoint=True,
    paired_initialization_verified=True,paired_sampling_verified=True,
    optimizer_steps=0,gpu_used=False,source_files_modified=False,
    limitation='Plan preflight with reused components only. New runner, synthetic end-to-end, GPU pilot and all-model result audits remain required',
    implementation_end_to_end_passed=False)
out=H/'checks/plan-preflight.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=True,boundaries=len(boundaries),paired_initializations=len(states),sample_epochs=len(samples),optimizer_steps=0,proof=str(out))))
