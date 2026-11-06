"""Check person boundaries, exact matched mass and equivalent loss algebra."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(HERE/'bundle/study-protocol.json');arms={};checks=0;maximum=0.
for arm in ['adult','child']:
    assert sha(BIG/arm/'data.npz')==p['data_sha256'][arm]
    with np.load(BIG/arm/'data.npz') as z:d={k:z[k] for k in z.files}
    arms[arm]=d;ext=d['source']!='Stanford';assert np.all(d['fold'][ext]==-1) and np.all(d['y'][ext]==100)
    for family in ['CNN','STT','bilateral']:
        spec=importlib.util.spec_from_file_location('checked_windows',HERE/'bundle'/family/'windows.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        x=d['x'] if family=='CNN' else np.ascontiguousarray(d['x'][...,:8].reshape(-1,2,124,4,2))
        ws=m.WindowSet(x,d['window_record'],len(d['y']),d['window_start'],d['canonical_fallback'],d['record_weight']);ws.validate()
        for f in range(5):
            for j in [None]+[j for j in range(5) if j!=f]:
                tr=(d['fold']!=f) if j is None else (d['fold']!=f)&(d['fold']!=j);train=ws[tr];train.validate()
                assert set(d['person_id'][tr]).isdisjoint(set(d['person_id'][~tr]))
                assert np.all(d['source'][~tr]=='Stanford') and tr[ext].all()
                w=train.side_weights().reshape(-1,2);mass=np.zeros((len(train),2));np.add.at(mass,train.owner,w/w.size)
                target=d['record_weight'][tr]/d['record_weight'][tr].sum()
                err=float(np.max(np.abs(mass.sum(axis=1)-target)));assert err<1e-8;maximum=max(maximum,err)
                # Weighted element loss equals independently averaged record-side losses.
                errors=np.arange(w.size,dtype=float).reshape(w.shape)%37
                records=np.stack([errors[train.owner==q].mean(axis=0) for q in range(len(train))])
                assert abs((errors*w).mean()-np.average(records.mean(axis=1),weights=d['record_weight'][tr]))<1e-5
                checks+=1
    assert np.isfinite(d['x']).all()
a,b=arms['adult'],arms['child']
for k in ['x','y','fold','window_record','window_start','canonical_fallback','person_id','age']:
    n=11261 if k in ['x','window_record','window_start','canonical_fallback'] else 1061
    assert np.array_equal(a[k][:n],b[k][:n]),k
assert a['x'].shape==b['x'].shape and np.array_equal(np.sort(a['record_weight']),np.sort(b['record_weight']))
original=HERE.parents[1]/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
assert sha(original)==read(HERE.parent/'20261105-child100-train-only/preparation.json')['source_data_sha256']
with np.load(original) as z:
    for k in ['x','y','fold','window_record','window_start','canonical_fallback']:
        assert np.array_equal(a[k][:len(z[k])],z[k]),k
    assert np.array_equal(a['person_id'][:1061],np.array(['Stanford:'+str(p) for p in z['person_id']]))
out=dict(passed=True,family_split_checks=checks,maximum_record_mass_error=maximum,exact_matched_people_records_windows=True,Stanford_arrays_identical=True,
    external_train_only=True,Stanford_record_weights_one=True,extra_person_mass_one=True,original_Stanford_data_sha256=sha(original),scope='Input/loss algebra. Actual calibration and serialized model behavior require pilot replay.')
(HERE/'preflight.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))
