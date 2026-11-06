"""Check approved exception scope, original feature equality and frozen prediction stability."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;DATA=Path('D:/gait-research-data/D_P/datasets/GAVD')
ALLOWED={'cljxjv25j001a3n6lw4iv9fwe','cll7ob018002y3o6lngai3iq4'}
def rows(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
results=[]
for base,shift in [('primary',0),('shift1',1)]:
    old=DATA/'runs'/base;new=DATA/'runs'/('confidence_'+base)
    audit=json.loads((new/'prepared/confidence_exception.json').read_text(encoding='utf-8'))
    assert set(audit['used_records'])<=ALLOWED
    assert not rows(new/'prepared/excluded.csv')
    with np.load(old/'prepared/prepared.npz',allow_pickle=False) as a,np.load(new/'prepared/prepared.npz',allow_pickle=False) as b:
        old_ids=a['record_id'].tolist();new_ids=b['record_id'].tolist();added=set(new_ids)-set(old_ids)
        assert added==set(audit['used_records'])
        for ident in old_ids:
            i=old_ids.index(ident);j=new_ids.index(ident);amask=a['window_record']==i;bmask=b['window_record']==j
            for key in ['x','window_start','frame_first','frame_last','window_missing_fraction']:
                assert np.array_equal(a[key][amask],b[key][bmask]),(base,ident,key)
            assert a['fps'][i]==b['fps'][j]
    oldpred={(r['record_id'],r['fold']):r for r in rows(old/'predictions/record_predictions.csv')}
    newpred={(r['record_id'],r['fold']):r for r in rows(new/'predictions/record_predictions.csv')}
    maximum=max(abs(float(r[k])-float(newpred[key][k])) for key,r in oldpred.items() for k in ['left_GDI','right_GDI','left_raw','right_raw'])
    assert maximum<=1e-4,'Original frozen inference replay tolerance exceeded'
    results.append(dict(run=base,original_records=len(old_ids),added_records=sorted(added),existing_features_bitwise_equal=True,existing_predictions_max_error=maximum,inference_tolerance=1e-4))
for freeze in ['primary_prescore_freeze.json','shift1_prescore_freeze.json']:
    audit=json.loads((HERE/freeze).read_text())
    assert all(hashlib.sha256((HERE/f).read_bytes()).hexdigest()==h for f,h in audit['sha256'].items())
bundle=HERE.parents[2]/'outputs/team-share-gdi-normal-cnn-20261031-v2/bundle'
files=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))['files']
assert all(hashlib.sha256((bundle/f).read_bytes()).hexdigest()==h for f,h in files.items())
out=dict(status='passed',runs=results,all_prescore_selections_unchanged=True,frozen_bundle_files_verified=len(files),frozen_bundle_unchanged=True)
(HERE/'results/official_confidence/exception_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
