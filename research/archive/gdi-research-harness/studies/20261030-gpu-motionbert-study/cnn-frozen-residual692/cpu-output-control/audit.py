"""Reconstruct saved affine predictions and selection without fitting any model."""
from pathlib import Path
import json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def macro(a,ids):return float(np.mean([a[ids==p].mean() for p in np.unique(ids)]))

def main():
    with np.load(ROOT/'oof_predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(ROOT.parent/'bundle/data.npz',allow_pickle=False) as z:
        for k in z.files:
            if k!='x':np.testing.assert_array_equal(z[k],d[k])
    ids=d['person_id'];fold=d['fold'];pred=np.full_like(d['y'],np.nan);seen=np.zeros(len(ids),int)
    summary=read(ROOT/'summary.json')
    for f in range(5):
        root=ROOT/('fold%d'%f);info=read(root/'result.json')
        curves=[];counts=[]
        for j in range(5):
            if j==f:continue
            inner=read(root/('inner%d.json'%j))
            for key,mask in [('train_people',(fold!=f)&(fold!=j)),('validation_people',fold==j),('outer_evaluation_people',fold==f)]:
                assert inner[key]==np.unique(ids[mask]).tolist()
            curves.append(inner['mae']);counts.append(len(inner['validation_people']))
        means=np.average(curves,axis=0,weights=counts)
        np.testing.assert_array_equal(means,info['inner_weighted_mae'])
        assert bool(np.argmin(means))==info['use_correction']
        assert info['train_people']==np.unique(ids[fold!=f]).tolist()
        assert info['outer_evaluation_people']==np.unique(ids[fold==f]).tolist()
        assert not set(info['train_people'])&set(info['outer_evaluation_people'])
        with np.load(root/'predictions.npz',allow_pickle=False) as z:
            ix=z['row_index'];np.testing.assert_array_equal(ix,np.flatnonzero(fold==f))
            np.testing.assert_array_equal(z['y'],d['y'][ix]);np.testing.assert_array_equal(z['baseline'],d['baseline'][ix])
            model=info['model'];base=z['baseline']
            design=np.column_stack(((base-model['mean'])/model['scale'],np.ones(len(base))))
            actual=base+design@model['coefficient'] if info['use_correction'] else base
            np.testing.assert_array_equal(actual,z['predictions'])
            assert abs(macro(np.abs(actual-z['y']),ids[ix])-info['mae'])<1e-12
            pred[ix]=actual;seen[ix]+=1
    assert (seen==1).all()
    np.testing.assert_array_equal(pred,d['predictions'])
    mae=macro(np.abs(pred-d['y']),ids)
    assert abs(mae-summary['metrics']['cnn_output_only_control']['mae'])<1e-12
    result=dict(passed=True,records=1061,people=692,inner_selections_recomputed=5,
        saved_affine_models_replayed=5,all_20_inner_memberships_verified=True,
        outer_memberships_verified=True,all_records_evaluated_once=True,
        mae_recomputed=mae,model_replay_max_error=float(np.abs(pred-d['predictions']).max()),
        oof_sha256=hashlib.sha256((ROOT/'oof_predictions.npz').read_bytes()).hexdigest())
    (ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
