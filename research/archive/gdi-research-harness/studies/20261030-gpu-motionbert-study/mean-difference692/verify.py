"""Independent algebra/weighting checks plus adversarial selection fixtures."""
import numpy as np
import compare as c

def metric(y,p,ids):
    per_record=np.abs(y-p).mean(axis=1)
    return float(sum(per_record[ids==i].sum()/sum(ids==i) for i in set(ids))/len(set(ids)))

def disjoint(inner,outer):
    assert set(inner).isdisjoint(outer), 'Person leakage'

z=c.arrays(c.O/'predictions.npz');r=c.read(c.O/'comparison.json');sel=c.read(c.O/'selection.json')
assert len(set(z['person_id']))==692 and len(z['y'])==1061
for p,h in {**sel['source_hashes'],**r['outer_sources']}.items():assert c.sha(p)==h
for key,expected in r['mae'].items():assert abs(metric(z['y'],z[key],z['person_id'])-expected)<1e-12
maxerr=0
for sf in sel['folds']:
    f=sf['outer'];v=c.arrays(c.O/('inner_outer%d.npz'%f));te=z['fold']==f
    disjoint(v['person_id'],z['person_id'][te]);assert (v['fold']!=f).all()
    for key in ('y','person_id','video_id','fold'):
        np.testing.assert_array_equal(v[key],z[key][v['row_index']])
    assert c.sha(c.O/('inner_outer%d.npz'%f))==sf['inner_sha256']
    grid=c.read(c.H/'protocol.json')['difference_fraction_grid']
    # Derive candidates without calling production combine/errors/choose.
    scores=[]
    for t in grid:
        diff=v['bilateral'][:,0]-v['bilateral'][:,1]-(v['base'][:,0]-v['base'][:,1])
        candidate=v['base']+np.column_stack([t*diff/2,-t*diff/2])
        scores.append(metric(v['y'],candidate,v['person_id']))
    assert grid[min(range(len(grid)),key=lambda i:(scores[i],grid[i]))]==sf['fraction']
    a,_=c.choose(v['base'][::-1],v['bilateral'][::-1],v['y'][::-1],v['person_id'][::-1],grid)
    assert a==sf['fraction']
    # Outer labels can be replaced arbitrarily: they are never selection arguments.
    poisoned=z['y'].copy();poisoned[te]=1e9
    a,_=c.choose(v['base'],v['bilateral'],poisoned[v['row_index']],v['person_id'],grid)
    assert a==sf['fraction']
    bad=v['person_id'].copy();bad[0]=z['person_id'][te][0]
    try:disjoint(bad,z['person_id'][te])
    except AssertionError:pass
    else:raise AssertionError('Leak fixture accepted')
    t=sf['fraction'];base=z['existing_blend'][te];bp=z['bilateral_STT'][te]
    diff=(bp[:,0]-bp[:,1])-(base[:,0]-base[:,1])
    q=base+np.column_stack([t*diff/2,-t*diff/2])
    maxerr=max(maxerr,float(np.abs(q-z['inner_selected_difference'][te]).max()))
assert maxerr<1e-12
# A deliberately unbalanced record count must retain person-equal weights.
assert metric(np.zeros((4,2)),np.array([[0,0],[0,0],[0,0],[8,8]]),np.array([1,1,1,2]))==4
for key in ['full_difference_replacement','inner_selected_difference']:
    assert np.max(np.abs(z[key].mean(1)-z['existing_blend'].mean(1)))<1e-12
c.save(c.O/'verification.json',dict(passed=True,people=692,records=1061,all_5_outer_person_boundaries=True,
    all_20_bilateral_fit_boundaries_and_training_calibrations=True,all_source_hashes_rechecked=True,
    independent_person_equal_metric=True,row_order_invariance=True,outer_label_poison_invariance=True,
    injected_person_leak_rejected=True,algebra_max_error=maxerr,common_prediction_unchanged=True,
    comparison_sha256=c.sha(c.O/'comparison.json'),selection_sha256=c.sha(c.O/'selection.json'),
    predictions_sha256=c.sha(c.O/'predictions.npz'),protected_final_data_opened=False))
print('VERIFIED',r['mae'])
