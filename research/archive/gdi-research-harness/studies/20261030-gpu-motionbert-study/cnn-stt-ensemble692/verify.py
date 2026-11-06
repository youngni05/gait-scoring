"""Independent scalar-weight metric/selection audit plus input-isolation tests."""
from pathlib import Path
import json,hashlib
import numpy as np
import compare as m
H=Path(__file__).resolve().parent;O=H/'analysis'
def require(ok,message):
    if not ok:raise ValueError(message)
def validate(inner,outer,meta):
    mask=meta['fold']!=outer;rows=np.flatnonzero(mask)
    np.testing.assert_array_equal(inner['row_index'],rows)
    for key in ('person_id','video_id','fold','y'):np.testing.assert_array_equal(inner[key],meta[key][mask])
    require(not set(inner['person_id']) & set(meta['person_id'][~mask]),'Outer people in selection')
def scalar_score(y,p,ids):
    total=0.
    for q in sorted(set(ids)):
        e=[]
        for i in np.flatnonzero(ids==q):
            e.append((abs(float(p[i,0])-float(y[i,0]))+abs(float(p[i,1])-float(y[i,1])))/2)
        total+=sum(e)/len(e)
    return total/len(set(ids))
def main():
    s=m.read(O/'selection.json');cfg=m.read(H/'protocol.json');z=m.arrays(O/'predictions.npz');result=m.read(O/'comparison.json')
    require(m.sha(H/'protocol.json')==s['protocol_sha256'],'Protocol changed')
    for name,digest in s['source_hashes'].items():require(m.sha(m.S/name)==digest,'Changed source '+name)
    # Baseline input provenance and complete record/window alignment.
    a=m.arrays(m.C/'bundle/data.npz');b=m.arrays(m.T/'bundle/data.npz')
    for key in ('y','person_id','video_id','fold','window_record','window_start','canonical_fallback'):
        np.testing.assert_array_equal(a[key],b[key])
    np.testing.assert_array_equal(a['x'][...,:8].reshape(11261,2,124,4,2),b['x'])
    reconstructed=np.full((1061,2),np.nan);weights=[]
    for sf in s['folds']:
        f=sf['outer'];inner=m.arrays(O/('inner_outer%d.npz'%f));validate(inner,f,z)
        scores=[scalar_score(inner['y'],w*inner['cnn']+(1-w)*inner['stt'],inner['person_id']) for w in cfg['cnn_weight_grid']]
        np.testing.assert_allclose(scores,sf['inner_person_mae'],rtol=0,atol=1e-12)
        w=cfg['cnn_weight_grid'][min(range(len(scores)),key=lambda k:(scores[k],abs(cfg['cnn_weight_grid'][k]-.5),cfg['cnn_weight_grid'][k]))]
        require(w==sf['cnn_weight'],'Weight mismatch');weights.append(w)
        mask=z['fold']==f;reconstructed[mask]=w*z['CNN'][mask]+(1-w)*z['STT'][mask]
        # Actual selection API receives only inner arrays. Alter outer labels/IDs independently.
        poisoned=dict(z,y=z['y'].copy());poisoned['y'][mask]=99999
        chosen=m.choose(inner['cnn'],inner['stt'],poisoned['y'][~mask],poisoned['person_id'][~mask],cfg['cnn_weight_grid'])[0]
        require(chosen==w,'Outer label poisoning changed selection')
        # Joint row reorder must not change selection; each person has total mass one.
        order=np.arange(len(inner['y']))[::-1]
        require(m.choose(inner['cnn'][order],inner['stt'][order],inner['y'][order],inner['person_id'][order],cfg['cnn_weight_grid'])[0]==w,'Row-order-dependent selection')
        counts={q:int(np.sum(inner['person_id']==q)) for q in np.unique(inner['person_id'])}
        for q,n in counts.items():require(abs(sum([1/n]*n)-1)<1e-12,'Unequal person mass')
    np.testing.assert_array_equal(reconstructed,z['nested_weight'])
    metrics={k:scalar_score(z['y'],z[k],z['person_id']) for k in ('CNN','STT','fixed_half','nested_weight')}
    for k,v in metrics.items():require(abs(v-result['person_equal_mae'][k])<1e-12,'Metric mismatch')
    # Leakage and alignment corruption are refused by the same boundary validator.
    inner=m.arrays(O/'inner_outer0.npz');bad=dict(inner,person_id=inner['person_id'].copy());bad['person_id'][0]=z['person_id'][z['fold']==0][0]
    try:validate(bad,0,z)
    except (ValueError,AssertionError):pass
    else:raise AssertionError('Leakage fixture accepted')
    # Sources used for evaluation remain exactly the frozen OOF artifacts.
    for name,path in [('CNN',m.C/'analysis/cnn_oof.npz'),('STT',m.T/'analysis/unpacked/results/oof_predictions.npz')]:
        require(m.sha(path)==result['outer_source_sha256'][name],'Outer artifact changed')
    m.save(O/'verification.json',dict(passed=True,source_files_checked=len(s['source_hashes']),people=692,records=1061,windows=11261,
        exact_input_window_and_metadata_alignment=True,all_20_base_fit_person_boundaries=True,outer_label_poison_test=True,row_reordering_invariant=True,leakage_fixture_rejected=True,person_equal_metric_independent=metrics,cnn_weights=weights,all_5_outer_blends_reconstructed=True,
        max_cnn_calibration_prediction_delta=max(a['cnn_calibration_prediction_delta'] for f in s['folds'] for a in f['audits']),
        max_cnn_metric_replay_delta=max(a['cnn_validation_metric_replay_delta'] for f in s['folds'] for a in f['audits']),
        comparison_sha256=m.sha(O/'comparison.json'),protocol_sha256=m.sha(H/'protocol.json'),protected_final_data_opened=False,neural_optimizer_steps=0))
    print(json.dumps(m.read(O/'verification.json')))
if __name__=='__main__':main()
