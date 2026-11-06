"""Describe input/error associations without changing any data or model."""
import sys
import csv
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
import cpu_head as m
import numpy as np
from scipy.stats import rankdata

BLEND = HERE.parent / 'pls-gpu-equal-blend695'
GPU = m.ROOT / 'studies/20261030-gpu-motionbert-study'
INPUT = GPU / 'expansion695/train_only.npz'
CONTRACT = GPU / 'cnn-followup695/bundle/input_contract.json'
FEATURES = m.load_json(HERE / 'protocol.json')['input_features']


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def corr(a, b):
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def features(x):
    upper, legs = [9,10,11,12,13,14,15,16], [1,2,3,4,5,6]
    xy, c = x[..., :2].astype(np.float64), x[..., 2]
    observed = c > 0
    boundary = (abs(xy) >= 1-1e-6).any(-1)
    leg_xy = xy[:, legs]
    lengths = np.stack([np.linalg.norm(xy[:, a] - xy[:, b], axis=-1) for a,b in [(1,2),(2,3),(4,5),(5,6)]], 1)
    assert (lengths.mean(0) > 0).all()
    return dict(upper_missing_fraction=float((c[:, upper] == 0).mean()),
                upper_confidence_mean=float(c[:, upper].mean()),
                leg_missing_fraction=float((c[:, legs] == 0).mean()),
                leg_confidence_mean=float(c[:, legs].mean()),
                observed_boundary_fraction=float((boundary & observed).sum() / observed.sum()),
                missing_boundary_fraction_all_points=float((boundary & ~observed).mean()),
                leg_step_p95=float(np.percentile(np.linalg.norm(np.diff(leg_xy, axis=0), axis=-1),95)),
                leg_acceleration_p95=float(np.percentile(np.linalg.norm(np.diff(leg_xy,n=2,axis=0),axis=-1),95)),
                leg_length_cv_mean=float(np.mean(lengths.std(0) / lengths.mean(0))))


def main():
    assert not (HERE / 'summary.json').exists(), 'Preserve completed diagnostic'
    contract = m.load_json(CONTRACT)
    assert m.digest(INPUT) == contract['source_dataset_sha256']
    source_paths = [INPUT, CONTRACT, HERE / 'protocol.json', m.WORKSPACE / 'tools/stanford-sample-check/prepare_ridge_data.py',
                    m.ROOT / 'short_window_contract.py', m.WORKSPACE / 'outputs/motionbert-smoke-20261015/assets/mmpose_inference_3d.py']
    for name, digest in m.load_json(BLEND / 'artifact_manifest.json').items():
        assert m.digest(BLEND / name) == digest, name
        source_paths.append(BLEND / name)
    hashes = {str(p.relative_to(m.WORKSPACE)).replace('\\','/'): m.digest(p) for p in source_paths}
    save('source_hashes.json', hashes)
    assert m.load_json(BLEND / 'verification.json')['verified']
    p, d = load_npz(BLEND / 'oof_predictions.npz'), load_npz(INPUT)
    for k in ['person_id','video_id','fold','y']:
        np.testing.assert_array_equal(p[k], d[k])
    x = d['x'][:,0]
    assert x.shape == (1064,124,17,3) and np.isfinite(x).all()
    assert (x[:,:,1:7,2] > 0).all()
    people = np.unique(p['person_id'])
    preds = p['predictions'][[p['methods'].tolist().index(k) for k in ['pls32','gpu_last_stage']]]
    errs = abs(preds - p['y'])
    record_features = [features(a) for a in x]
    rec = [dict(person_id=str(p['person_id'][i]),video_id=str(p['video_id'][i]),fold=int(p['fold'][i]),**f) for i,f in enumerate(record_features)]
    train_means = {}
    for fold in range(5):
        tr = p['fold'] != fold
        ids = np.unique(p['person_id'][tr])
        assert len(ids)==556 and not set(ids)&set(p['person_id'][~tr])
        train_means[fold] = np.mean([p['y'][p['person_id']==pid].mean(0) for pid in ids],0)
    rows = []
    for pid in people:
        idx = np.flatnonzero(p['person_id']==pid)
        assert len(set(p['fold'][idx]))==1
        f=int(p['fold'][idx[0]])
        y=p['y'][idx]
        e=errs[:,idx].mean((1,2))
        row=dict(person_id=str(pid), fold=f, cohort=int(d['cohort'][idx[0]]), records=len(idx),
                 actual_mean_GDI=float(y.mean()), target_distance_from_outer_training_mean=float(abs(y-train_means[f]).mean()),
                 actual_left_right_difference=float(abs(y[:,0]-y[:,1]).mean()),
                 pls_MAE=float(e[0]), gpu_MAE=float(e[1]), common_error=float(min(e)))
        row.update({k:float(np.mean([record_features[i][k] for i in idx])) for k in FEATURES})
        rows.append(row)
    top=sorted(range(695),key=lambda i:(-rows[i]['common_error'],rows[i]['person_id']))[:70]
    for i,row in enumerate(rows):
        row['review_top70']=i in top
    names=FEATURES+['actual_mean_GDI','target_distance_from_outer_training_mean','actual_left_right_difference']
    strata={}
    for i,row in enumerate(rows):
        strata.setdefault((int(np.floor(row['actual_mean_GDI']/10)),row['cohort']),[]).append(i)
    selected=[indices for indices in strata.values() if len(indices)>=5]
    within_idx=np.array([i for indices in selected for i in indices])
    def centered(a):
        ranks=rankdata(a)
        return np.concatenate([ranks[indices]-ranks[indices].mean() for indices in selected])
    associations=[]
    for feature in names:
        a=np.array([row[feature] for row in rows])
        for outcome in ['pls_MAE','gpu_MAE','common_error']:
            b=np.array([row[outcome] for row in rows])
            associations.append(dict(feature=feature,outcome=outcome,people=695,
                spearman=corr(rankdata(a),rankdata(b)),within_score_cohort_rank_correlation=corr(centered(a),centered(b)),
                within_people=len(within_idx),constant_feature=bool(np.ptp(a)<1e-12)))
    distribution=[]
    for name in names:
        a=np.array([row[name] for row in rows])
        distribution.append(dict(feature=name,min=float(a.min()),median=float(np.median(a)),max=float(a.max()),
                                 top70_mean=float(a[top].mean()),other625_mean=float(a[[i for i in range(695) if i not in top]].mean())))
    joints=[]
    for j,name in enumerate(contract['joints']):
        c=x[:,:,j,2]
        joints.append(dict(joint=j,name=name,zero_confidence_points=int((c==0).sum()),
                           records_with_missing=int((c==0).any(1).sum()),mean_confidence=float(c.mean())))
    boundary=(abs(x[...,:2].astype(np.float64))>=1-1e-6).any(-1)
    inventory=dict(people=695,records=1064,frames_per_record=124,joints=17,
                   zero_confidence_points=int((x[...,2]==0).sum()),leg_zero_confidence_points=int((x[:,:,1:7,2]==0).sum()),
                   numeric_xy_outside_bounds=int((abs(x[...,:2])>1).sum()),
                   observed_boundary_points=int((boundary&(x[...,2]>0)).sum()),
                   missing_boundary_points=int((boundary&(x[...,2]==0)).sum()))
    assert inventory['zero_confidence_points']==contract['missing_confidence_points']
    for name,data in [('record_input_features.csv',rec),('person_input_errors.csv',rows),('associations.csv',associations),
                       ('feature_summary.csv',distribution),('joint_inventory.csv',joints),('review_top70.csv',[rows[i] for i in top])]:
        table(name,data)
    for path,digest in hashes.items():
        assert m.digest(m.WORKSPACE/path)==digest,path
    save('summary.json',dict(inventory=inventory,associations=associations,feature_summary=distribution,
                            joint_inventory=joints,within_strata=[dict(score_bin=k[0],cohort=k[1],people=len(v)) for k,v in strata.items()],
                            review_top70_mean_common_error=float(np.mean([rows[i]['common_error'] for i in top])),
                            reference_MAE=dict(pls=float(np.mean([r['pls_MAE'] for r in rows])),gpu=float(np.mean([r['gpu_MAE'] for r in rows]))),
                            preserved_source_files=len(hashes),training=False,GPU_used=False,causal_validation=False))
    print(json.dumps({'inventory':inventory,'common_error_associations':[v for v in associations if v['outcome']=='common_error'],
                      'feature_summary':distribution},indent=2))


if __name__=='__main__':
    main()
