"""Compare frozen-CNN scores by view, preserving source-video/person clusters.

Use after real preparation/inference. No threshold fitting or GDI ground truth.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from prepare_comparison import PATHOLOGICAL, VIEWS, read_csv
from audit_gavd import write_json, write_csv


def effect(normal, abnormal, wn=None, wa=None):
    n=np.asarray(normal,dtype=float);a=np.asarray(abnormal,dtype=float)
    if not len(n) or not len(a):raise ValueError('Both classes required')
    wn=np.ones(len(n)) if wn is None else np.asarray(wn,dtype=float)
    wa=np.ones(len(a)) if wa is None else np.asarray(wa,dtype=float)
    if wn.sum()<=0 or wa.sum()<=0:raise ValueError('Both classes require positive weight')
    pairs=(n[:,None]>a[None,:]).astype(float)+.5*(n[:,None]==a[None,:])
    auc=float(np.sum(pairs*wn[:,None]*wa[None,:])/(wn.sum()*wa.sum()))
    return float(np.average(n,weights=wn)-np.average(a,weights=wa)),auc


def summarize(rows, repeats, seed):
    # One observation per dependency cluster and label in the requested view.
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['cluster_id'],r['label'])].append(r['score'])
    values=[dict(cluster_id=k[0],label=k[1],score=float(np.mean(v)),records=len(v)) for k,v in sorted(groups.items())]
    n=[r for r in values if r['label']=='Normal Gait'];a=[r for r in values if r['label']=='Abnormal Gait']
    if not n or not a:return dict(status='insufficient_classes',normal_clusters=len(n),abnormal_clusters=len(a))
    ns=np.array([r['score'] for r in n]);ass=np.array([r['score'] for r in a])
    delta,auc=effect(ns,ass)
    ids=sorted({r['cluster_id'] for r in values});index={k:i for i,k in enumerate(ids)}
    ni=np.array([index[r['cluster_id']] for r in n]);ai=np.array([index[r['cluster_id']] for r in a])
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats if len(n)>=2 and len(a)>=2 else 0):
        counts=np.bincount(rng.integers(0,len(ids),size=len(ids)),minlength=len(ids))
        if counts[ni].sum() and counts[ai].sum():boot.append(effect(ns,ass,counts[ni],counts[ai]))
    ci=np.quantile(np.asarray(boot),[.025,.975],axis=0) if boot else None
    return dict(status='exploratory',normal_clusters=len(n),abnormal_clusters=len(a),
                independent_cluster_ids=len(ids),normal_mean=float(ns.mean()),abnormal_mean=float(ass.mean()),
                normal_median=float(np.median(ns)),abnormal_median=float(np.median(ass)),
                normal_minus_abnormal=delta,auc_higher_score_is_normal=auc,
                mean_difference_ci95=ci[:,0].tolist() if ci is not None else None,
                auc_ci95=ci[:,1].tolist() if ci is not None else None,
                bootstrap_draws=repeats,valid_bootstrap_draws=len(boot),seed=seed,
                ci_note='Exploratory cluster percentile interval; omitted if either label has fewer than 2 clusters. '
                        'No correction for comparisons across views/types. '
                        'Small cluster counts and unknown cross-video identities limit inference.')


def merge(predictions,labels):
    mapping={r['record_id']:r for r in labels}
    if len(mapping)!=len(labels):raise ValueError('Duplicate label record ID')
    scores=collections.defaultdict(dict)
    for r in predictions:
        key=r['record_id'];fold=int(r['fold'])
        if key not in mapping:raise ValueError('Prediction has no reviewed label: '+key)
        if r['person_id']!=mapping[key]['cluster_id']:raise ValueError('Cluster mismatch: '+key)
        if fold in scores[key]:raise ValueError('Duplicate record/fold: '+key)
        pair=[float(r['left_GDI']),float(r['right_GDI'])]
        if not np.isfinite(pair).all():raise ValueError('Nonfinite score')
        scores[key][fold]=float(np.mean(pair))
    if not scores:raise ValueError('No predictions')
    rows=[]
    for key,folds in scores.items():
        if set(folds)!=set(range(5)):raise ValueError('All five unchanged folds required: '+key)
        r=mapping[key]
        if r['label'] not in {'Normal Gait','Abnormal Gait'} or r['view'] not in VIEWS:
            raise ValueError('Unresolved label/view')
        fps=float(r['fps'])
        if not np.isfinite(fps) or fps<=0:raise ValueError('Missing actual FPS')
        rows.append(dict(r,score=float(np.mean(list(folds.values()))),**{'fold_'+str(k):v for k,v in folds.items()}))
    return rows, sorted(set(mapping)-set(scores))


def main(args):
    rows,missing=merge(read_csv(args.predictions),read_csv(args.labels))
    results={}
    for scope in ['all_abnormal_patterns','pathological_patterns_only']:
        eligible=[r for r in rows if scope=='all_abnormal_patterns' or r['label']=='Normal Gait' or r['gait_pattern'] in PATHOLOGICAL]
        results[scope]={}
        for view in sorted(VIEWS):
            selected=[r for r in eligible if r['view']==view]
            results[scope][view]=dict(
                all_fps_exploratory=summarize(selected,args.bootstrap,args.seed),
                fps_strata={str(fps):summarize([r for r in selected if float(r['fps'])==fps],args.bootstrap,args.seed)
                            for fps in sorted({float(r['fps']) for r in selected})},
                per_fold_descriptive={str(k):summarize([dict(r,score=r['fold_'+str(k)]) for r in selected],0,args.seed)
                                      for k in range(5)})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'record_scores.csv',rows)
    write_json(out/'comparison.json',dict(results=results,approved_without_prediction=missing,
               unit='Reviewed cluster; video-only identity must not be reported as people',
               estimand='Equal cluster weight within label/view; records and then folds/sides averaged',
               actual_GDI_ground_truth=False,classification_threshold_fitted=False))
    print(json.dumps(dict(records=len(rows),missing_predictions=len(missing),output=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--out',required=True);p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261031)
    a=p.parse_args()
    if a.bootstrap<100:p.error('Use at least 100 bootstrap draws')
    main(a)
