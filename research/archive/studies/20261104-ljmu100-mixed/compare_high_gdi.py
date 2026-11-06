"""Measured Stanford high-GDI outcomes only, with paired person-level uncertainty."""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files if k!='x'}
assert sha(BIG/'data.npz')==read(HERE/'bundle/study-protocol.json')['data_sha256']
eligible_source=(d['source']=='Stanford')&(d['label_kind']=='measured_gdi')
assert eligible_source.sum()==1061 and len(set(d['person_id'][eligible_source]))==692
assert not np.any(eligible_source&(d['source']=='LJMU'))
source=HERE.parents[1]/'gdi-research-harness/studies/20261030-gpu-motionbert-study/cnn-multiwindow695/bundle/data.npz'
with np.load(source) as z:
    assert np.array_equal(z['y'],d['y'][eligible_source])
    assert np.array_equal(z['fold'],d['fold'][eligible_source])
    assert np.array_equal(np.array(['Stanford:'+str(p) for p in z['person_id']]),d['person_id'][eligible_source])

audited={};audit_files=[]
for path in (HERE/'audits').glob('*.json'):
    r=read(path)
    if r.get('passed'):
        audited.setdefault(r['family'],set()).update(f['fold'] for f in r['folds'])
        audit_files.append(dict(path=str(path),sha256=sha(path)))
families={name for name,folds in audited.items() if folds==set(range(5))}
models={'CNN_refit':('CNN','refit'),'RF':('RF','refit'),'STT_refit':('STT','refit'),
        'CNN_bag':('CNN','bag'),'STT_bag':('STT','bag'),'bilateral_refit':('bilateral','refit')}
models={name:v for name,v in models.items() if v[0] in families}
provenance=[];predictions={}
for arm in ['baseline','mixed']:
    predictions[arm]={}
    for name,(family,kind) in models.items():
        root=BIG/('runs' if family=='RF' else 'received/data/runs')/arm/family
        p=np.full_like(d['y'],np.nan,dtype=float)
        for f in range(5):
            folder=root/('fold%d'%f);path=folder/'predictions.npz';complete=read(folder/'complete.json')
            assert sha(path)==complete['prediction_sha256']
            with np.load(path) as z:
                assert np.array_equal(z['row_index'],np.flatnonzero(d['fold']==f))
                p[z['row_index']]=z[kind]
            provenance.append(dict(model=name,arm=arm,path=str(path),sha256=sha(path)))
        assert np.isfinite(p).all();predictions[arm][name]=p
    if all(name in models for name in ['CNN_bag','STT_bag','bilateral_refit']):
        p=predictions[arm];fusion=np.empty_like(d['y'],dtype=float)
        for v in read(HERE/'bundle/ensemble.json'):
            m=d['fold']==v['outer'];w=v['cnn_weight'];t=v['fraction']
            base=w*p['CNN_bag'][m]+(1-w)*p['STT_bag'][m];common=base.mean(axis=1);bil=p['bilateral_refit'][m]
            diff=(1-t)*(base[:,0]-base[:,1])+t*(bil[:,0]-bil[:,1])
            fusion[m]=np.stack([common+diff/2,common-diff/2],axis=1)
        predictions[arm]['fusion']=fusion

rows=[];counts=[]
for threshold in [90,100]:
    mask=eligible_source[:,None]&(d['y']>=threshold)
    people=np.unique(d['person_id'][mask.any(axis=1)])
    count=dict(threshold=threshold,people=len(people),records=int(mask.any(axis=1).sum()),side_observations=int(mask.sum()))
    assert (len(people),int(mask.sum()))==((162,255) if threshold==90 else (36,44))
    counts.append(count)
    draws=np.random.default_rng(20261105).integers(0,len(people),(10000,len(people)))
    for name in predictions['baseline']:
        individual={};metrics={}
        for arm in ['baseline','mixed']:
            error=predictions[arm][name]-d['y'];values=[]
            for person in people:
                e=error[mask&(d['person_id']==person)[:,None]]
                values.append([np.abs(e).mean(),np.maximum(-e,0).mean(),e.mean(),(e<0).mean()])
            individual[arm]=np.array(values)
        for k,metric in enumerate(['mae','underprediction_shortfall','bias_prediction_minus_truth','underprediction_fraction']):
            before=individual['baseline'][:,k];after=individual['mixed'][:,k];delta=after-before
            metrics[metric]=dict(before=float(before.mean()),after=float(after.mean()),delta=float(delta.mean()),
                delta95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
        # Independent flattened weighting check: eligible side-records share one person's total weight.
        row_index,side_index=np.nonzero(mask);sample_ids=d['person_id'][row_index]
        weights=np.array([1/(len(people)*np.sum(sample_ids==person)) for person in sample_ids])
        assert abs(weights.sum()-1)<1e-12
        for arm,key in [('baseline','before'),('mixed','after')]:
            errors=predictions[arm][name][row_index,side_index]-d['y'][row_index,side_index]
            for metric,values in [('mae',np.abs(errors)),('underprediction_shortfall',np.maximum(-errors,0)),
                                  ('bias_prediction_minus_truth',errors),('underprediction_fraction',(errors<0).astype(float))]:
                assert abs(float(weights@values)-metrics[metric][key])<1e-10
        rows.append(dict(model=name,**count,metrics=metrics))

out=dict(status='verified_completed_models_only',evaluation_source='Stanford measured GDI only; all LJMU and all child data excluded',
    grouping='Threshold applied to actual GDI of each record-side, not to predictions or person mean. Within-person eligible errors averaged first, then equal-person mean.',
    underprediction_definition='mean(max(actual GDI - predicted GDI, 0)); overshoots contribute zero, so cannot cancel undershoots. Bias is reported separately.',
    uncertainty='Paired person bootstrap10000 seed20261105; descriptive 95% intervals, not adjusted for multiple comparisons; conditional on fitted development OOF models, not independent final validation.',
    data_sha256=sha(BIG/'data.npz'),counts=counts,rows=rows,prediction_provenance=provenance,audit_provenance=audit_files,
    independent_weighted_recalculation_passed=True)
(HERE/'high-gdi-comparison.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'counts':counts,'rows':rows},indent=2))
