"""Post-hoc diagnostics; no network retraining or changes to registered comparisons."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def coefficient(p,y):return np.linalg.lstsq(np.column_stack([p.reshape(-1),np.ones(p.size)]),y.reshape(-1),rcond=None)[0]
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
assert sha(BIG/'data.npz')==read(HERE/'bundle/study-protocol.json')['data_sha256']
y=d['y'];ids=d['person_id'];stan=d['source']=='Stanford';ljmu=~stan;fold=d['fold']
with np.load(ROOT/'gdi-research-harness/studies/20261031-ljmu-normal-cnn/diagnostics/development_age_metadata.npz') as z:
    assert np.array_equal(d['source_person_id'][stan].astype(str),z['person_id'].astype(str))
    age=np.full(len(y),np.nan);age[stan]=z['age']
lj=read(HERE/'ljmu-membership.json');assert len(lj)==int(ljmu.sum());age[ljmu]=[r['age'] for r in lj]
def metric(p,mask):
    people=np.unique(ids[mask.any(axis=1)])
    if not len(people):return None
    errors=[(p-y)[mask&(ids==q)[:,None]] for q in people]
    return dict(people=len(people),side_observations=int(mask.sum()),mae=float(np.mean([np.abs(e).mean() for e in errors])),
                bias=float(np.mean([e.mean() for e in errors])),shortfall=float(np.mean([np.maximum(-e,0).mean() for e in errors])))
groups={'Stanford_all':np.broadcast_to(stan[:,None],y.shape),'Stanford_ge90':stan[:,None]&(y>=90),'Stanford_ge100':stan[:,None]&(y>=100),
        'Stanford_lt70':stan[:,None]&(y<70),'LJMU':np.broadcast_to(ljmu[:,None],y.shape)}
for agegroup,am in [('under18',age<18),('adult',age>=18),('age5to12',(age>=5)&(age<13)),('age13to17',(age>=13)&(age<18))]:
    for score,sm in [('all',np.ones_like(y,bool)),('ge90',y>=90),('ge100',y>=100)]:groups[agegroup+'_'+score]=(stan&am)[:,None]&sm
baseline_members=read(HERE/'bundle/baseline-models.json')
out={};calibrations=[];epoch_analysis=[]
for family in ['CNN','STT']:
    # Main diagnostics retain trained weights, selected epochs and original bag membership.
    pred={arm:{kind:np.full_like(y,np.nan,dtype=float) for kind in ['refit','bag']} for arm in ['baseline','mixed','mixed_Stanford_only_calibration']}
    raw={arm:np.full_like(y,np.nan,dtype=float) for arm in ['baseline','mixed']}
    for f in range(5):
        te=fold==f;folder=BIG/'received/data/runs/mixed'/family/f'fold{f}'
        done=read(folder/'complete.json');assert done['bundle_sha256']==sha(HERE/'bundle/manifest.json')
        for arm in ['baseline','mixed']:
            pf=BIG/'received/data/runs'/arm/family/f'fold{f}'/'predictions.npz'
            assert sha(pf)==read(pf.parent/'complete.json')['prediction_sha256']
            with np.load(pf) as z:
                assert np.array_equal(z['row_index'],np.flatnonzero(te))
                for kind in ['refit','bag']:pred[arm][kind][te]=z[kind]
                raw[arm][te]=z['refit_raw']
        diagnostic_bag=[]
        for m in done['members']:
            tag=m['tag'];p=folder/tag/'replay.npz';j=None if tag=='refit' else int(tag[5:])
            tr=(fold!=f) if j is None else (fold!=f)&(fold!=j)
            with np.load(p) as z:
                assert np.array_equal(z['train_row'],np.flatnonzero(tr)) and np.array_equal(z['evaluation_row'],np.flatnonzero(te))
                tp=z['train_raw'];op=z['evaluation_raw']
                allcoef=coefficient(tp,y[tr]);sm=stan[tr];sc=coefficient(tp[sm],y[tr][sm])
                assert np.max(np.abs(allcoef-m['calibration']))<1e-8
                assert not set(ids[tr]) & set(ids[te])
                # Independently verify the two-parameter least-squares solution.
                xx=tp[sm].reshape(-1);yy=y[tr][sm].reshape(-1)
                slope=np.dot(xx-xx.mean(),yy-yy.mean())/np.dot(xx-xx.mean(),xx-xx.mean())
                assert np.max(np.abs(sc-[slope,yy.mean()-slope*xx.mean()]))<1e-8
                diagnostic=op*sc[0]+sc[1]
                if tag=='refit':pred['mixed_Stanford_only_calibration']['refit'][te]=diagnostic
                if not m['excluded_from_bag']:diagnostic_bag.append(diagnostic)
                calibrations.append(dict(family=family,fold=f,member=tag,mixed_coef=allcoef.tolist(),Stanford_only_coef=sc.tolist(),
                    LJMU_training_record_fraction=float(ljmu[tr].mean()),LJMU_training_person_fraction=len(set(ids[tr&ljmu]))/len(set(ids[tr])),
                    posthoc_evaluation_shift_stanford=float((diagnostic-(op*allcoef[0]+allcoef[1]))[stan[te]].mean())))
        pred['mixed_Stanford_only_calibration']['bag'][te]=np.mean(diagnostic_bag,axis=0)
        # Epoch diagnostics need each epoch's raw validation outputs, available for STT.
        if family=='STT':
            curves=[];stan_curves=[];weights=[];stan_weights=[]
            for j in range(5):
                if j==f:continue
                h=read(folder/f'inner{j}'/'curve.json');va=fold==j
                curves.append([r['validation_person_mae'] for r in h]);weights.append(len(set(ids[va])))
                curve=[]
                for row in h:
                    p=folder/f'inner{j}'/'raw_predictions'/('epoch%04d.npz'%row['epoch'])
                    assert sha(p)==row['raw_predictions_sha256']
                    with np.load(p) as z:vp=z['validation_raw']*row['calibration'][0]+row['calibration'][1]
                    yy=y[va][stan[va]];pp=vp[stan[va]];ii=ids[va][stan[va]]
                    curve.append(float(np.mean([np.abs(pp[ii==q]-yy[ii==q]).mean() for q in np.unique(ii)])))
                stan_curves.append(curve);stan_weights.append(len(set(ids[va&stan])))
            orig=np.average(curves,axis=0,weights=weights);scurve=np.average(stan_curves,axis=0,weights=stan_weights)
            selected=int(np.argmin(orig))+1;alternate=int(np.argmin(scurve))+1
            assert selected==done['selected_epoch']
            epoch_analysis.append(dict(family=family,fold=f,original_epoch=selected,Stanford_only_validation_epoch=alternate,
                Stanford_inner_mae_at_original=float(scurve[selected-1]),Stanford_inner_min_mae=float(scurve[alternate-1]),
                note='Diagnostic selection from inner Stanford rows only. Alternate refit not trained; not an OOF performance result.'))
    for kind in ['refit','bag']:
        out[family+'_'+kind]={arm:{g:metric(v[kind],mask) for g,mask in groups.items()} for arm,v in pred.items()}
    out[family+'_refit_raw']={arm:{g:metric(v,mask) for g,mask in groups.items()} for arm,v in raw.items()}

# Baseline caches contain train and held-out predictions from frozen original models.
# Identify the aggregation matching each saved calibration rather than assume an axis index.
train_generalization=[]
for family in ['CNN','STT']:
    for f in range(5):
        old=next(m for m in baseline_members if m['family']==family and m['fold']==f and m['tag']=='refit')
        p=Path(old['cache']);assert sha(p)==old['cache_sha256']
        with np.load(p) as z:
            assert np.array_equal(z['y'],y[stan])
            index=int(np.argmin(np.max(np.abs(z['calibration']-np.array(old['calibration'])),axis=1)))
            assert np.max(np.abs(z['calibration'][index]-old['calibration']))<1e-3
            bp=z['corrected'][index];braw=z['aggregate_raw'][index];tr=z['train_mask']
            base={}
            for scope,m in [('train',tr),('heldout',fold[stan]==f)]:
                for threshold in [90,100]:
                    em=m[:,None]&(y[stan]>=threshold);pp=np.full_like(y,np.nan);pp[stan]=bp;mask=np.zeros_like(y,bool);mask[stan]=em
                    base[scope+'_ge'+str(threshold)]=metric(pp,mask)
        folder=BIG/'received/data/runs/mixed'/family/f'fold{f}'
        m=next(m for m in read(folder/'complete.json')['members'] if m['tag']=='refit');coef=np.array(m['calibration'])
        with np.load(folder/'refit/replay.npz') as z:
            mp=np.full_like(y,np.nan,dtype=float);mp[z['train_row']]=z['train_raw']*coef[0]+coef[1];mp[z['evaluation_row']]=z['evaluation_raw']*coef[0]+coef[1]
        mix={}
        for scope,rm in [('train',fold!=f),('heldout',fold==f)]:
            for threshold in [90,100]:mix[scope+'_ge'+str(threshold)]=metric(mp,(stan&rm)[:,None]&(y>=threshold))
        train_generalization.append(dict(family=family,fold=f,baseline=base,mixed=mix))

age_summary={}
for source,mask in [('Stanford',stan),('LJMU',ljmu)]:
    people=np.unique(ids[mask]);earliest=np.array([age[mask&(ids==p)].min() for p in people])
    age_summary[source]=dict(people=len(people),records=int(mask.sum()),earliest_observed_age_mean=float(earliest.mean()),earliest_observed_age_median=float(np.median(earliest)),
        age_min=float(age[mask].min()),age_max=float(age[mask].max()),earliest_observed_under18_people=int((earliest<18).sum()))
result=dict(scope='Post-hoc diagnostic. Original models, epoch choices, predictions and main results unchanged. No final223 accessed.',
    age=age_summary,overall_LJMU_record_share=float(ljmu.mean()),overall_LJMU_person_share=len(set(ids[ljmu]))/len(set(ids)),
    models=out,calibration_diagnostics=calibrations,STT_epoch_diagnostics=epoch_analysis,train_heldout_diagnostics=train_generalization)
matches=0
for row in read(HERE/'high-gdi-comparison.json')['rows']:
    if row['model'] not in out:continue
    for previous,current in [('before','baseline'),('after','mixed')]:
        assert abs(row['metrics']['mae'][previous]-out[row['model']][current]['Stanford_ge'+str(row['threshold'])]['mae'])<1e-10
        matches+=1
assert matches==16
result['verification']=dict(original_high_gdi_metric_matches=matches,Stanford_only_OLS_closed_form_verified=True,calibration_train_evaluation_person_disjoint=True)
(HERE/'transfer-diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'age':age_summary,'verification':result['verification']},indent=2))
