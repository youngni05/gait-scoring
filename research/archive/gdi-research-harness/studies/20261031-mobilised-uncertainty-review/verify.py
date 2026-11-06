"""Recompute audit claims from source forms and ranking from pair comparisons."""
from common import *
import zipfile

report={'checks':{},'new_models_trained':0}
acq=json.loads((HERE/'acquisition.json').read_text(encoding='utf8'))
for entry in acq:
    p=HERE/'raw'/entry['name']
    assert hashlib.md5(p.read_bytes()).hexdigest()==entry['md5']
    assert sha(p)==entry['sha256']
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
report['checks']['source_archives_md5_sha256_crc']=len(acq)

audit=json.loads((HERE/'mobilised-audit.json').read_text(encoding='utf8'))
main_path=HERE/'extracted/Main datasets for analysis/CSV files/All_cohorts_dataset.csv'
main=pd.read_csv(main_path,low_memory=False)
assert sha(main_path)==audit['main_sha256']
keys=['participantid','visit.number']
assert not main.duplicated(keys).any() and main.participantid.nunique()==2315
weekly=pd.concat([pd.read_csv(p) for p in (HERE/'extracted/Aggregated').glob('*weekly_agg_filtered.csv')]).rename(columns={'visit_number':'visit.number'})
assert not weekly.duplicated(keys).any()
weekly=weekly[(weekly.ws_1030_avg_w>0)&(weekly.n_days_w>=3)]
weekly=weekly.merge(main[keys+['proto_dev','cohort']],on=keys,validate='one_to_one')
weekly=weekly[weekly.proto_dev=='N']
selected=pd.read_csv(HERE/'first-eligible-per-target.csv')
details={}
for target,form,label in [('TUG','tug','tugtime'),('STS','sppb','5crtresult'),('SPPB','sppb','total_sppb_score')]:
    if target=='SPPB':
        clinical=main[keys+[label]]
    else:
        clinical=pd.read_csv(HERE/f'extracted/Clinical Data/T1-T5-study-instances-{form}.csv').rename(columns={'visit-number':'visit.number'})[keys+[label]]
    joined=weekly.merge(clinical,on=keys,validate='one_to_one')
    ok=joined[label].between(0,12) if target=='SPPB' else joined[label]>0
    valid=joined[ok].sort_values(keys).drop_duplicates('participantid')
    saved=selected[selected.audit_target==target].sort_values(keys)
    assert valid[keys].reset_index(drop=True).equals(saved[keys].reset_index(drop=True))
    np.testing.assert_allclose(valid[label],saved[label])
    tail=valid[label]<=6 if target=='SPPB' else valid[label]>=20
    assert len(valid)==audit['first_available'][target+'_timing_ok']['people']
    expected=audit['first_available'][target+'_timing_ok'][label]['le6_people' if target=='SPPB' else 'ge20_people']
    assert int(tail.sum())==expected
    details[target]={'people':len(valid),'tail_people':int(tail.sum()),'positive_under3_people':int(((valid[label]>0)&(valid[label]<3)).sum()) if target!='SPPB' else None,
                     't2':{}}
    t2=joined[ok&joined['visit.number'].eq('t2')]
    for cohort,g in t2.groupby('cohort'):
        tail2=g[label]<=6 if target=='SPPB' else g[label]>=20
        details[target]['t2'][cohort]={'people':len(g),'tail_people':int(tail2.sum())}
report['source_linkage_recomputation']=details
report['checks']['source_form_weekly_join_and_first_visit_selection']=True

source=HERE.parent/'20261031-opencap-walk-quality'
rank=pd.read_csv(HERE/'current-heldout-ranking.csv');checked=0;maxdiff=0.
for outcome in ['STS','TUG']:
    folder=source/(outcome+'_common6');pred=pd.read_csv(folder/'predictions.csv')
    data=np.load(folder/'input.npz');ids=data['ids'];targets=dict(zip(ids,data['y']))
    assert 'sub-283' not in ids
    manifest=pd.read_csv(folder/'manifest.csv')
    assert len(manifest)==len(ids) and set(manifest.id)==set(ids)
    assert manifest.seconds.eq(6).all()
    splits=json.loads((folder/'splits.json').read_text(encoding='utf8'))
    for partition in splits:
        tested=[]
        for fold in partition['splits']:
            train=set(fold['train']);test=set(fold['test'])
            assert train.isdisjoint(test) and train|test==set(range(len(ids)))
            tested.extend(test)
            records=pred[(pred['repeat']==partition['repeat'])&(pred.fold==fold['fold'])]
            assert set(records.id)==set(ids[list(test)])
        assert sorted(tested)==list(range(len(ids)))
    for (model,view,seed,repeat),g in pred[pred.view.isin(['speed','quality','pattern','robust'])].groupby(['model','view','seed','repeat']):
        assert set(g.id)==set(ids) and len(g)==len(ids)
        np.testing.assert_allclose(g.target,g.id.map(targets))
        y=g.target.to_numpy();p=g.prediction.to_numpy();pos=p[y>=20];neg=p[y<20]
        auc=float(((pos[:,None]>neg[None,:])+0.5*(pos[:,None]==neg[None,:])).mean())
        r=rank[(rank.outcome==outcome)&(rank.model==model)&(rank.view==view)&(rank.seed==seed)&(rank['repeat']==repeat)].iloc[0]
        maxdiff=max(maxdiff,abs(auc-r.auroc));assert abs(auc-r.auroc)<1e-12
        assert r.tp==int(((y>=20)&(p>=20)).sum())
        assert r.fp==int(((y<20)&(p>=20)).sum())
        assert r.fn==int(((y>=20)&(p<20)).sum())
        # Independently compute step-integrated precision/recall with tied scores grouped.
        order=np.argsort(-p,kind='stable');sp=p[order];sy=y[order]>=20
        endpoints=np.r_[np.where(np.diff(sp)!=0)[0],len(sp)-1]
        cumulative=np.cumsum(sy)[endpoints];recall=cumulative/len(pos);precision=cumulative/(endpoints+1)
        ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
        assert abs(ap-r.average_precision)<1e-12
        checked+=1
report['checks']['heldout_prediction_sets_recomputed']=checked
report['checks']['auroc_max_absolute_difference']=maxdiff
report['checks']['same_people_targets_excluded283']=True
report['checks']['six_second_manifest_and_person_disjoint_outer_folds']=True
report['input_preservation']={outcome:{name:sha(source/(outcome+'_common6')/name) for name in ['input.npz','manifest.csv','splits.json','predictions.csv']} for outcome in ['STS','TUG']}
report['limits']=['No new probability calibration, intervals, classifier, or external prediction evaluated.','Linked Mobilise-D records precede label-quality adjudication; positive under3s and dictionary-range contradictions remain.','Checks verify file linkage and calculations, not correctness of original stopwatch labels or exact temporal boundaries.']
report['passed']=True
save('verification.json',report)
print(json.dumps(report,indent=2))
