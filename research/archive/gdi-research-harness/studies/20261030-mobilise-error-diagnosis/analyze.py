"""Descriptive audits of existing data/predictions; no new prediction model fitting."""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
REF=HERE.parent/'20261030-mobilise-deep-sets-stability'
spec=importlib.util.spec_from_file_location('reference',REF/'train.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
ref=module.ref;BASE=ref.BASE;SOURCE=HERE.parents[2]/'outputs/sppb-tug-candidates-20261030'
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return {'n':len(x),'min':float(x.min()),'p10':float(np.quantile(x,.1)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'max':float(x.max())} if len(x) else {'n':0}

def person_mean(frame,column):return float(frame.groupby('participantid')[column].mean().mean())

def run():
    f=pd.read_csv(BASE/'features.csv');b=pd.read_csv(BASE/'selected-bouts.csv');b=b[b.draw.eq(0)]
    headers=pd.read_csv(SOURCE/'All_cohorts_dataset.csv',nrows=0).columns
    wanted=['participantid','visit.number','cohort','visit.date','balance_score','4mwt_speed_score','5crt_score','total_sppb_score','tugtime','4mwt_max_speed','4mwt1','4mwt2','4mwt1secs','4mwt2secs','4mwt1fail','4mwt2fail','5crtsafe','5crtresult','sppbinst','tuginst','tugaids']
    main=pd.read_csv(SOURCE/'All_cohorts_dataset.csv',usecols=[c for c in wanted if c in headers],low_memory=False).rename(columns={'visit.number':'visit_number'})
    assert not main.duplicated(['participantid','visit_number']).any()
    p=pd.read_csv(REF/'predictions.csv');p=p[p.budget.eq(600)&p.draw.eq(0)&p.band.eq('long')&p.k.eq(5)].copy()
    ratio1=abs(b.averagecadence*b.averagestrideduration/120-1)
    ratio2=abs(b.averagestridespeed*b.averagestrideduration/b.averagestridelength-1)
    redundancy={'unique_bouts':int(b.drop_duplicates(['participantid','visit_number','wb_id']).shape[0]),'cadence_times_stride_duration_relative_error':stats(ratio1),'speed_times_duration_div_length_relative_error':stats(ratio2),'cadence_identity_within_1percent':float((ratio1<=.01).mean()),'speed_identity_within_1percent':float((ratio2<=.01).mean()),'spearman':b[ref.FEATURES].corr(method='spearman').to_dict()}
    extra={'turn_count_rows':len(b),'turn_count_missing':int(b.turn_number_so.isna().sum()),'turn_count_positive':int(b.turn_number_so.gt(0).sum()),'turn_flag_not_retained_in_selected_table':True}
    distributions=[];cohorts=[];label_audit={};neighbors=[];foldchecks=[]
    # Match unordered five-bout inputs by sorting their speeds, not arbitrary sampling rank.
    bagkey=['participantid','visit_number','band'];sorted_bouts=b.sort_values(bagkey+['averagestridespeed','averagestridelength','averagestrideduration','averagecadence'])
    ids=sorted_bouts[bagkey].drop_duplicates().reset_index(drop=True);ids['bag_index']=np.arange(len(ids))
    array=sorted_bouts[ref.FEATURES].to_numpy().reshape(len(ids),5,4)
    assert sorted_bouts.groupby(bagkey).size().eq(5).all()
    data=f[f.draw.eq(0)&f.band.eq('long')&f.k.eq(5)].merge(ids,on=bagkey,validate='one_to_one')
    for target,label in [('sppb','total_sppb_score'),('tug','tugtime')]:
        d=data[data[target+'_ok']].copy();d['tail']=d[label].le(3) if target=='sppb' else d[label].gt(20)
        m=main[np.isfinite(main[label])].copy()
        if target=='sppb':m=m[m[label].between(0,12)&m[label].mod(1).eq(0)]
        else:m=m[m[label].gt(0)]
        mt=m[label].le(3) if target=='sppb' else m[label].gt(20)
        a={'selected_people':d.participantid.nunique(),'selected_visits':len(d),'tail_people':d[d['tail']].participantid.nunique(),'tail_visits':int(d['tail'].sum()),'tail_person_weight_mass':person_mean(d,'tail'),'all_valid_label_people':m.participantid.nunique(),'all_valid_label_visits':len(m),'all_valid_tail_people':m[mt].participantid.nunique(),'all_valid_tail_visits':int(mt.sum()),'label_stats':stats(d[label])}
        a['selected_bout_total_seconds']=stats(d.total_seconds);a['selected_bout_span_hours']=stats(d.span_seconds/3600)
        for group,g in d.groupby('tail'):
            for name in ref.FEATURES:distributions.append({'target':target,'tail':bool(group),'feature':name,**stats(g[name])})
            for cohort,c in g.groupby('cohort'):cohorts.append({'target':target,'tail':bool(group),'cohort':cohort,'people':c.participantid.nunique(),'visits':len(c)})
        pred=p[p.target.eq(target)].copy();pred['tail']=pred.actual.le(3) if target=='sppb' else pred.actual.gt(20)
        pred['error']=pred.prediction-pred.actual;pred['abs_error']=abs(pred.error)
        a['error_groups']={}
        for tail,g in pred.groupby('tail'):
            a['error_groups'][str(bool(tail))]={'mae':float(g.groupby('seed').apply(lambda v:person_mean(v,'abs_error'),include_groups=False).mean()),'bias':float(g.groupby('seed').apply(lambda v:person_mean(v,'error'),include_groups=False).mean()),'predictions':stats(g.prediction),'actual':stats(g.actual),'underestimate_fraction':float((g.prediction<g.actual).mean())}
        if target=='sppb':
            comp=d[['participantid','visit_number',label,'tail']].merge(main.drop(columns=[label,'cohort']),on=['participantid','visit_number'],validate='one_to_one')
            cols=['balance_score','4mwt_speed_score','5crt_score'];complete=comp[cols].notna().all(axis=1)
            a['component_complete_visits']=int(complete.sum());a['component_sum_mismatches']=int((comp.loc[complete,cols].sum(axis=1)!=comp.loc[complete,label]).sum())
            a['component_groups']={}
            for tail,g in comp.groupby('tail'):
                a['component_groups'][str(bool(tail))]={'visits':len(g),'all_components_missing':int(g[cols].isna().all(axis=1).sum()),'components':{c:{'missing':int(g[c].isna().sum()),'counts':{str(k):int(v) for k,v in g[c].value_counts(dropna=False).items()}} for c in cols},'clinic_walk_speed':stats(g['4mwt_max_speed'])}
            paired=comp[comp['tail']&comp['4mwt_max_speed'].notna()].merge(d[['participantid','visit_number','averagestridespeed']],on=['participantid','visit_number'],validate='one_to_one')
            a['low_group_daily_minus_clinic_speed']=stats(paired.averagestridespeed-paired['4mwt_max_speed'])
        else:a['outside_ranges']={'below3_visits':int(d[label].lt(3).sum()),'above60_visits':int(d[label].gt(60).sum())}
        label_audit[target]=a
        for fold in sorted(d.fold.unique()):
            tr=d[d.fold.ne(fold)];te=d[d.fold.eq(fold)]
            assert not set(tr.participantid)&set(te.participantid)
            bundle=dict(np.load(REF/'models'/f'{target}-long-5-f{fold}-s20261030-b600.npz'))
            tx=((array[tr.bag_index]-bundle['x_mean'])/bundle['x_std']).reshape(len(tr),20)
            xx=((array[te.bag_index]-bundle['x_mean'])/bundle['x_std']).reshape(len(te),20)
            nn=NearestNeighbors(n_neighbors=1,algorithm='brute').fit(tx);dist,index=nn.kneighbors(xx)
            matched=tr.iloc[index[:,0]]
            assert not (te.participantid.to_numpy()==matched.participantid.to_numpy()).any()
            for mode in [False,True]:
                mask=te['tail'].to_numpy()==mode
                neighbors.append({'target':target,'tail':mode,'fold':int(fold),'visits':int(mask.sum()),'distance':stats(dist[mask,0]),'absolute_label_gap':stats(abs(te[label].to_numpy()[mask]-matched[label].to_numpy()[mask])),'neighbor_not_tail_fraction':float((~matched['tail'].to_numpy()[mask]).mean())})
            # A few independent brute-force checks verify the nearest-neighbor result.
            for j in [0,len(te)//2,len(te)-1]:
                manual=np.sqrt(((tx-xx[j])**2).sum(axis=1));np.testing.assert_allclose(dist[j,0],manual.min(),rtol=1e-9,atol=1e-9)
            foldchecks.append({'target':target,'fold':int(fold),'train_people':int(tr.participantid.nunique()),'test_people':int(te.participantid.nunique()),'disjoint':True})
    pd.DataFrame(distributions).to_csv(HERE/'feature-distributions.csv',index=False)
    pd.DataFrame(cohorts).to_csv(HERE/'cohort-counts.csv',index=False)
    result={'scope':'descriptive existing data/error analysis; no new predictive model fitting, no treatment or clinical inference','redundancy':redundancy,'additional_input_audit':extra,'label_audit':label_audit,'nearest_neighbor_diagnostic':neighbors,'neighbor_limit':'one nearest training visit in standardized speed-sorted five-bout input; diagnostic only, not an irreducible-error bound, not an evaluated predictive model, frequent-visit participants may be nearest more often','fold_checks':foldchecks,'source_sha256':{str(path):ref.sha(path) for path in [SOURCE/'All_cohorts_dataset.csv',BASE/'features.csv',BASE/'selected-bouts.csv',REF/'predictions.csv']}}
    (HERE/'diagnosis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf8')
    print(json.dumps({'redundancy':{k:v for k,v in redundancy.items() if k!='spearman'},'label_audit':label_audit},indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
