"""Availability and label audit, never model training or clinical threshold setting."""
from common import *

def stats(df):
    out={'rows':len(df),'people':int(df.participantid.nunique())}
    for col in ['tugtime','5crtresult','total_sppb_score']:
        v=pd.to_numeric(df[col],errors='coerce');valid=v.notna()&(v>0 if col!='total_sppb_score' else v.between(0,12))
        out[col]={'rows':int(valid.sum()),'people':int(df.loc[valid,'participantid'].nunique()),'min':float(v[valid].min()) if valid.any() else None,'max':float(v[valid].max()) if valid.any() else None}
        if col!='total_sppb_score':out[col]['ge20_people']=int(df.loc[valid&(v>=20),'participantid'].nunique())
        else:out[col]['le6_people']=int(df.loc[valid&(v<=6),'participantid'].nunique())
    return out

def main():
    p=HERE/'extracted/Main datasets for analysis/CSV files/All_cohorts_dataset.csv'
    full=pd.read_csv(p,low_memory=False);keys=['participantid','visit.number'];assert not full.duplicated(keys).any()
    use=['participantid','participant.site','cohort','visit.number','visit.date','proto_dev','proto_dev_err','tugtime','tugaids','tuginst','5crtsafe','5crtresult','1crtresult','5crt_score','balance_score','4mwt_speed_score','total_sppb_score','ws_1030_avg_w','ws_30_avg_w','n_days_w']
    a=full[use].copy()
    # Confirm main-table labels against separate original clinical forms, keyed by person and visit.
    orig=HERE/'extracted/Clinical Data';checks={}
    for filename,columns in [('tug',['tugtime']),('sppb',['5crtresult'])]:
        b=pd.read_csv(orig/f'T1-T5-study-instances-{filename}.csv').rename(columns={'visit-number':'visit.number'})
        assert not b.duplicated(keys).any()
        j=a.merge(b[keys+columns],on=keys,suffixes=('_main','_form'),validate='one_to_one')
        for c in columns:np.testing.assert_allclose(j[c+'_main'],j[c+'_form'],equal_nan=True)
        checks[filename]={'rows_matched':len(j),'rows_in_form':len(b)}
    # Provider-filtered weekly tables; retain visit, no using later visits as extra people.
    weeks=pd.concat([pd.read_csv(x) for x in sorted((HERE/'extracted/Aggregated').glob('*weekly_agg_filtered.csv'))],ignore_index=True).rename(columns={'visit_number':'visit.number'})
    assert not weeks.duplicated(keys).any();j=a.merge(weeks[keys+['ws_1030_avg_w','n_days_w']],on=keys,validate='one_to_one',suffixes=('_main','_week'))
    np.testing.assert_allclose(j.ws_1030_avg_w_main,j.ws_1030_avg_w_week,equal_nan=True);np.testing.assert_allclose(j.n_days_w_main,j.n_days_w_week,equal_nan=True)
    a['dmo_available']=(a.ws_1030_avg_w>0)&(a.n_days_w>=3)
    assert set(a.proto_dev.dropna().unique()) <= {'N','Y'}
    a['timing_ok']=a.proto_dev.eq('N')
    a['dmo_timing_ok']=a.dmo_available&a.timing_ok
    a['tug_positive']=a.tugtime>0;a['sts_positive']=a['5crtresult']>0
    a['sppb_available']=a.total_sppb_score.between(0,12)
    a.to_csv(HERE/'availability-by-person-visit.csv',index=False)
    report={'all_main':stats(a),'main_sha256':sha(p),'clinical_form_checks':checks,'weekly_rows_checked':len(j),
            'proto_dev_values':{str(k):int(v) for k,v in a.proto_dev.value_counts(dropna=False).items()},
            'by_visit':{},'first_available':{},'ranges':{},
            'criteria':'DMO: positive ws_1030_avg_w and provider filtered weekly n_days_w>=3; timing sensitivity additionally requires proto_dev false (+/-14days clinical-DMO). First available selected separately per target, sorted visit, one record per person. TUG field tugtime only; never pool PD minibest tugt1n/dual-task time.',
            'license':'CC BY-NC-ND4.0 per official release; research review only, no product-use permission inferred'}
    for visit,g in a.groupby('visit.number'):
        report['by_visit'][visit]={'all':stats(g),'with_dmo':stats(g[g.dmo_available]),'with_dmo_timing_ok':stats(g[g.dmo_timing_ok])}
    selected=[]
    for target,flag in [('TUG','tug_positive'),('STS','sts_positive'),('SPPB','sppb_available')]:
        for timing in [False,True]:
            g=a[a[flag]&a['dmo_timing_ok' if timing else 'dmo_available']].sort_values(['participantid','visit.number']).drop_duplicates('participantid').copy()
            report['first_available'][target+('_timing_ok' if timing else '')]=stats(g)
            if timing:
                g['audit_target']=target;selected.append(g)
                report['first_available'][target+'_timing_ok']['cohorts']={str(k):int(v) for k,v in g.cohort.value_counts().items()}
                report['first_available'][target+'_timing_ok']['visits']={str(k):int(v) for k,v in g['visit.number'].value_counts().items()}
                report['first_available'][target+'_timing_ok']['by_cohort']={str(k):stats(v) for k,v in g.groupby('cohort')}
                report['first_available'][target+'_timing_ok']['sites']=int(g['participant.site'].nunique())
    pd.concat(selected,ignore_index=True).to_csv(HERE/'first-eligible-per-target.csv',index=False)
    for col in ['tugtime','5crtresult']:
        s=a[col];report['ranges'][col]={'zero_rows':int((s==0).sum()),'missing_rows':int(s.isna().sum()),'positive_under1_rows':int(((s>0)&(s<1)).sum()),'positive_under3_rows':int(((s>0)&(s<3)).sum()),'above_dictionary_max_rows':int((s>(60 if col=='tugtime' else 120)).sum()),'equal60_rows':int((s==60).sum()),'above60_rows':int((s>60).sum())}
    scorecols=['balance_score','4mwt_speed_score','5crt_score'];mask=a[scorecols+['total_sppb_score']].notna().all(axis=1)
    report['sppb_component_sum_checked_rows']=int(mask.sum());assert np.all(a.loc[mask,scorecols].sum(axis=1)==a.loc[mask,'total_sppb_score'])
    report['sts_zero_with_chair_score0']=int(((a['5crtresult']==0)&(a['5crt_score']==0)).sum())
    report['all_with_sppb0_but_no_positive_sts']=int(((a['5crt_score']==0)&~a.sts_positive).sum())
    save('mobilised-audit.json',report)
    print(json.dumps({'main':report['all_main'],'t1':report['by_visit']['t1'],'first_available':report['first_available'],'ranges':report['ranges']},indent=2))
if __name__=='__main__':main()
