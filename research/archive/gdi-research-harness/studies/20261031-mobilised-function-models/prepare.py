from common import *
from sklearn.model_selection import StratifiedKFold

def sts_status(a):
    t=a['5crtresult'];s=pd.Series('missing_time_unresolved',index=a.index)
    s.loc[t.isna() & (a['5crt_score']==0)]='score0_without_time_reason_unresolved'
    blocked=a['1crtsafe'].eq(False)|a['5crtsafe'].eq(False)|a['1crtresult'].isin(['Participant used arms to stand','Test not completed'])
    s.loc[t.isna()&blocked]='not_standard_completion_or_not_safe'
    s.loc[t==0]='zero_invalid_time'
    s.loc[(t>0)&(t<3)]='under3s_unresolved'
    s.loc[t>120]='above_dictionary120_unresolved'
    plausible=t.between(3,120)
    s.loc[plausible]='completion_metadata_unresolved'
    expected=np.select([t>60,t>=16.7,t>=13.7,t>=11.2],[0,1,2,3],default=4)
    standard=a['1crtsafe'].eq(True)&a['5crtsafe'].eq(True)&a['1crtresult'].eq('Participant stood without using arms')
    s.loc[plausible&standard&(a['5crt_score']==expected)]='valid_standard_time'
    s.loc[(t>0)&blocked]='positive_time_conflicts_with_completion'
    return s

def main():
    path=SRC/'extracted/Main datasets for analysis/CSV files/All_cohorts_dataset.csv'
    a=pd.read_csv(path,low_memory=False)
    assert not a.duplicated(['participantid','visit.number']).any()
    a['sts_status']=sts_status(a)
    t=a.tugtime
    a['tug_status']=np.select([t.isna(),t<=0,(t>0)&(t<3),t>60],['missing','nonpositive','under3s_unresolved','positive_over60_dictionary_conflict_retained'],default='positive_3to60')
    elig=(a.ws_1030_avg_w>0)&(a.n_days_w>=3)&a.proto_dev.eq('N')
    # Original first eligible record fixed before further label cleaning: no backfill with a preferred later label.
    first=pd.read_csv(SRC/'first-eligible-per-target.csv')
    counts={};allsets=[]
    for target,label in [('STS','5crtresult'),('TUG','tugtime')]:
        select=first[first.audit_target==target][['participantid','visit.number']]
        g=select.merge(a,on=['participantid','visit.number'],validate='one_to_one')
        valid=g.sts_status.eq('valid_standard_time') if target=='STS' else g.tug_status.isin(['positive_3to60','positive_over60_dictionary_conflict_retained'])
        counts[target]={'before':len(g),'excluded':int((~valid).sum()),'reasons':g.loc[~valid,'sts_status' if target=='STS' else 'tug_status'].value_counts().to_dict()}
        g.loc[~valid,['participantid','visit.number','cohort',label,'sts_status','tug_status']].to_csv(OUT/f'{target}-excluded.csv',index=False)
        g=g[valid].copy();g['target']=g[label];g['slow']=g.target.ge(20).astype(int);g['outcome']=target
        cols=['participantid','visit.number','cohort','participant.site','outcome','target','slow','sts_status','tug_status']+SPEED+PATTERN
        g[cols].to_csv(OUT/f'{target}-cohort.csv',index=False);allsets.append(g[cols])
        counts[target].update(people=len(g),slow=int(g.slow.sum()),cohorts=g.groupby('cohort').agg(people=('participantid','size'),slow=('slow','sum')).to_dict('index'),over60=int((g.target>60).sum()))
    allsets=pd.concat(allsets,ignore_index=True)
    people=allsets[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
    assert not people.participantid.duplicated().any()
    folds=[]
    for rep,seed in enumerate([17,41,89]):
        split=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for fold,(_,idx) in enumerate(split.split(people,people.cohort)):
            for i in idx:folds.append({'participantid':int(people.iloc[i].participantid),'repeat':rep,'fold':fold,'seed':seed})
    pd.DataFrame(folds).to_csv(OUT/'person-folds.csv',index=False)
    auditcols=['participantid','visit.number','cohort','1crtsafe','1crtresult','5crtsafe','5crtresult','5crt_score','tugtime','sts_status','tug_status']
    a[a['5crtresult'].notna()|a['5crt_score'].notna()][auditcols].to_csv(OUT/'label-audit-ledger.csv',index=False)
    save('label-audit.json',{'all_rows':len(a),'all_sts_statuses':a.sts_status.value_counts().to_dict(),'with_dmo_sts_statuses':a.loc[elig,'sts_status'].value_counts().to_dict(),'selected':counts,'source_sha256':sha(path),'source_first_eligible_sha256':sha(SRC/'first-eligible-per-target.csv'),'note':'No changes to original values. Under3s records unresolved, not claimed physiologically proven impossible. STS requires both safety flags, unassisted single stand, concordant score; cannot infer numerical time for incomplete tests. TUG >60 kept and sensitivity evaluated separately. No replacement of excluded first eligible visits.'})
    save('protocol.json',{'threshold_seconds':20,'threshold_purpose':'inherited descriptive slow-group definition, not clinical diagnosis','speed_features':SPEED,'pattern_added_features':PATTERN,'views':['speed','pattern','missingness_control'],'models':['Ridge','Logistic','RFreg','RFclass'],'fixed_settings':{'Ridge_alpha':10,'Logistic_C':1,'RF_trees':200,'RF_min_samples_leaf':10,'RF_max_features':1.0,'RF_n_jobs':2},'partitions':'3 repeats x5 folds shared master people across both targets, stratify cohort only; test fold k, calibration(k+1)%5, training other3. No repeat visits or target leakage. Same train/cal/test people for all model/view cells.','calibration':'Sigmoid logistic on training-held-out calibration score (regression time or classifier logit); C=1e6, no weighting. Same calibration cohort determines >= threshold just above 90th percentile of negative calibrated probabilities.','evaluation':'Held-out only; target calibration specificity90%, observed specificity/sensitivity/PPV, Brier/logloss/AUROC/AP, calibration bins; regression MAE and tail under/over. Threshold20 and probability.5 also report.','time_scope':'provider-filtered weekly DMO, not one10-30s bout nor6s video','label_selection':counts,'prior_results_modified':False})
    print(json.dumps(counts,indent=2))
if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);main()
