from common import *
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss

def metrics(g):
    y=g.slow.to_numpy();p=g.probability.to_numpy();d=g.decision.to_numpy(bool)
    tp=int(((y==1)&d).sum());fn=int(((y==1)&~d).sum());fp=int(((y==0)&d).sum());tn=int(((y==0)&~d).sum())
    r={'people':len(g),'slow':int(y.sum()),'tp':tp,'fn':fn,'fp':fp,'tn':tn,
       'miss_rate':fn/(tp+fn) if tp+fn else np.nan,'sensitivity':tp/(tp+fn) if tp+fn else np.nan,
       'fpr':fp/(fp+tn) if fp+tn else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,
       'auroc':roc_auc_score(y,p) if len(set(y))==2 else np.nan,'ap':average_precision_score(y,p) if y.sum() else np.nan,
       'brier':brier_score_loss(y,p),'baseline_brier':brier_score_loss(y,g.train_prevalence),
       'logloss':log_loss(y,p,labels=[0,1]),'p50_miss_rate':float((~g.loc[g.slow==1,'decision_p50']).mean()),
       'mean_probability':float(p.mean()),'actual_slow_rate':float(y.mean()),'calibration_bias':float(p.mean()-y.mean())}
    bins=np.minimum((p*10).astype(int),9);ece=0
    for b in np.unique(bins):
        mask=bins==b;ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    r['ece10']=ece
    if g.prediction_seconds.notna().all():
        e=g.prediction_seconds.to_numpy()-g.target.to_numpy();slow=y==1
        r.update(mae=float(abs(e).mean()),median_baseline_mae=float(abs(g.train_median-g.target).mean()),
                 slow_mae=float(abs(e[slow]).mean()) if slow.any() else np.nan,
                 slow_under=float(np.maximum(-e[slow],0).mean()) if slow.any() else np.nan,
                 slow_over=float(np.maximum(e[slow],0).mean()) if slow.any() else np.nan,
                 time20_miss_rate=float((~g.loc[g.slow==1,'decision_time20']).mean()),negative_prediction_count=int((g.prediction_seconds<0).sum()))
    return r

def paired_ci(values,seed=1021):
    values=np.asarray(values,float);rng=np.random.default_rng(seed);means=[]
    for _ in range(20):means.extend(values[rng.integers(0,len(values),size=(100,len(values)))].mean(axis=1))
    return {'difference':float(values.mean()),'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975)),'people':len(values)}

def main():
    p=pd.read_csv(OUT/'predictions.csv.gz');rows=[];bins=[]
    labels=pd.read_csv(OUT/'label-audit-ledger.csv')
    first=pd.read_csv(SRC/'first-eligible-per-target.csv')
    first=first[first.audit_target=='SPPB'][['participantid','visit.number']].merge(labels,on=['participantid','visit.number'],how='left',validate='one_to_one')
    save('label-scope.json',{'all_record_status_people':labels.groupby('sts_status').participantid.nunique().to_dict(),'first_SPPB_linked_visit_people':len(first),'first_SPPB_visit_statuses':first.sts_status.fillna('no_chair_time_or_score').value_counts().to_dict(),'note':'SPPB-first-visit population illustrates informative noncompletion. It is not the same selected visit as first positive STS; counts must not be subtracted from1970.'})
    for key,g in p.groupby(['outcome','model','view','repeat']):
        base=dict(zip(['outcome','model','view','repeat'],key))
        for cohort in ['all']+sorted(g.cohort.unique().tolist())+(['time_le60'] if key[0]=='TUG' else []):
            h=g if cohort=='all' else g[g.target<=60] if cohort=='time_le60' else g[g.cohort==cohort]
            rows.append(base|{'cohort':cohort}|metrics(h))
        g=g.copy();g['bin']=np.minimum((g.probability*10).astype(int),9)
        for b,h in g.groupby('bin'):bins.append(base|{'bin':int(b),'people':len(h),'probability_mean':float(h.probability.mean()),'actual_rate':float(h.slow.mean())})
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics-by-repeat.csv',index=False)
    summary=table.groupby(['outcome','model','view','cohort']).mean(numeric_only=True).drop(columns=['repeat']).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False);pd.DataFrame(bins).to_csv(OUT/'calibration-bins.csv',index=False)
    # Person-cluster bootstrap: average repeated held-out loss before resampling people.
    p['brier_loss']=(p.probability-p.slow)**2;p['miss_loss']=1-p.decision.astype(float)
    p['abs_error']=abs(p.prediction_seconds-p.target)
    comparisons=[]
    for outcome in ['STS','TUG']:
        q=p[p.outcome==outcome]
        configs=[(m,'speed',m,'pattern') for m in ['Ridge','Logistic','RFreg','RFclass']]
        configs += [(m,'missingness_control',m,'pattern') for m in ['Ridge','Logistic','RFreg','RFclass']]
        configs += [('RFreg',v,'RFclass',v) for v in ['speed','pattern']]+[('Ridge',v,'Logistic',v) for v in ['speed','pattern']]
        for ma,va,mb,vb in configs:
            for metric in ['brier_loss','miss_loss','abs_error']:
                a=q[(q.model==ma)&(q.view==va)];b=q[(q.model==mb)&(q.view==vb)]
                if metric=='miss_loss':a=a[a.slow==1];b=b[b.slow==1]
                a=a.groupby('participantid')[metric].mean();b=b.groupby('participantid')[metric].mean()
                delta=(b-a).dropna()
                if len(delta):comparisons.append({'outcome':outcome,'reference':ma+'/'+va,'candidate':mb+'/'+vb,'metric':metric,**paired_ci(delta)})
    pd.DataFrame(comparisons).to_csv(OUT/'paired-comparisons.csv',index=False)
    # Diagnosis-only benchmark: trained within identical base-training folds, not an input to gait models.
    parts=pd.read_csv(OUT/'person-folds.csv');prior=[]
    for outcome in ['STS','TUG']:
        data=pd.read_csv(OUT/f'{outcome}-cohort.csv')
        for rep in [0,1,2]:
            f=data.participantid.map(parts[parts['repeat']==rep].set_index('participantid').fold)
            for fold in range(5):
                train=data[(f!=fold)&(f!=(fold+1)%5)];test=data[f==fold].copy()
                test['probability']=test.cohort.map(train.groupby('cohort').slow.mean())
                test['seconds']=test.cohort.map(train.groupby('cohort').target.median())
                test['repeat']=rep;test['outcome']=outcome;prior.append(test[['participantid','repeat','outcome','slow','target','probability','seconds']])
    prior=pd.concat(prior,ignore_index=True);prior.to_csv(OUT/'diagnosis-only-predictions.csv',index=False)
    prior_rows=[]
    for (outcome,rep),g in prior.groupby(['outcome','repeat']):prior_rows.append({'outcome':outcome,'repeat':rep,'brier':brier_score_loss(g.slow,g.probability),'auroc':roc_auc_score(g.slow,g.probability),'mae':float(abs(g.target-g.seconds).mean())})
    pd.DataFrame(prior_rows).to_csv(OUT/'diagnosis-only-metrics.csv',index=False)
    print(summary[(summary.cohort=='all')&(summary.view!='missingness_control')][['outcome','model','view','mae','slow_under','auroc','brier','miss_rate','fpr','precision','time20_miss_rate']].round(4).to_string(index=False))
    print('Disease RFclass pattern');print(summary[(summary.model=='RFclass')&(summary.view=='pattern')][['outcome','cohort','people','slow','miss_rate','fpr','auroc']].round(4).to_string(index=False))
    save('analysis-completion.json',{'predictions_sha256':sha(OUT/'predictions.csv.gz'),'summary_sha256':sha(OUT/'summary.csv'),'paired_bootstrap_replicates':2000,'bootstrap_limit':'Conditional on fitted models and observed people; not full retraining uncertainty or external validation.','completed':True})

if __name__=='__main__':main()
