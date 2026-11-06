from common import *

MODELS=['baseline','Ridge','SVR','RF']
def summarize():
    p=pd.read_csv(HERE/'predictions.csv')
    rows=[];subgroups=[];contrasts=[]
    rng=np.random.default_rng(20261030)
    for (target,band,k,draw),d in p.groupby(['target','band','k','draw']):
        errors=d[['participantid']].copy()
        for model in MODELS:errors[model]=abs(d.actual-d[model])
        perperson=errors.groupby('participantid')[MODELS].mean()
        boot=rng.integers(0,len(perperson),size=(2000,len(perperson))) if draw==0 else None
        for model in MODELS:
            mae=float(perperson[model].mean());base=float(perperson.baseline.mean())
            row={'target':target,'band':band,'k':int(k),'draw':int(draw),'model':model,'people':len(perperson),'visits':len(d),'person_mae':mae,'baseline_person_mae':base,'improvement_absolute':base-mae,'improvement_percent':100*(base-mae)/base,'person_rmse':float(np.sqrt(pd.DataFrame({'id':d.participantid,'se':(d.actual-d[model])**2}).groupby('id').se.mean().mean())),'row_mae':float(abs(d.actual-d[model]).mean()),'row_error_p90':float(abs(d.actual-d[model]).quantile(.9))}
            if draw==0:
                row['mae_ci_low'],row['mae_ci_high']=np.quantile(perperson[model].to_numpy()[boot].mean(axis=1),[.025,.975]).tolist()
                row['improvement_ci_low'],row['improvement_ci_high']=np.quantile((perperson.baseline-perperson[model]).to_numpy()[boot].mean(axis=1),[.025,.975]).tolist()
            rows.append(row)
        if draw==0:
            for cohort,g in d.groupby('cohort'):
                for model in MODELS:
                    subgroups.append({'target':target,'band':band,'k':int(k),'group_type':'cohort','group':cohort,'model':model,'people':int(g.participantid.nunique()),'visits':len(g),'person_mae':person_mae(g.actual.to_numpy(),g[model].to_numpy(),g.participantid.to_numpy())})
            cuts=[-np.inf,3,6,9,np.inf] if target=='sppb' else [-np.inf,10,20,np.inf]
            d=d.copy();d['score_band']=pd.cut(d.actual,cuts)
            for score,g in d.groupby('score_band',observed=True):
                for model in MODELS:
                    subgroups.append({'target':target,'band':band,'k':int(k),'group_type':'descriptive_target_band','group':str(score),'model':model,'people':int(g.participantid.nunique()),'visits':len(g),'person_mae':person_mae(g.actual.to_numpy(),g[model].to_numpy(),g.participantid.to_numpy())})
    for target in ['sppb','tug']:
        d=p[p.target.eq(target)&p.draw.eq(0)].copy()
        for model in MODELS[1:]:
            d['error']=abs(d.actual-d[model])
            e=d.groupby(['participantid','band','k']).error.mean().unstack(['band','k'])
            boot=rng.integers(0,len(e),size=(2000,len(e)))
            comparisons=[(('short',k),('long',k)) for k in [1,3,5]]+[((b,1),(b,5)) for b in ['short','long']]
            for a,b in comparisons:
                diff=e[a]-e[b]
                ci=np.quantile(diff.to_numpy()[boot].mean(axis=1),[.025,.975])
                contrasts.append({'target':target,'model':model,'from':f'{a[0]}-{a[1]}','to':f'{b[0]}-{b[1]}','person_mae_reduction':float(diff.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])})
    summary=pd.DataFrame(rows);summary.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(subgroups).to_csv(HERE/'subgroup-metrics.csv',index=False)
    save('paired-comparisons.json',contrasts)
    best=[]
    for target in ['sppb','tug']:
        candidates=summary[summary.target.eq(target)&summary.draw.eq(0)&summary.model.ne('baseline')]
        row=candidates.loc[candidates.person_mae.idxmin()].to_dict();best.append(row)
    save('summary.json',{'best_primary_conditions':best,'primary_draw':0,'interval':'2000 paired person bootstrap, conditional on fitted OOF models; does not refit or adjust for model selection','metrics_rows':len(summary),'primary_metrics_rows':int(summary.draw.eq(0).sum()),'draws_1_2':'held-out input resampling only, same trained models, separate predictions and errors','not_external_validation':True})
    f=pd.read_csv(HERE/'features.csv');sampling=[];target_stats=[];cohorts=[]
    for target in ['sppb','tug']:
        d=f[f[target+'_ok']&f.draw.eq(0)]
        for (band,k),g in d.groupby(['band','k']):
            sampling.append({'target':target,'band':band,'k':int(k),'median_total_seconds':float(g.total_seconds.median()),'median_span_seconds':float(g.span_seconds.median())})
        z=d[d.band.eq('short')&d.k.eq(1)];col='total_sppb_score' if target=='sppb' else 'tugtime'
        target_stats.append({'target':target,'min':float(z[col].min()),'max':float(z[col].max()),'row_median':float(z[col].median()),'people':int(z.participantid.nunique()),'visits':len(z)})
        for cohort,g in z.groupby('cohort'):
            cohorts.append({'target':target,'cohort':cohort,'people':int(g.participantid.nunique()),'visits':len(g)})
    tug=f[f.tug_ok&f.draw.eq(0)&f.band.eq('short')&f.k.eq(1)]
    save('sampling-summary.json',{'sampling':sampling,'targets':target_stats,'cohorts':cohorts,'tug_label_audit':{'below_3s_visits':int(tug.tugtime.lt(3).sum()),'above_60s_visits':int(tug.tugtime.gt(60).sum()),'positive_values_retained_without_posthoc_trimming':True}})
    x=summary[summary.draw.eq(0)].pivot(index=['target','band','k'],columns='model',values='person_mae')
    print(x.round(4).to_string(),flush=True)
    print('Best primary conditions:',json.dumps(best,ensure_ascii=False),flush=True)
if __name__=='__main__':summarize()
