"""Analyze initialization stability without choosing seeds or ensembling predictions."""
from train import HERE,REFERENCE,BASE,MLP,SEEDS,BUDGETS,save,ref
import numpy as np
import pandas as pd
import json
KEY=['participantid','visit_number','draw','fold','target','band','k']
def groups(d,target):
    yield 'all',d
    if target=='sppb':
        for name,mask in [('low',d.actual.le(3)),('middle',d.actual.gt(3)&d.actual.le(9)),('high',d.actual.gt(9))]:yield name,d[mask]
    else:
        yield 'long',d[d.actual.gt(20)];yield 'other',d[d.actual.le(20)]
    for cohort,g in d.groupby('cohort'):yield 'cohort_'+cohort,g
def metric(d,column):
    error=(d[column]-d.actual).to_numpy()
    ids=d.participantid.to_numpy();w=ref.weights(ids)
    order=np.argsort(abs(error));cdf=np.cumsum(w[order]);q=abs(error[order][min(np.searchsorted(cdf,.9),len(error)-1)])
    return {'people':int(d.participantid.nunique()),'visits':len(d),'mae':float(np.dot(w,abs(error))),'bias':float(np.dot(w,error)),'rmse':float(np.sqrt(np.dot(w,error**2))),'error_p90':float(q)}
def run():
    p=pd.read_csv(HERE/'predictions.csv')
    original=pd.read_csv(REFERENCE/'paired-predictions.csv')
    assert len(p)==len(original)*10
    p=p.merge(original[KEY+['actual','baseline','SVR','MLP','DeepSets','cohort']],on=KEY,validate='many_to_one',suffixes=('','_reference'))
    np.testing.assert_array_equal(p.actual,p.actual_reference)
    rows=[];references=[]
    for (target,band,k,draw),d in original.groupby(['target','band','k','draw']):
        for group,g in groups(d,target):
            for model in ['baseline','SVR','MLP','DeepSets']:
                references.append({'target':target,'band':band,'k':int(k),'draw':int(draw),'group':group,'model':model,**metric(g,model)})
    reference=pd.DataFrame(references);reference.to_csv(HERE/'reference-metrics.csv',index=False)
    for (target,band,k,draw,seed,budget),d in p.groupby(['target','band','k','draw','seed','budget']):
        for group,g in groups(d,target):
            rows.append({'target':target,'band':band,'k':int(k),'draw':int(draw),'seed':int(seed),'budget':int(budget),'group':group,**metric(g,'prediction')})
    metrics=pd.DataFrame(rows);metrics.to_csv(HERE/'seed-metrics.csv',index=False)
    aggregate=[];paired=[]
    grouping=['target','band','k','draw','group']
    for key,d in metrics.groupby(grouping):
        info=dict(zip(grouping,key));info={a:int(b) if isinstance(b,np.integer) else b for a,b in info.items()}
        rr=reference
        for name,value in info.items():rr=rr[rr[name].eq(value)]
        assert len(rr)==4
        for budget,g in d.groupby('budget'):
            assert set(g.seed)==set(SEEDS)
            row={**info,'budget':int(budget),'people':int(g.people.iloc[0]),'visits':int(g.visits.iloc[0])}
            for name in ['mae','bias','rmse','error_p90']:
                row[name+'_mean']=float(g[name].mean());row[name+'_sd']=float(g[name].std(ddof=1));row[name+'_min']=float(g[name].min());row[name+'_max']=float(g[name].max())
            for model in ['baseline','SVR','MLP','DeepSets']:
                baseline=float(rr[rr.model.eq(model)].mae.iloc[0]);row[model+'_mae']=baseline
                row[model+'_improvement']=baseline-row['mae_mean'];row[model+'_improvement_percent']=100*(baseline-row['mae_mean'])/baseline
                row['seeds_better_than_'+model]=int(g.mae.lt(baseline).sum())
            aggregate.append(row)
        a=d[d.budget.eq(120)].set_index('seed');b=d[d.budget.eq(600)].set_index('seed')
        for seed in SEEDS:
            paired.append({**info,'seed':seed,'mae_reduction':float(a.loc[seed,'mae']-b.loc[seed,'mae']),'absolute_bias_reduction':float(abs(a.loc[seed,'bias'])-abs(b.loc[seed,'bias'])),'rmse_reduction':float(a.loc[seed,'rmse']-b.loc[seed,'rmse']),'error_p90_reduction':float(a.loc[seed,'error_p90']-b.loc[seed,'error_p90'])})
    agg=pd.DataFrame(aggregate);agg.to_csv(HERE/'aggregate-metrics.csv',index=False)
    paired=pd.DataFrame(paired);paired.to_csv(HERE/'paired-budgets.csv',index=False)
    comparisons=[];rng=np.random.default_rng(20261030)
    for (target,band,k),d in p[p.draw.eq(0)].groupby(['target','band','k']):
        for group,g in groups(d,target):
            if group not in ['all','low','long']:continue
            g=g.copy();g['error']=abs(g.prediction-g.actual)
            person=g.groupby(['participantid','seed','budget']).error.mean().unstack(['seed','budget'])
            # Average errors across seeds, not predictions across models.
            short=person.loc[:,[(seed,120) for seed in SEEDS]].mean(axis=1)
            long=person.loc[:,[(seed,600) for seed in SEEDS]].mean(axis=1)
            unique=g[g.seed.eq(SEEDS[0])&g.budget.eq(120)]
            values={'paired_120':short-long}
            for model in ['SVR','MLP','DeepSets']:
                e=pd.DataFrame({'id':unique.participantid,'e':abs(unique[model]-unique.actual)}).groupby('id').e.mean()
                assert e.index.equals(person.index)
                values[model]=e-long
            boot=rng.integers(0,len(person),size=(2000,len(person)))
            for model,delta in values.items():
                lo,hi=np.quantile(delta.to_numpy()[boot].mean(axis=1),[.025,.975])
                comparisons.append({'target':target,'band':band,'k':int(k),'group':group,'reference':model,'people':len(person),'mae_reduction':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    save('paired-bootstrap.json',comparisons)
    selections=[]
    for path in sorted((HERE/'folds').glob('*.json')):
        x=json.loads(path.read_text())
        for budget in BUDGETS:
            selections.append({name:x[name] for name in ['target','band','k','fold','seed']}|{'budget':budget,**x['selected'][str(budget)]})
    selection=pd.DataFrame(selections);selection.to_csv(HERE/'selections.csv',index=False)
    summary={'seeds':SEEDS,'budgets':BUDGETS,'outer_models':600,'main':'per-seed results and their mean/range, no best-seed selection, no prediction ensemble','evaluation':'same person folds and bouts; exploratory reuse of development OOF predictions','subgroups':'actual-label-selected visits, equal weight per represented person; low SPPB<=3, long TUG>20s; descriptive thresholds, not diagnostic grades','bootstrap':'2000 paired person draws on five-seed mean errors; conditional on these folds/models, excludes external-dataset and full retraining uncertainty','epoch_boundary_counts':{str(b):int((selection.budget.eq(b)&selection.epochs.eq(b)).sum()) for b in BUDGETS},'long5_primary':agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','low','long'])].to_dict(orient='records'),'artifact_sha256':{name:ref.sha(HERE/name) for name in ['seed-metrics.csv','aggregate-metrics.csv','paired-budgets.csv','paired-bootstrap.json','selections.csv','reference-metrics.csv']}}
    save('summary.json',summary)
    print(agg[agg.draw.eq(0)&agg.budget.eq(600)&agg.group.eq('all')][['target','band','k','mae_mean','mae_min','mae_max','SVR_improvement_percent','seeds_better_than_SVR','seeds_better_than_MLP']].round(5).to_string(index=False),flush=True)
    print('LONG5 SUBGROUPS',agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','low','long'])][['target','group','budget','people','visits','mae_mean','mae_min','mae_max','bias_mean','rmse_mean','error_p90_mean']].round(5).to_string(index=False),flush=True)
    print('EPOCH_BOUNDARIES',summary['epoch_boundary_counts'],flush=True)
if __name__=='__main__':run()
