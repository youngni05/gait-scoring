"""Paired evaluation with the original, unboosted person weights."""
from train import HERE,REFERENCE,SEEDS,FACTORS,KEY,ref,save
import numpy as np
import pandas as pd

def groups(d,target):
    yield 'all',d
    tail=d.actual.le(3) if target=='sppb' else d.actual.gt(20)
    yield 'tail',d[tail];yield 'non_tail',d[~tail]
    if target=='sppb':
        yield 'middle',d[d.actual.gt(3)&d.actual.le(9)]
        yield 'high',d[d.actual.gt(9)]
    for cohort,g in d.groupby('cohort'):yield 'cohort_'+cohort,g

def metric(d):
    e=(d.prediction-d.actual).to_numpy();w=ref.weights(d.participantid.to_numpy())
    order=np.argsort(abs(e));q=abs(e[order][min(np.searchsorted(np.cumsum(w[order]),.9),len(e)-1)])
    return {'people':int(d.participantid.nunique()),'visits':len(d),'mae':float(np.dot(w,abs(e))),'bias':float(np.dot(w,e)),'rmse':float(np.sqrt(np.dot(w,e**2))),'error_p90':float(q)}

def load_predictions():
    old=pd.read_csv(REFERENCE/'predictions.csv');old=old[old.budget.eq(600)].drop(columns='budget');old['factor']=1
    new=pd.read_csv(HERE/'predictions.csv')
    assert len(new)==len(old)*2
    metadata=pd.read_csv(ref.HERE/'paired-predictions.csv')[KEY[:-1]+['cohort']]
    return pd.concat([old,new],ignore_index=True).merge(metadata,on=KEY[:-1],validate='many_to_one')

def run():
    p=load_predictions();rows=[]
    keys=['target','band','k','draw','seed','factor']
    for key,d in p.groupby(keys):
        for group,g in groups(d,key[0]):rows.append(dict(zip(keys,key))|{'group':group,**metric(g)})
    m=pd.DataFrame(rows);m.to_csv(HERE/'seed-metrics.csv',index=False)
    baseline=m[m.factor.eq(1)].drop(columns='factor')
    paired=m[m.factor.ne(1)].merge(baseline,on=['target','band','k','draw','seed','group'],suffixes=('','_reference'),validate='many_to_one')
    for name in ['mae','rmse','error_p90']:paired[name+'_reduction']=paired[name+'_reference']-paired[name]
    paired['absolute_bias_reduction']=abs(paired.bias_reference)-abs(paired.bias)
    paired.to_csv(HERE/'paired-metrics.csv',index=False)
    aggregate=[]
    for key,d in m.groupby(['target','band','k','draw','factor','group']):
        row=dict(zip(['target','band','k','draw','factor','group'],key));row.update(people=int(d.people.iloc[0]),visits=int(d.visits.iloc[0]))
        assert set(d.seed)==set(SEEDS)
        for name in ['mae','bias','rmse','error_p90']:
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:row[name+'_'+suffix]=float(value)
        if row['factor']!=1:
            q=paired
            for name,value in zip(['target','band','k','draw','factor','group'],key):q=q[q[name].eq(value)]
            for name in ['mae','rmse','error_p90','absolute_bias']:
                delta=q[name+'_reduction'];row[name+'_reduction_mean']=float(delta.mean());row[name+'_improved_seeds']=int(delta.gt(0).sum())
            row['mae_reduction_percent']=100*row['mae_reduction_mean']/float(q.mae_reference.mean())
        aggregate.append(row)
    agg=pd.DataFrame(aggregate);agg.to_csv(HERE/'aggregate-metrics.csv',index=False)
    joint=paired[paired.group.eq('all')][['target','band','k','draw','seed','factor','mae_reduction']].merge(paired[paired.group.eq('tail')][['target','band','k','draw','seed','factor','mae_reduction']],on=['target','band','k','draw','seed','factor'],suffixes=('_all','_tail'),validate='one_to_one')
    joint['tail_better_global_not_worse']=joint.mae_reduction_tail.gt(0)&joint.mae_reduction_all.ge(0)
    joint.to_csv(HERE/'joint-tradeoffs.csv',index=False)
    intervals=[];rng=np.random.default_rng(20261030)
    for (target,band,k),d in p[p.draw.eq(0)].groupby(['target','band','k']):
        for group,g in groups(d,target):
            if group not in ['all','tail','non_tail']:continue
            g=g.copy();g['error']=abs(g.prediction-g.actual)
            # Equal person weighting within each seed; average errors, never predictions.
            person=g.groupby(['participantid','seed','factor']).error.mean().unstack(['seed','factor'])
            reference=person.loc[:,[(seed,1) for seed in SEEDS]].mean(axis=1)
            boot=rng.integers(0,len(person),size=(2000,len(person)))
            for factor in FACTORS:
                delta=reference-person.loc[:,[(seed,factor) for seed in SEEDS]].mean(axis=1)
                lo,hi=np.quantile(delta.to_numpy()[boot].mean(axis=1),[.025,.975])
                intervals.append({'target':target,'band':band,'k':int(k),'group':group,'factor':factor,'people':len(person),'mae_reduction':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),'people_mae_improved':int(delta.gt(0).sum()),'people_mae_worsened':int(delta.lt(0).sum())})
    save('paired-bootstrap.json',intervals)
    overview=[]
    for key,d in joint[joint.draw.eq(0)].groupby(['target','band','k','factor']):
        overview.append(dict(zip(['target','band','k','factor'],key))|{'global_mae_change':float(-d.mae_reduction_all.mean()),'tail_mae_change':float(-d.mae_reduction_tail.mean()),'tail_better_global_not_worse_seeds':int(d.tail_better_global_not_worse.sum())})
    trade=pd.DataFrame(overview);trade.to_csv(HERE/'condition-tradeoffs.csv',index=False)
    save('summary.json',{'seeds':SEEDS,'factors':[1]+FACTORS,'weighted_models':600,'selection':'fixed reference budget600 width/epochs per fold and seed, loss weights only; no optimized weight selected from test results','metrics':'original equal-person evaluation; five-seed mean errors, not ensemble','subgroups':'visit-label-defined, people can appear in both tail/non-tail; SPPB<=3 and TUG>20 descriptive groups','bootstrap':'2000 paired person draws conditional on fixed folds/models; mean of five-seed errors, not full retraining uncertainty or noninferiority proof','primary_tradeoffs':overview,'long5':agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail','non_tail'])].fillna(0).to_dict(orient='records'),'artifact_sha256':{name:ref.sha(HERE/name) for name in ['seed-metrics.csv','paired-metrics.csv','aggregate-metrics.csv','joint-tradeoffs.csv','condition-tradeoffs.csv','paired-bootstrap.json']}})
    print(trade.round(5).to_string(index=False),flush=True)
    print('LONG5',agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail','non_tail'])][['target','factor','group','people','visits','mae_mean','mae_min','mae_max','bias_mean','error_p90_mean']].round(5).to_string(index=False),flush=True)
if __name__=='__main__':run()
