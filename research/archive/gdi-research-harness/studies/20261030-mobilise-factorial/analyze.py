"""Paired factorial effects on the same held-out people; lower error is better."""
from train import HERE,BASE,SEEDS,VARIANTS,KEY,ref,save
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
CONTRASTS={'input_mean':[-1,1,0,0],'input_meanmax':[0,0,-1,1],'structure_core':[-1,0,1,0],'structure_expanded':[0,-1,0,1],'combined':[-1,0,0,1],'interaction':[1,-1,-1,1],'input_average':[-.5,.5,-.5,.5],'structure_average':[-.5,-.5,.5,.5]}
METRICS=['mae','rmse','error_p90','bias']

def load():
    p=pd.read_csv(HERE/'predictions.csv.gz')
    original_rows=len(p)
    meta=pd.read_csv(BASE/'features.csv',usecols=['participantid','cohort']).drop_duplicates()
    assert not meta.participantid.duplicated().any()
    p=p.merge(meta,on='participantid',validate='many_to_one')
    flags=pd.read_csv(HERE/'label-review-flags.csv')
    p=p.merge(flags[['participantid','visit_number','sppb_component_review','tug_range_review']],on=['participantid','visit_number'],validate='many_to_one')
    bouts=pd.read_csv(BASE/'selected-bouts.csv');bouts['rate']=bouts.turn_number_so/bouts.duration
    rates=[]
    for k in [1,3,5]:
        r=bouts[bouts['rank'].le(k)].groupby(ref.BAGKEY).rate.max().rename('max_turn_rate').reset_index();r['k']=k;rates.append(r)
    p=p.merge(pd.concat(rates,ignore_index=True),on=ref.BAGKEY+['k'],validate='many_to_one')
    assert len(p)==original_rows and not p.duplicated(KEY).any()
    assert p[['cohort','sppb_component_review','tug_range_review','max_turn_rate']].notna().all().all()
    return p

def groups(d,target):
    tail=d.actual.le(3) if target=='sppb' else d.actual.gt(20)
    yield 'all',d;yield 'tail',d[tail];yield 'non_tail',d[~tail]
    if target=='sppb':
        yield 'middle',d[d.actual.gt(3)&d.actual.le(9)];yield 'high',d[d.actual.gt(9)]
    for cohort,g in d.groupby('cohort'):yield 'cohort_'+cohort,g
    flag=d.sppb_component_review if target=='sppb' else d.tug_range_review
    yield 'unflagged_all',d[~flag];yield 'unflagged_tail',d[~flag&tail]
    # Descriptive high-rate sensitivity only; 1 turn/s is not a validated exclusion threshold.
    yield 'turn_rate_below1_all',d[d.max_turn_rate.lt(1)]
    yield 'turn_rate_below1_tail',d[d.max_turn_rate.lt(1)&tail]

def metric(d,variant):
    e=(d[variant]-d.actual).to_numpy();w=ref.weights(d.participantid.to_numpy());order=np.argsort(abs(e))
    q=abs(e[order][min(np.searchsorted(np.cumsum(w[order]),.9),len(e)-1)])
    return {'people':int(d.participantid.nunique()),'visits':len(d),'mae':float(np.dot(w,abs(e))),'bias':float(np.dot(w,e)),'rmse':float(np.sqrt(np.dot(w,e**2))),'error_p90':float(q)}

def run():
    p=load();rows=[];keys=['target','band','k','draw','seed']
    for key,d in p.groupby(keys):
        for group,g in groups(d,key[0]):
            for variant in VARIANTS:rows.append(dict(zip(keys,key))|{'group':group,'variant':variant,**metric(g,variant)})
    m=pd.DataFrame(rows);m.to_csv(HERE/'seed-metrics.csv',index=False)
    index=['target','band','k','draw','seed','group'];effects=[]
    for key,d in m.groupby(index):
        d=d.set_index('variant').loc[list(VARIANTS)]
        for contrast,c in CONTRASTS.items():
            effects.append(dict(zip(index,key))|{'contrast':contrast,**{metric+'_change':float(np.dot(c,d[metric])) for metric in METRICS}})
    e=pd.DataFrame(effects);e.to_csv(HERE/'seed-factorial-effects.csv',index=False)
    aggregate=[]
    for key,d in m.groupby(['target','band','k','draw','variant','group']):
        assert set(d.seed)==set(SEEDS)
        row=dict(zip(['target','band','k','draw','variant','group'],key))|{'people':int(d.people.iloc[0]),'visits':int(d.visits.iloc[0])}
        for name in METRICS:
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:row[name+'_'+suffix]=float(value)
        aggregate.append(row)
    agg=pd.DataFrame(aggregate);agg.to_csv(HERE/'aggregate-metrics.csv',index=False)
    ef=[]
    for key,d in e.groupby(['target','band','k','draw','contrast','group']):
        row=dict(zip(['target','band','k','draw','contrast','group'],key))
        for name in METRICS:
            change=d[name+'_change'];row[name+'_change_mean']=float(change.mean());row[name+'_improved_seeds']=int(change.lt(0).sum());row[name+'_change_min']=float(change.min());row[name+'_change_max']=float(change.max())
        ef.append(row)
    effects_agg=pd.DataFrame(ef);effects_agg.to_csv(HERE/'aggregate-factorial-effects.csv',index=False)
    joint=[]
    for allgroup,tailgroup,view in [('all','tail','primary'),('unflagged_all','unflagged_tail','label_sensitivity'),('turn_rate_below1_all','turn_rate_below1_tail','turn_rate_sensitivity')]:
        keys=['target','band','k','draw','seed','contrast']
        j=e[e.group.eq(allgroup)].merge(e[e.group.eq(tailgroup)],on=keys,validate='one_to_one',suffixes=('_all','_tail'))
        j['overall_not_worse']=j.mae_change_all.le(0)
        j['tail_better_global_not_worse']=j.overall_not_worse&j.mae_change_tail.lt(0)
        j['tail_mae_and_p90_better_global_not_worse']=j.tail_better_global_not_worse&j.error_p90_change_tail.lt(0)
        j['view']=view;joint.append(j)
    joint=pd.concat(joint,ignore_index=True);joint.to_csv(HERE/'seed-tradeoffs.csv',index=False)
    conditions=[]
    for key,d in joint.groupby(['target','band','k','draw','contrast','view']):
        conditions.append(dict(zip(['target','band','k','draw','contrast','view'],key))|{'global_mae_change':float(d.mae_change_all.mean()),'tail_mae_change':float(d.mae_change_tail.mean()),'tail_p90_change':float(d.error_p90_change_tail.mean()),'tail_rmse_change':float(d.rmse_change_tail.mean()),'global_not_worse_seeds':int(d.overall_not_worse.sum()),'tail_better_global_not_worse_seeds':int(d.tail_better_global_not_worse.sum()),'tail_mae_p90_better_global_not_worse_seeds':int(d.tail_mae_and_p90_better_global_not_worse.sum())})
    conditions=pd.DataFrame(conditions);conditions.to_csv(HERE/'condition-tradeoffs.csv',index=False)
    rng=np.random.default_rng(20261030);intervals=[]
    for (target,band,k),d in p[p.draw.eq(0)].groupby(['target','band','k']):
        for group,g in groups(d,target):
            if group not in ['all','tail','unflagged_all','unflagged_tail']:continue
            errors=abs(g[list(VARIANTS)].sub(g.actual,axis=0));errors['participantid']=g.participantid
            # Every person has the same visits in all five seeds. Average errors, never predictions.
            people=errors.groupby('participantid')[list(VARIANTS)].mean()
            boot=rng.integers(0,len(people),size=(2000,len(people)))
            for contrast,c in CONTRASTS.items():
                delta=people.to_numpy()@np.array(c);lo,hi=np.quantile(delta[boot].mean(axis=1),[.025,.975])
                intervals.append({'target':target,'band':band,'k':int(k),'group':group,'contrast':contrast,'people':len(people),'mae_change':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi),'people_improved':int((delta<0).sum()),'people_worsened':int((delta>0).sum())})
    save('paired-bootstrap.json',intervals)
    ledgers=[json.loads(path.read_text()) for path in (HERE/'folds').glob('*.json')]
    epochs=pd.DataFrame([{k:d[k] for k in ['target','band','k','fold','seed','variant','width','epochs','parameters','inner_person_mae']} for d in ledgers]);epochs.to_csv(HERE/'selections.csv',index=False)
    checks=conditions[conditions.draw.eq(0)&conditions.view.eq('primary')&conditions.contrast.isin(['input_mean','structure_core','combined'])]
    overview=[]
    for contrast,d in checks.groupby('contrast'):
        overview.append({'contrast':contrast,'conditions':len(d),'mean_global_not_worse':int(d.global_mae_change.le(0).sum()),'mean_tail_improved':int(d.tail_mae_change.lt(0).sum()),'mean_tail_p90_improved':int(d.tail_p90_change.lt(0).sum()),'mean_both_criteria':int((d.global_mae_change.le(0)&d.tail_mae_change.lt(0)).sum()),'all5seeds_both_criteria':int(d.tail_better_global_not_worse_seeds.eq(5).sum()),'all5seeds_all3criteria':int(d.tail_mae_p90_better_global_not_worse_seeds.eq(5).sum())})
    save('summary.json',{'metrics':'equal-person evaluation of held-out predictions; five-seed mean metrics, not an ensemble','variant_names':json.loads((HERE/'protocol.json').read_text())['variants'],'negative_changes':'lower errors than the indicated paired comparison; interaction is D-B-C+A, not a standalone model','overview_primary_12_conditions':overview,'long5':agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail','non_tail'])].to_dict(orient='records'),'short5':agg[agg.draw.eq(0)&agg.band.eq('short')&agg.k.eq(5)&agg.group.isin(['all','tail'])].to_dict(orient='records'),'long5_tradeoffs':conditions[conditions.draw.eq(0)&conditions.band.eq('long')&conditions.k.eq(5)].to_dict(orient='records'),'epoch_search_max_selected':int(epochs.epochs.max()),'epoch_search_boundary_count':int(epochs.epochs.eq(600).sum()),'uncertainty':'2000 paired person bootstrap draws conditional on fixed OOF models; seed stability is not a proof of population noninferiority. Reused exploratory outer folds, not independent final validation.','artifact_sha256':{name:ref.sha(HERE/name) for name in ['seed-metrics.csv','seed-factorial-effects.csv','aggregate-metrics.csv','aggregate-factorial-effects.csv','seed-tradeoffs.csv','condition-tradeoffs.csv','paired-bootstrap.json','selections.csv']}})
    print(json.dumps(overview,indent=2),flush=True)
    print(agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail'])][['target','variant','group','mae_mean','mae_min','mae_max','error_p90_mean','rmse_mean','bias_mean']].round(6).to_string(index=False),flush=True)
    print('ANALYSIS_FINISHED',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
