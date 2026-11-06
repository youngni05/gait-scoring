"""Compare the internally constrained procedure with fixed same-seed references."""
from train import HERE,WEIGHTED,REFERENCE,KEY,SEEDS,ref,save
import importlib.util,json
import numpy as np
import pandas as pd
# Reuse only unchanged subgroup/metric helpers; never run the fixed-weight analysis.
spec=importlib.util.spec_from_file_location('metric_helpers',WEIGHTED/'analyze.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def run():
    new=pd.read_csv(HERE/'predictions.csv').rename(columns={'prediction':'constrained'})
    old=pd.read_csv(REFERENCE/'predictions.csv');old=old[old.budget.eq(600)].rename(columns={'prediction':'baseline'})
    assert len(new)==len(old) and not new.duplicated(KEY).any()
    p=new.merge(old[KEY+['actual','baseline']],on=KEY,validate='one_to_one',suffixes=('','_reference'))
    np.testing.assert_array_equal(p.actual,p.actual_reference);p=p.drop(columns='actual_reference')
    metadata=pd.read_csv(ref.HERE/'paired-predictions.csv')[KEY[:-1]+['cohort']]
    p=p.merge(metadata,on=KEY[:-1],validate='many_to_one');p.to_csv(HERE/'paired-predictions.csv',index=False)
    rows=[];keys=['target','band','k','draw','seed']
    for key,d in p.groupby(keys):
        for group,g in helpers.groups(d,key[0]):
            for model in ['baseline','constrained']:
                rows.append(dict(zip(keys,key))|{'group':group,'model':model,**helpers.metric(g.rename(columns={model:'prediction'}))})
    m=pd.DataFrame(rows);m.to_csv(HERE/'seed-metrics.csv',index=False)
    a=m[m.model.eq('baseline')].drop(columns='model');b=m[m.model.eq('constrained')].drop(columns='model')
    paired=b.merge(a,on=keys+['group'],suffixes=('','_reference'),validate='one_to_one')
    for name in ['mae','rmse','error_p90']:paired[name+'_reduction']=paired[name+'_reference']-paired[name]
    paired['absolute_bias_reduction']=abs(paired.bias_reference)-abs(paired.bias)
    paired.to_csv(HERE/'paired-metrics.csv',index=False)
    aggregate=[]
    for key,d in paired.groupby(['target','band','k','draw','group']):
        row=dict(zip(['target','band','k','draw','group'],key));row.update(people=int(d.people.iloc[0]),visits=int(d.visits.iloc[0]))
        assert set(d.seed)==set(SEEDS)
        for name in ['mae','bias','rmse','error_p90']:
            row[name+'_reference_mean']=float(d[name+'_reference'].mean())
            for suffix,value in [('mean',d[name].mean()),('sd',d[name].std(ddof=1)),('min',d[name].min()),('max',d[name].max())]:row[name+'_'+suffix]=float(value)
        for name in ['mae','rmse','error_p90','absolute_bias']:
            delta=d[name+'_reduction'];row[name+'_reduction_mean']=float(delta.mean());row[name+'_better_seeds']=int(delta.gt(0).sum());row[name+'_same_seeds']=int(delta.eq(0).sum());row[name+'_worse_seeds']=int(delta.lt(0).sum())
        aggregate.append(row)
    agg=pd.DataFrame(aggregate);agg.to_csv(HERE/'aggregate-metrics.csv',index=False)
    joint=paired[paired.group.eq('all')][keys+['mae_reduction']].merge(paired[paired.group.eq('tail')][keys+['mae_reduction']],on=keys,suffixes=('_all','_tail'),validate='one_to_one')
    joint['tail_better_global_not_worse']=joint.mae_reduction_tail.gt(0)&joint.mae_reduction_all.ge(0)
    joint.to_csv(HERE/'joint-tradeoffs.csv',index=False)
    overview=[]
    for key,d in joint[joint.draw.eq(0)].groupby(['target','band','k']):
        overview.append(dict(zip(['target','band','k'],key))|{'global_mae_change':float(-d.mae_reduction_all.mean()),'tail_mae_change':float(-d.mae_reduction_tail.mean()),'global_worse_seeds':int(d.mae_reduction_all.lt(0).sum()),'tail_better_seeds':int(d.mae_reduction_tail.gt(0).sum()),'both_objectives_met_seeds':int(d.tail_better_global_not_worse.sum())})
    pd.DataFrame(overview).to_csv(HERE/'condition-tradeoffs.csv',index=False)
    selection=[]
    for path in sorted((HERE/'folds').glob('*.json')):
        x=json.loads(path.read_text());s=x['selected'];base=x['baseline']
        selection.append({name:x[name] for name in ['target','band','k','fold','seed','width','feasible_candidates']}|{'factor':s['factor'],'epochs':s['epochs'],'reference_epochs':base['epochs'],'inner_mae':s['inner_mae'],'reference_inner_mae':base['inner_mae'],'inner_tail_mae':s['inner_tail_mae'],'reference_inner_tail_mae':base['inner_tail_mae'],'fallback':s['factor']==1,'inner_global_reduction':base['inner_mae']-s['inner_mae'],'inner_tail_reduction':base['inner_tail_mae']-s['inner_tail_mae']})
    selection=pd.DataFrame(selection);selection.to_csv(HERE/'selections.csv',index=False)
    intervals=[];rng=np.random.default_rng(20261030)
    for (target,band,k),d in p[p.draw.eq(0)].groupby(['target','band','k']):
        for group,g in helpers.groups(d,target):
            if group not in ['all','tail','non_tail']:continue
            g=g.copy();g['difference']=abs(g.baseline-g.actual)-abs(g.constrained-g.actual)
            person=g.groupby(['participantid','seed']).difference.mean().groupby('participantid').mean()
            boot=rng.integers(0,len(person),size=(2000,len(person)))
            lo,hi=np.quantile(person.to_numpy()[boot].mean(axis=1),[.025,.975])
            intervals.append({'target':target,'band':band,'k':int(k),'group':group,'people':len(person),'mae_reduction':float(person.mean()),'ci_low':float(lo),'ci_high':float(hi),'people_improved':int(person.gt(0).sum()),'people_unchanged':int(person.eq(0).sum()),'people_worsened':int(person.lt(0).sum())})
    save('paired-bootstrap.json',intervals)
    save('summary.json',{'models':300,'seeds':SEEDS,'primary_draw':0,'inner_constraint':'zero slack against same-seed/fold baseline inner MAE; no guarantee on held-out people','objective':'tail inner MAE, with overall MAE feasibility; p90/RMSE are separately evaluated outcomes','selected_factors':{str(k):int(v) for k,v in selection.factor.value_counts().items()},'epoch_range':[int(selection.epochs.min()),int(selection.epochs.max())],'all_inner_constraints_pass':bool(selection.inner_global_reduction.ge(0).all()),'all_inner_tail_not_worse':bool(selection.inner_tail_reduction.ge(0).all()),'primary_tradeoffs':overview,'long5':agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail','non_tail'])].to_dict(orient='records'),'uncertainty':'2000 paired-person bootstrap of five-seed mean errors conditional on fixed splits/models; exploratory same-development-data reuse, not external or clinical noninferiority validation','artifact_sha256':{name:ref.sha(HERE/name) for name in ['paired-predictions.csv','seed-metrics.csv','paired-metrics.csv','aggregate-metrics.csv','joint-tradeoffs.csv','condition-tradeoffs.csv','selections.csv','paired-bootstrap.json']}})
    print(pd.DataFrame(overview).round(6).to_string(index=False),flush=True)
    print('LONG5',agg[agg.draw.eq(0)&agg.band.eq('long')&agg.k.eq(5)&agg.group.isin(['all','tail','non_tail'])][['target','group','mae_reference_mean','mae_mean','mae_min','mae_max','mae_better_seeds','mae_worse_seeds','bias_mean','error_p90_reference_mean','error_p90_mean']].round(6).to_string(index=False),flush=True)
    print('SELECTED FACTORS',selection.factor.value_counts().to_dict(),flush=True)
if __name__=='__main__':run()
