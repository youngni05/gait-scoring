from train import HERE,BASE,MLP,save,sha
import numpy as np
import pandas as pd
import json

def run():
    keys=['participantid','visit_number','draw','fold','target','band','k']
    p=pd.read_csv(HERE/'predictions.csv');original=pd.read_csv(BASE/'predictions.csv');mlp=pd.read_csv(MLP/'predictions.csv')
    p=p.merge(original[keys+['actual','baseline','SVR','cohort']],on=keys,validate='one_to_one',suffixes=('','_svr'))
    p=p.merge(mlp[keys+['actual','MLP']],on=keys,validate='one_to_one',suffixes=('','_mlp'))
    assert len(p)==len(original)==len(mlp)
    np.testing.assert_allclose(p.actual,p.actual_svr,atol=1e-12,rtol=0)
    np.testing.assert_allclose(p.actual,p.actual_mlp,atol=1e-12,rtol=0)
    p=p.drop(columns=['actual_svr','actual_mlp']);p.to_csv(HERE/'paired-predictions.csv',index=False)
    models=['baseline','SVR','MLP','DeepSets'];metrics=[];contrasts=[];subgroups=[]
    rng=np.random.default_rng(20261030)
    for (target,band,k,draw),g in p.groupby(['target','band','k','draw']):
        e=g[['participantid']].copy()
        for model in models:e[model]=abs(g.actual-g[model])
        person=e.groupby('participantid')[models].mean()
        boot=rng.integers(0,len(person),size=(2000,len(person))) if draw==0 else None
        row={'target':target,'band':band,'k':int(k),'draw':int(draw),'people':len(person),'visits':len(g)}
        for model in models:row[model+'_mae']=float(person[model].mean())
        for reference in ['baseline','SVR','MLP']:
            delta=person[reference]-person.DeepSets
            row[reference+'_improvement']=float(delta.mean())
            row[reference+'_improvement_percent']=float(100*delta.mean()/person[reference].mean())
            if draw==0:
                lo,hi=np.quantile(delta.to_numpy()[boot].mean(axis=1),[.025,.975])
                row[reference+'_improvement_ci_low']=float(lo);row[reference+'_improvement_ci_high']=float(hi)
        metrics.append(row)
        if draw==0:
            cuts=[-np.inf,3,6,9,np.inf] if target=='sppb' else [-np.inf,10,20,np.inf]
            g=g.copy();g['score_band']=pd.cut(g.actual,cuts)
            for group_type in ['cohort','score_band']:
                for label,d in g.groupby(group_type,observed=True):
                    for model in models:
                        v=pd.DataFrame({'id':d.participantid,'error':abs(d.actual-d[model])}).groupby('id').error.mean()
                        subgroups.append({'target':target,'band':band,'k':int(k),'group_type':group_type,'group':str(label),'model':model,'people':len(v),'visits':len(d),'person_mae':float(v.mean())})
    metrics=pd.DataFrame(metrics);metrics.to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(subgroups).to_csv(HERE/'subgroup-metrics.csv',index=False)
    for target in ['sppb','tug']:
        d=p[p.target.eq(target)&p.draw.eq(0)].copy();d['error']=abs(d.actual-d.DeepSets)
        e=d.groupby(['participantid','band','k']).error.mean().unstack(['band','k'])
        boot=rng.integers(0,len(e),size=(2000,len(e)))
        for a,b in [(('short',k),('long',k)) for k in [1,3,5]]+[((band,1),(band,5)) for band in ['short','long']]:
            delta=e[a]-e[b];ci=np.quantile(delta.to_numpy()[boot].mean(axis=1),[.025,.975])
            contrasts.append({'target':target,'from':f'{a[0]}-{a[1]}','to':f'{b[0]}-{b[1]}','mae_reduction':float(delta.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])})
    save('condition-contrasts.json',contrasts)
    primary=metrics[metrics.draw.eq(0)]
    best=[primary[primary.target.eq(t)].sort_values('DeepSets_mae').iloc[0].to_dict() for t in ['sppb','tug']]
    save('summary.json',{'best_primary_deep_sets_conditions':best,'primary_draw':0,'resampled_inputs':'draws1/2 use the same fitted models, no retraining or prediction averaging','interval':'2000 paired person bootstrap on fixed OOF predictions; no refitting, selection adjustment, or independent final test','initialization':'same single seed schedule as MLP','models':60,'parameter_counts':[705,2433],'comparability':'same people, visits, outer/inner person splits, selected bouts, four features, width/epoch grid, optimizer and loss as MLP; training X scaling uses raw selected bouts instead of median inputs'})
    print(primary[['target','band','k','SVR_mae','MLP_mae','DeepSets_mae','SVR_improvement_percent','MLP_improvement_percent','SVR_improvement_ci_low','SVR_improvement_ci_high']].round(5).to_string(index=False),flush=True)
    print('BEST',json.dumps(best,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
