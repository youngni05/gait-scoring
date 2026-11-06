from common import *

def bootstrap(delta):
    rng=np.random.default_rng(20261031); a=np.asarray(delta)
    draws=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1)
    return np.quantile(draws,[.025,.975]).tolist()

def main():
    audit=json.loads((HERE/'preparation.json').read_text()); metrics=[];comparisons=[];allpeople=[];allruns=[];baselines=[]
    for name in audit['cohorts']:
        p=pd.read_csv(HERE/name/'predictions.csv');p['error']=p.prediction-p.target;p['ae']=abs(p.error)
        p['under']=np.maximum(-p.error,0);p['over']=np.maximum(p.error,0)
        person=p.groupby(['model','view','id'],as_index=False).agg(target=('target','first'),prediction=('prediction','mean'),ae=('ae','mean'),under=('under','mean'),over=('over','mean'),prediction_sd=('prediction','std'),bias=('error','mean'))
        person['dataset']=name;allpeople.append(person)
        run=p.groupby(['model','view','repeat','seed'],as_index=False).agg(mae=('ae','mean'))
        run['dataset']=name;allruns.append(run)
        for (model,view),g in person.groupby(['model','view']):
            for group,gg in [('all',g),('ge20',g[g.target>=20]),('lt20',g[g.target<20])]:
                metrics.append({'dataset':name,'model':model,'view':view,'group':group,'n':len(gg),
                                'mae':gg.ae.mean(),'under':gg.under.mean(),'over':gg.over.mean(),'bias':gg.bias.mean()})
        for family in ['RF','Ridge']:
            a=person[(person.model==family)&(person.view=='speed')].set_index('id')
            b=person[(person.model==family)&(person.view=='pattern')].set_index('id').loc[a.index]
            median=person[person.model=='Median'].set_index('id').loc[a.index]
            for view,g in [('speed',a),('pattern',b)]:
                delta=g.ae-median.ae
                baselines.append({'dataset':name,'model':family,'view':view,'delta_model_minus_median':float(delta.mean()),'conditional_ci95':bootstrap(delta)})
            for group,mask in [('all',np.ones(len(a),bool)),('ge20',a.target.to_numpy()>=20)]:
                delta=(b.ae-a.ae)[mask];ud=(b.under-a.under)[mask]
                subset=p[p.model==family]
                if group=='ge20':subset=subset[subset.target>=20]
                paired=subset.groupby(['repeat','seed','view']).ae.mean().unstack('view')
                comparisons.append({'dataset':name,'model':family,'group':group,'n':len(delta),'mae_delta_pattern_minus_speed':float(delta.mean()),
                    'conditional_ci95':bootstrap(delta),'under_delta':float(ud.mean()),'under_ci95':bootstrap(ud),
                    'people_improved':int((delta<0).sum()),
                    'paired_run_wins':int((paired.pattern<paired.speed).sum()),'paired_runs':len(paired)})
    persons=pd.concat(allpeople,ignore_index=True);persons.to_csv(HERE/'person-results.csv',index=False)
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False);pd.concat(allruns).to_csv(HERE/'run-metrics.csv',index=False)
    save('comparisons.json',comparisons)
    save('baseline-comparisons.json',baselines)
    duration=[]
    for outcome in ['STS','TUG']:
        for family in ['RF','Ridge']:
            for view in ['speed','pattern']:
                a=persons[(persons.dataset==outcome+'_common3')&(persons.model==family)&(persons.view==view)].set_index('id')
                b=persons[(persons.dataset==outcome+'_common6')&(persons.model==family)&(persons.view==view)].set_index('id').loc[a.index]
                delta=b.ae-a.ae
                duration.append({'outcome':outcome,'model':family,'view':view,'n':len(a),'mae_delta_6_minus_3':float(delta.mean()),'conditional_ci95':bootstrap(delta)})
    save('duration-comparisons.json',duration)
    print(pd.DataFrame(metrics).query("model == 'RF'").to_string(index=False),flush=True)

if __name__=='__main__':main()
