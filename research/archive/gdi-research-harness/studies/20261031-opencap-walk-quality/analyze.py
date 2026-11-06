from common import *

def ci(a):
    a=np.asarray(a);rng=np.random.default_rng(20261031)
    return np.quantile(a[rng.integers(0,len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()

def masks(p):
    return {'all':np.ones(len(p),bool),'stable':p.stable.to_numpy(),'flagged':~p.stable.to_numpy(),
            'tail':p.target.to_numpy()>=20,'stable_tail':p.stable.to_numpy()&(p.target.to_numpy()>=20),
            'flagged_tail':~p.stable.to_numpy()&(p.target.to_numpy()>=20)}

def main():
    meta=json.loads((HERE/'preparation.json').read_text());metrics=[];comparisons=[];people=[];runs=[]
    for name in meta['cohorts']:
        d=load_data(name);p=pd.read_csv(HERE/name/'predictions.csv');p['stable']=p.id.map(dict(zip(d['ids'],d['stable'])))
        p['error']=p.prediction-p.target;p['ae']=abs(p.error);p['under']=np.maximum(-p.error,0);p['over']=np.maximum(p.error,0)
        agg=p.groupby(['model','view','id'],as_index=False).agg(target=('target','first'),stable=('stable','first'),prediction=('prediction','mean'),prediction_sd=('prediction','std'),ae=('ae','mean'),under=('under','mean'),over=('over','mean'))
        agg['dataset']=name;people.append(agg)
        for (model,view),g in agg.groupby(['model','view']):
            for group,mask in masks(g).items():
                z=g[mask]
                metrics.append({'dataset':name,'model':model,'view':view,'group':group,'n':len(z),
                                'mae':float(z.ae.mean()),'under':float(z.under.mean()),'over':float(z.over.mean())})
        for model in ['RF','Ridge']:
            for new,base in [('quality','speed'),('pattern','quality'),('robust','quality'),('robust','pattern'),('pattern','speed'),('robust','speed')]:
                a=agg[(agg.model==model)&(agg.view==base)].set_index('id')
                b=agg[(agg.model==model)&(agg.view==new)].set_index('id').loc[a.index]
                for group,mask in masks(a).items():
                    delta=(b.ae-a.ae)[mask];ud=(b.under-a.under)[mask]
                    ids=a.index[mask];sub=p[(p.model==model)&p.id.isin(ids)]
                    rm=sub.groupby(['repeat','seed','view']).ae.mean().unstack('view')
                    comparisons.append({'dataset':name,'model':model,'new':new,'base':base,'group':group,'n':len(delta),
                                        'mae_delta':float(delta.mean()),'ci95':ci(delta),'under_delta':float(ud.mean()),'under_ci95':ci(ud),
                                        'people_improved':int((delta<0).sum()),'run_wins':int((rm[new]<rm[base]).sum()),'runs':len(rm)})
        for group,mask in masks(p).items():
            z=p[mask].groupby(['model','view','repeat','seed'],as_index=False).agg(mae=('ae','mean'),under=('under','mean'),over=('over','mean'))
            z['dataset']=name;z['group']=group;runs.append(z)
    pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False);pd.concat(people).to_csv(HERE/'person-results.csv',index=False)
    pd.concat(runs).to_csv(HERE/'run-results.csv',index=False);save('comparisons.json',comparisons)
    print(pd.DataFrame(metrics).query("model=='RF' and group in ['all','stable','tail','stable_tail']").to_string(index=False),flush=True)

if __name__=='__main__':main()
