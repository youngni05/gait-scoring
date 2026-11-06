from study import *
LARGE=['sub-246','sub-266','sub-310','sub-406']
METRICS=['MAE','underprediction','overprediction','bias']
def band(g,name):return g[g.id.isin(LARGE)] if name=='large4' else g[g.target>=20] if name=='ge20' else g
def main():
    out=HERE/'cohort176';p=pd.read_csv(out/'predictions.csv');err=p.prediction-p.target
    p['MAE']=abs(err);p['underprediction']=np.maximum(-err,0);p['overprediction']=np.maximum(err,0);p['bias']=err
    person=p.groupby(['view','id']).agg(target=('target','first'),prediction=('prediction','mean'),**{k:(k,'mean') for k in METRICS}).reset_index();person.to_csv(out/'person-results.csv',index=False)
    metrics=[];runs=[];cases=[]
    for view,g in person.groupby('view'):
        for name in ['all','ge20','large4']:
            z=band(g,name);metrics.append({'view':view,'band':name,'people':len(z),**{k:float(z[k].mean()) for k in METRICS}})
        for rep,rg in p[p.view==view].groupby('repeat'):
            for name in ['all','ge20','large4']:
                z=band(rg,name);runs.append({'view':view,'repeat':int(rep),'band':name,**{k:float(z[k].mean()) for k in METRICS}})
        for pid,cg in p[(p.view==view)&(p.target>=20)].groupby('id'):
            cases.append({'view':view,'id':pid,'target':float(cg.target.iloc[0]),'prediction':float(cg.prediction.mean()),'sd':float(cg.prediction.std(ddof=0)),'minimum':float(cg.prediction.min()),'maximum':float(cg.prediction.max()),**{k:float(cg[k].mean()) for k in METRICS}})
    contrasts=[];run=pd.DataFrame(runs)
    pairs=[(v,'Matched') for v in person.view.unique() if v not in ['Original','Matched']]+[('Matched','Original'),('AddBoth','Original')]
    for va,vb in pairs:
        for name in ['all','ge20','large4']:
            a=band(person[person.view==va],name).set_index('id').sort_index();b=band(person[person.view==vb],name).set_index('id').loc[a.index]
            ra=run[(run.view==va)&(run.band==name)].set_index('repeat');rb=run[(run.view==vb)&(run.band==name)].set_index('repeat').loc[ra.index]
            ix=np.random.default_rng(20261031).integers(len(a),size=(10000,len(a)));c={'a':va,'b':vb,'band':name,'people':len(a)}
            for k in METRICS:
                delta=(a[k]-b[k]).to_numpy();rd=ra[k]-rb[k];c[k]={'difference':float(delta.mean()),'conditional95':np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),'people_better':int(sum(delta<0)),'better_partitions':int(sum(rd<0)),'partition_deltas':rd.tolist()}
            contrasts.append(c)
    save('summary.json',{'metrics':metrics,'contrasts':contrasts});pd.DataFrame(metrics).to_csv(HERE/'metrics.csv',index=False);run.to_csv(HERE/'partition-metrics.csv',index=False);pd.DataFrame(cases).to_csv(HERE/'case-results.csv',index=False)
    print(pd.DataFrame(metrics)[['view','band','MAE','underprediction','overprediction']].to_string(index=False),flush=True)
if __name__=='__main__':main()
