from common import *
import pandas as pd

def metrics(target,pred):
    e=pred-target
    return {'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt((e**2).mean())),'bias':float(e.mean()),'underprediction':float(np.maximum(-e,0).mean()),'overprediction':float(np.maximum(e,0).mean())}

def paired_ci(delta):
    rng=np.random.default_rng(20261031);means=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return {'mean':float(delta.mean()),'conditional_person_bootstrap95':np.quantile(means,[.025,.975]).tolist()}

def main():
    assert json.loads((HERE/'verification.json').read_text())['passed']
    d=load();y=d['y'];pred=pd.read_csv(HERE/'predictions.csv');method_map=METHODS.copy()
    if (HERE/'followup-completion.json').exists():
        pred=pd.concat([pred,pd.read_csv(HERE/'followup-predictions.csv')],ignore_index=True);method_map.update(FOLLOWUP_METHODS)
    groups={'all':np.ones(len(y),dtype=bool),'short':y<10,'mid':(y>=10)&(y<20),'long':y>=20}
    rows=[];personrows=[]
    for (method,repeat,seed),g in pred.groupby(['method','repeat','seed']):
        g=g.sort_values('sample');assert len(g)==298
        for band,mask in groups.items():rows.append({'method':method,'repeat':int(repeat),'seed':int(seed),'group':band,'people':int(mask.sum()),**metrics(y[mask],g.prediction.to_numpy()[mask])})
    run=pd.DataFrame(rows);run.to_csv(HERE/'run-metrics.csv',index=False)
    pred['abs_error']=abs(pred.prediction-pred.target);pred['signed_error']=pred.prediction-pred.target
    for (method,sample),g in pred.groupby(['method','sample']):
        personrows.append({'method':method,'sample':int(sample),'id':d['ids'][sample],'target':float(y[sample]),'MAE':float(g.abs_error.mean()),'mean_prediction':float(g.prediction.mean()),'bias':float(g.signed_error.mean()),'prediction_std':float(g.prediction.std())})
    people=pd.DataFrame(personrows);people.to_csv(HERE/'person-results.csv',index=False)
    summaries={}
    for method,g in run.groupby('method'):
        summaries[method]={}
        for band,z in g.groupby('group'):
            summaries[method][band]={'people':int(z.people.iloc[0]),'runs':len(z),**{k:float(z[k].mean()) for k in ['MAE','RMSE','bias','underprediction','overprediction']},'MAE_range':[float(z.MAE.min()),float(z.MAE.max())]}
    baseline=people[people.method=='BaselineRF'].set_index('sample').reindex(range(298));comparisons={}
    for method in method_map:
        if method=='BaselineRF':continue
        other=people[people.method==method].set_index('sample').reindex(range(298));delta=(other.MAE-baseline.MAE).to_numpy();result={}
        for band,mask in groups.items():result[band]=paired_ci(delta[mask])
        longdelta=delta[y>=20]
        result['long_people_improved']=int((longdelta<0).sum());result['long_people_worsened']=int((longdelta>0).sum())
        result['leave_one_long_person_out_mean_difference_range']=[float(min(np.delete(longdelta,i).mean() for i in range(9))),float(max(np.delete(longdelta,i).mean() for i in range(9)))]
        a=run[(run.method==method)&(run.group=='long')];b=run[(run.method=='BaselineRF')&(run.group=='long')]
        if method_map[method][3]=='Ridge':b=b.groupby('repeat',as_index=False).MAE.mean();joined=a.merge(b,on='repeat',suffixes=('','_base'))
        else:joined=a.merge(b,on=['repeat','seed'],suffixes=('','_base'))
        result['long_MAE_lower_runs']=int((joined.MAE<joined.MAE_base).sum());result['paired_runs']=len(joined)
        comparisons[method]=result
    long=people[people.target>=20].pivot(index=['sample','id','target'],columns='method',values=['MAE','mean_prediction'])
    long.to_csv(HERE/'long-person-comparison.csv')
    selections=json.loads((HERE/'selections.json').read_text());counts={}
    if (HERE/'followup-completion.json').exists():selections+=json.loads((HERE/'followup-selections.json').read_text())
    for method in method_map:
        values=[str(r['setting']) for r in selections if r['method']==method];counts[method]={s:values.count(s) for s in sorted(set(values))}
    result={'people':298,'unique_long_people':9,'models':summaries,'differences_vs_baseline':comparisons,'selection_counts':counts,'scope':'All estimates reuse the same development cohort. Nine long people, not 135 independent cases; conditional bootstrap does not capture training or selection uncertainty. Balanced selection deliberately trades group errors; no model selected using outer test targets.'}
    save('summary.json',result)
    print(json.dumps({m:{'all':v['all']['MAE'],'short':v['short']['MAE'],'long':v['long']['MAE'],'long_bias':v['long']['bias']} for m,v in summaries.items()},indent=2))

if __name__=='__main__':main()
