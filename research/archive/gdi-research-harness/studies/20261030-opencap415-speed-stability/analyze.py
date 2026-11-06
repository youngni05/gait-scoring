"""Describe every repeat, paired input comparisons, and persistent large errors."""
from common import *
import math
import pandas as pd
from scipy.stats import spearmanr

def ci(values):
    rng=np.random.default_rng(20261030)
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return {'mean_difference_seconds':float(values.mean()),'conditional_person_bootstrap95':np.quantile(means,[.025,.975]).tolist()}

def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clean(v) for v in obj]
    if isinstance(obj,(np.integer,np.floating)):obj=obj.item()
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj

def main():
    d=load();y=d['y'];pred=pd.read_csv(HERE/'cpu-predictions.csv')
    if (HERE/'cnn-predictions.csv').exists():pred=pd.concat([pred,pd.read_csv(HERE/'cnn-predictions.csv')],ignore_index=True)
    pred['abs_error']=abs(pred.prediction-pred.target);pred['signed_error']=pred.prediction-pred.target
    runs=[]
    for (rep,seed,view,model),g in pred.groupby(['repeat','seed','view','model']):
        assert len(g)==298
        runs.append({'repeat':int(rep),'seed':int(seed),'view':view,'model':model,'MAE':float(g.abs_error.mean()),'RMSE':float(np.sqrt(np.mean(g.signed_error**2)))})
    runs=pd.DataFrame(runs);runs.to_csv(HERE/'run-metrics.csv',index=False)
    summary=[]
    for (view,model),g in runs.groupby(['view','model']):
        partition=g.groupby('repeat').MAE.mean()
        within=g.groupby('repeat').MAE.agg(lambda z:z.max()-z.min())
        summary.append({'view':view,'model':model,'runs':len(g),'mean_MAE':float(g.MAE.mean()),'min_MAE':float(g.MAE.min()),'max_MAE':float(g.MAE.max()),'std_across_runs':float(g.MAE.std()),'mean_RMSE':float(g.RMSE.mean()),'split_mean_range':[float(partition.min()),float(partition.max())],'mean_seed_range_within_split':float(within.mean()),'max_seed_range_within_split':float(within.max())})
    comparisons={};rf=pred[pred.model=='RF'];median=pred[pred.model=='Median']
    full=rf[rf.view=='full'];person_full=full.groupby('sample').abs_error.mean().reindex(range(298))
    for other in ['mean_speed','speed_summary','pelvis_motion','Median']:
        b=median if other=='Median' else rf[rf.view==other]
        person_b=b.groupby('sample').abs_error.mean().reindex(range(298))
        result=ci((person_full-person_b).to_numpy())
        a_run=runs[(runs.model=='RF')&(runs.view=='full')]
        b_run=runs[runs.model=='Median'] if other=='Median' else runs[(runs.model=='RF')&(runs.view==other)]
        join=['repeat'] if other=='Median' else ['repeat','seed']
        pairs=a_run.merge(b_run,on=join,suffixes=('_full','_other'))
        result.update(full_lower_MAE_runs=int((pairs.MAE_full<pairs.MAE_other).sum()),paired_runs=len(pairs),run_difference_min=float((pairs.MAE_full-pairs.MAE_other).min()),run_difference_max=float((pairs.MAE_full-pairs.MAE_other).max()))
        comparisons['full_RF_minus_'+other]=result
    # One row per person. Average absolute errors, not error of an averaged prediction.
    people=pd.read_csv(HERE/'manifest.csv');people['sample']=range(len(people))
    labels=pd.read_csv(SOURCE/'raw/participant_info.csv')
    assert not labels.duplicated(['subid','visit']).any()
    people=people.merge(labels,left_on=['id','visit'],right_on=['subid','visit'],validate='one_to_one')
    people=people.sort_values('sample').reset_index(drop=True);assert len(people)==298
    assert np.allclose(people.tft_tug,y,atol=1e-10)
    stats=full.groupby('sample').agg(mean_abs_error=('abs_error','mean'),mean_signed_error=('signed_error','mean'),mean_prediction=('prediction','mean'),prediction_std=('prediction','std'),worst_abs_error=('abs_error','max'))
    people=people.join(stats,on='sample')
    for view in VIEWS:
        people[f'{view}_RF_MAE']=rf[rf.view==view].groupby('sample').abs_error.mean().reindex(range(298)).to_numpy()
    people['baseline_MAE']=median.groupby('sample').abs_error.mean().reindex(range(298)).to_numpy()
    high=people.sort_values(['mean_abs_error','sample'],ascending=[False,True]).head(math.ceil(len(people)*.2))['sample']
    people['high_error']=people['sample'].isin(high)
    people['mean_speed']=d['x'][:,17,:].mean(1);people['speed_std']=d['x'][:,17,:].std(1)
    people['peak_speed']=d['x'][:,17,:].max(1)
    people['max_adjacent_joint_degrees']=np.max(np.abs(np.diff(d['x'][:,:17,:],axis=2)),axis=(1,2))
    overlap=json.loads((SOURCE/'overlap-summary.json').read_text())
    known=set(overlap['file_fingerprint_matches'].values());probable=set(overlap['clinical4times_height_weight_matches'].values())
    people['overlap']=np.where(people.id.isin(known),'file_confirmed',np.where(people.id.isin(probable),'clinical_candidate','not_identified'))
    for k,row in people.iterrows():
        f=SOURCE/'raw'/row.file;assert sha(f)==row.source_sha256
        text=f.read_text().splitlines();start=text.index('endheader')+2
        times=np.array([float(line.split()[0]) for line in text[start:] if line.strip()])
        people.loc[k,'source_max_dt']=float(np.diff(times).max())
    people['TUG_band']=pd.cut(people.target,[0,10,20,np.inf],right=False,labels=['<10','10-<20','>=20']).astype(str)
    columns=['sample','id','visit','target','diag','age','sex','ambulation_status','height','weight','overlap','TUG_band','mean_speed','speed_std','peak_speed','max_adjacent_joint_degrees','source_max_dt','source_seconds','active_max_seconds','mean_abs_error','mean_signed_error','mean_prediction','prediction_std','worst_abs_error','baseline_MAE','high_error']+[f'{v}_RF_MAE' for v in VIEWS]
    people[columns].to_csv(HERE/'person-error-analysis.csv',index=False)
    group_summaries=[]
    for column in ['TUG_band','diag','sex','ambulation_status','overlap','high_error']:
        for value,g in people.groupby(column,dropna=False):
            group_summaries.append({'group':column,'value':str(value),'people':len(g),'full_RF_MAE':float(g.mean_abs_error.mean()),'mean_speed_RF_MAE':float(g.mean_speed_RF_MAE.mean()),'baseline_MAE':float(g.baseline_MAE.mean()),'mean_signed_error':float(g.mean_signed_error.mean()),'high_error_people':int(g.high_error.sum()),'TUG_median':float(g.target.median()),'age_median':float(g.age.median()),'mean_speed_median':float(g.mean_speed.median()),'speed_std_median':float(g.speed_std.median()),'max_joint_step_median':float(g.max_adjacent_joint_degrees.median()),'source_max_dt_max':float(g.source_max_dt.max())})
    pd.DataFrame(group_summaries).to_csv(HERE/'error-groups.csv',index=False)
    correlations={}
    for key in ['target','age','height','weight','mean_speed','speed_std','peak_speed','max_adjacent_joint_degrees','source_seconds','active_max_seconds']:
        finite=np.isfinite(people[key]);correlations[key]={'people':int(finite.sum()),'spearman_with_mean_absolute_error':float(spearmanr(people.loc[finite,key],people.loc[finite,'mean_abs_error']).statistic)}
    # Within TUG bands, avoid interpreting unadjusted diagnosis differences as causal.
    stratified=[]
    for (band,diag),g in people.groupby(['TUG_band','diag']):
        if len(g)>=5:stratified.append({'TUG_band':band,'diag':diag,'people':len(g),'MAE':float(g.mean_abs_error.mean())})
    high_by_band=[]
    for band,g in people[people.high_error].groupby('TUG_band'):
        high_by_band.append({'TUG_band':band,'people':len(g),'MAE':float(g.mean_abs_error.mean()),'signed_error':float(g.mean_signed_error.mean()),'mean_speed_median':float(g.mean_speed.median())})
    result={'people':298,'metric':'mean person absolute held-out error in seconds; no seed ensemble or best-seed picking','models':summary,'paired_comparisons':comparisons,'high_error_definition':'60 people with highest mean absolute full-RF error over 15 repeats','high_error_by_TUG':high_by_band,'error_groups':group_summaries,'error_correlations_descriptive':correlations,'diagnosis_within_TUG_band_descriptive_min5':stratified,'scope':'all repetitions reuse one development cohort; bootstrap conditions on fitted predictions; no independent external or clinical validation; small subgroups are descriptive'}
    save('summary.json',clean(result))
    print(json.dumps({'models':summary,'paired_comparisons':comparisons},indent=2))

if __name__=='__main__':main()
