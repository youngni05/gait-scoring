from study import *
METRICS=['MAE','underprediction','overprediction','bias']

def old_predictions(cohort, info):
    c='prep_event' if cohort=='prep' else 'geometry'
    names={'PrepAll':'Full','TurnContext':'Context'} if cohort=='prep' else {'Walk3Turn':'Full','WalkMatchedSpeed':'MatchedWalk'}
    p=pd.read_csv(PRIOR/c/'predictions.csv');p=p[(p.family=='Ridge') & p.view.isin(names)]
    q=pd.read_csv(HERE.parent/info['source']/'predictions.csv');q=q[q.method.isin(names)].rename(columns={'method':'view'});q['family']='RF'
    p=pd.concat([p,q],ignore_index=True);p['view']=p.view.map(names)
    p['scenario']='Original';e=p[p.id!=REMOVED].copy();e['scenario']='EvalExcluded'
    return pd.concat([p,e],ignore_index=True)

def aggregate(p):
    p=p.copy();err=p.prediction-p.target
    p['MAE']=abs(err);p['underprediction']=np.maximum(-err,0);p['overprediction']=np.maximum(err,0);p['bias']=err
    keys=['scenario','family','view']
    agg={m:(m,'mean') for m in METRICS};agg.update(target=('target','first'),prediction=('prediction','mean'))
    part=p.groupby(keys+['repeat','id']).agg(**agg).reset_index()
    persons=part.groupby(keys+['id']).agg(**agg).reset_index()
    runs=[]
    for key,g in part.groupby(keys+['repeat']):
        for band,mask in [('all',np.ones(len(g),bool)),('ge20',g.target>=20),('lt20',g.target<20)]:
            z=g.loc[mask];runs.append(dict(zip(keys+['repeat'],key))|{'band':band,'people':len(z)}|{m:float(z[m].mean()) for m in METRICS})
    runs=pd.DataFrame(runs); summary=[]
    for key,g in runs.groupby(keys+['band']):
        summary.append(dict(zip(keys+['band'],key))|{'people':int(g.people.iloc[0])}|{m:float(g[m].mean()) for m in METRICS}|{'MAE_min':float(g.MAE.min()),'MAE_max':float(g.MAE.max())})
    return part,persons,runs,summary

def contrast(persons,runs,a,b,band):
    def select(frame,key):
        scenario,family,view=key
        return frame[(frame.scenario==scenario)&(frame.family==family)&(frame.view==view)]
    pa=select(persons,a).set_index('id').sort_index();pb=select(persons,b).set_index('id')
    pa=pa[pa.target>=20] if band=='ge20' else (pa[pa.target<20] if band=='lt20' else pa)
    pb=pb.loc[pa.index];assert np.array_equal(pa.target,pb.target)
    ra=select(runs,a);rb=select(runs,b);ra=ra[ra.band==band].set_index('repeat');rb=rb[rb.band==band].set_index('repeat').loc[ra.index]
    ix=np.random.default_rng(20261031).integers(len(pa),size=(10000,len(pa)))
    result={'a':a,'b':b,'band':band,'people':len(pa),'ids':pa.index.tolist()}
    for m in METRICS:
        delta=(pa[m]-pb[m]).to_numpy();boot=delta[ix].mean(1)
        result[m]={'difference':float(delta.mean()),'conditional_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
            'people_better':int(sum(delta<0)),'people_worse':int(sum(delta>0)),'better_partitions':int(sum(ra[m]<rb[m])),
            'person_deltas':dict(zip(pa.index,delta.tolist())),'partition_deltas':(ra[m]-rb[m]).tolist(),
            'leave_one_person_out_range':[float(min((sum(delta)-delta)/(len(delta)-1))),float(max((sum(delta)-delta)/(len(delta)-1)))] if len(delta)>1 else None}
    return result

def main():
    protocol=json.loads((HERE/'protocol.json').read_text());all_summary={}
    for cohort,info in protocol['cohorts'].items():
        out=HERE/cohort;assert json.loads((out/'completion.json').read_text())['complete']
        p=pd.read_csv(out/'predictions.csv');p['scenario']='TrainExcluded';p=pd.concat([p,old_predictions(cohort,info)],ignore_index=True)
        part,persons,runs,summary=aggregate(p)
        part.to_csv(out/'partition-person.csv',index=False);persons.to_csv(out/'person-results.csv',index=False)
        persons[persons.target>=20].to_csv(out/'long-people.csv',index=False);runs.to_csv(out/'partition-metrics.csv',index=False)
        pairs=[];control='Context' if cohort=='prep' else 'MatchedWalk'
        for scenario in ['Original','EvalExcluded','TrainExcluded']:
            for view in ['Full',control]:pairs.append(((scenario,'Ridge',view),(scenario,'RF',view)))
            pairs.append(((scenario,'Ridge','Full'),(scenario,'Ridge',control)))
        for view in info['ridge_views']:
            if view!='Full': pairs.append((('TrainExcluded','Ridge',view),('TrainExcluded','Ridge','Full')))
        for family in ['Ridge','RF']:
            for view in ['Full',control]:pairs.append((('TrainExcluded',family,view),('EvalExcluded',family,view)))
        comparisons=[contrast(persons,runs,a,b,band) for a,b in pairs for band in ['all','ge20','lt20']]
        all_summary[cohort]={'metrics':summary,'contrasts':comparisons}
        save(cohort+'/summary.json',all_summary[cohort]);pd.DataFrame(summary).to_csv(out/'metrics.csv',index=False)
        print(cohort,pd.DataFrame(summary).query("band=='ge20'")[['scenario','family','view','MAE','underprediction']].to_string(index=False),flush=True)
    save('summary.json',all_summary)
    # Descriptive feature values: source measurements, not learned importance.
    names=['walk_speed_mean','walk_speed_std','walk_speed_change','turn_seconds','turn_speed_mean','turn_speed_std','turn_speed_change','turn_path_length','turn_lateral_range','turn_direction_deg','turn_direction_speed_mean','turn_direction_speed_p95','turn_slow_fraction']
    phase=['seconds','height_change','height_range','vertical_velocity_mean','vertical_velocity_std','vertical_speed_p95','vertical_speed_change','stationary_fraction']
    names += [f'{prefix}_{name}' for prefix in ['prep_rise','sit'] for name in phase]
    d=np.load(HERE/'prep/input.npz');m=json.loads((HERE/'prep/manifest.json').read_text());rows=[]
    low=d['Full'][d['y']<20]
    for i in np.flatnonzero(d['y']>=20):
        for j,name in enumerate(names):rows.append({'id':str(d['ids'][i]),'target':float(d['y'][i]),'feature':name,'value':float(d['Full'][i,j]),'other97_median':float(np.median(low[:,j])),'other97_q25':float(np.quantile(low[:,j],.25)),'other97_q75':float(np.quantile(low[:,j],.75))})
    pd.DataFrame(rows).to_csv(HERE/'feature-values.csv',index=False)
    save('long-phase-times.json',[{'id':r['id'],'target':r['target'],'turn_seconds':r['turn_observed_seconds'],
        'preparation_seconds':r['core_rise'][0]-r['phase_windows']['rise'][0],
        'core_rise_seconds':r['core_rise'][1]-r['core_rise'][0],
        'sit_seconds':r['phase_windows']['sit'][1]-r['phase_windows']['sit'][0],
        'observed_seconds':r['observed_seconds']} for r in m if r['target']>=20])

if __name__=='__main__':main()
