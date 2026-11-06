"""Matched component and total-score comparisons on person-held-out predictions."""
from train import HERE, SEEDS, LABELS, COMPONENTS, ref, save
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ENDPOINTS = {
    'total': ('total_sppb_score', {'original4':'A', 'direct6':'B', 'RF_direct':'RF_total_direct', 'RF_components':'RF_total', 'DS_components':'DS_total', 'constant':'constant_total', 'gait_lookup':'lookup_total_direct_remaining', 'DS_bounded':'DS_bounded', 'direct6_bounded':'B_bounded'}),
    'gait': (LABELS[0], {'RF':'RF_gait', 'DS':'DS_gait', 'constant':'constant_gait'}),
    'balance': (LABELS[1], {'RF':'RF_balance', 'DS':'DS_balance', 'constant':'constant_balance', 'gait_lookup':'lookup_balance'}),
    'chair': (LABELS[2], {'RF':'RF_chair', 'DS':'DS_chair', 'constant':'constant_chair', 'gait_lookup':'lookup_chair'}),
    'remaining': ('actual_remaining', {'RF':'RF_remaining', 'DS':'DS_remaining', 'constant':'constant_remaining', 'gait_lookup':'lookup_remaining'}),
}
METRICS = ['mae', 'rmse', 'p90', 'bias']
CELL = ['band','k','draw','seed']

def metric(d, actual, prediction):
    d=d[d[actual].notna() & d[prediction].notna()]
    if len(d)==0: return None
    e=(d[prediction]-d[actual]).to_numpy(); w=ref.weights(d.participantid.to_numpy())
    order=np.argsort(abs(e),kind='stable')
    return {'people':int(d.participantid.nunique()),'visits':len(d),'mae':float(np.dot(w,abs(e))),'rmse':float(np.sqrt(np.dot(w,e*e))),'p90':float(abs(e)[order[min(np.searchsorted(np.cumsum(w[order]),.9),len(e)-1)]]),'bias':float(np.dot(w,e))}

def groups(d):
    yield 'all',d
    yield 'low_sppb',d[d.total_sppb_score.le(3)]
    yield 'other_sppb',d[d.total_sppb_score.gt(3)]
    yield 'known_gait',d[d[LABELS[0]].notna()]
    for score in range(5): yield f'gait_{score}',d[d[LABELS[0]].eq(score)]
    for cohort,g in d.groupby('cohort'): yield 'cohort_'+cohort,g

def augment(p):
    p['actual_remaining']=p[LABELS[1:]].sum(axis=1,min_count=2)
    p['RF_remaining']=p.RF_balance+p.RF_chair
    p['DS_remaining']=p.DS_balance+p.DS_chair
    p['DS_bounded']=p[['DS_'+c for c in COMPONENTS]].clip(0,4).sum(axis=1)
    p['B_bounded']=p.B.clip(0,12)
    return p

def baseline_only():
    labels=pd.read_csv(HERE/'labels.csv'); b=pd.read_csv(HERE/'baselines.csv')
    p=labels.merge(b,on=['participantid','visit_number','fold'],validate='one_to_one')
    p['actual_remaining']=p[LABELS[1:]].sum(axis=1,min_count=2)
    rows=[]
    for endpoint,label in [('balance',LABELS[1]),('chair',LABELS[2]),('remaining','actual_remaining')]:
        for group,d in groups(p):
            # Match both comparators exactly, including the five missing gait inputs.
            d=d[d[label].notna() & d['lookup_'+endpoint].notna()]
            for model in ['constant','lookup']:
                m=metric(d,label,model+'_'+endpoint)
                if m: rows.append({'endpoint':endpoint,'group':group,'model':model,**m})
    m=pd.DataFrame(rows);m.to_csv(HERE/'baseline-metrics.csv',index=False)
    profiles=[]
    for score,d in p[p[LABELS[0]].notna()].groupby(LABELS[0]):
        for endpoint,label in [('balance',LABELS[1]),('chair',LABELS[2])]:
            d0=d[d[label].notna()];w=ref.weights(d0.participantid.to_numpy());y=d0[label].to_numpy()
            profiles.append({'gait_score':int(score),'endpoint':endpoint,'people':int(d0.participantid.nunique()),'visits':len(d0),'weighted_mean':float(np.dot(w,y)),'weighted_sd':float(np.sqrt(np.dot(w,(y-np.dot(w,y))**2))),**{f'fraction_score_{j}':float(w[y==j].sum()) for j in range(5)}})
    pd.DataFrame(profiles).to_csv(HERE/'gait-score-profiles.csv',index=False)
    print(m[m.group.eq('all')][['endpoint','model','people','visits','mae','p90']].round(6).to_string(index=False),flush=True)

def contrasts(endpoint):
    if endpoint=='total':
        return [('DS_components','direct6'),('RF_components','direct6'),('RF_direct','direct6'),('RF_components','RF_direct'),('DS_components','original4'),('DS_bounded','direct6_bounded')]
    pairs=[('RF','constant'),('DS','constant'),('DS','RF')]
    if endpoint!='gait':pairs += [('RF','gait_lookup'),('DS','gait_lookup'),('gait_lookup','constant')]
    return pairs

def rank_concordance(d, actual, prediction):
    # One average per distinct person within a gait-score stratum and outer fold.
    # Compare only different people in the SAME held-out fold; lookup is then
    # constant, so its concordance is exactly .5 on every untied true-score pair.
    d=d[d[actual].notna() & d[prediction].notna()]
    people=d.groupby(['fold','participantid'])[[actual,prediction]].mean().reset_index()
    score=0.; pairs=0
    for _,g in people.groupby('fold'):
        y=g[actual].to_numpy(); pred=g[prediction].to_numpy()
        a,b=np.triu_indices(len(g),1); delta=y[a]-y[b]; keep=abs(delta)>1e-12
        delta=delta[keep]; pdiff=(pred[a]-pred[b])[keep]
        score+=float(((delta*pdiff)>0).sum()+.5*(pdiff==0).sum());pairs+=len(delta)
    return {'people':len(people),'pairs':pairs,'concordance':score/pairs if pairs else None}

def run():
    baseline_only()
    p=augment(pd.read_csv(HERE/'predictions.csv.gz'))
    rows=[];paired=[];rank=[]
    for key,cell in p.groupby(CELL):
        meta=dict(zip(CELL,key))
        for group,d in groups(cell):
            for endpoint,(actual,models) in ENDPOINTS.items():
                for model,column in models.items():
                    m=metric(d,actual,column)
                    if m: rows.append(meta|{'group':group,'endpoint':endpoint,'model':model}|m)
                for model,reference in contrasts(endpoint):
                    columns=[actual,models[model],models[reference]]
                    matched=d[d[columns].notna().all(axis=1)]
                    a=metric(matched,actual,models[model]);b=metric(matched,actual,models[reference])
                    if a: paired.append(meta|{'group':group,'endpoint':endpoint,'model':model,'reference':reference,'people':a['people'],'visits':a['visits']}|{name+'_change':a[name]-b[name] for name in METRICS})
        if key[2]==0:
            for score in range(5):
                d=cell[cell[LABELS[0]].eq(score)]
                for endpoint in ['balance','chair','remaining']:
                    actual,models=ENDPOINTS[endpoint]
                    for model in ['RF','DS']:
                        rank.append(meta|{'gait_score':score,'endpoint':endpoint,'model':model}|rank_concordance(d,actual,models[model]))
    m=pd.DataFrame(rows);m.to_csv(HERE/'seed-metrics.csv',index=False)
    e=pd.DataFrame(paired);e.to_csv(HERE/'paired-seed-metrics.csv',index=False)
    r=pd.DataFrame(rank);r.to_csv(HERE/'within-gait-ranking.csv',index=False)
    keys=['band','k','draw','group','endpoint','model']
    agg=m.groupby(keys).agg(people=('people','first'),visits=('visits','first'),**{name+'_'+kind:(name,kind) for name in METRICS for kind in ['mean','min','max','std']}).reset_index()
    agg.to_csv(HERE/'aggregate-metrics.csv',index=False)
    pe=e.groupby(keys+['reference']).agg(people=('people','first'),visits=('visits','first'),**{name+'_change_mean':(name+'_change','mean') for name in METRICS},improved_seeds=('mae_change',lambda x:int((x<0).sum())),nonworse_seeds=('mae_change',lambda x:int((x<=0).sum()))).reset_index()
    pe.to_csv(HERE/'paired-aggregate.csv',index=False)
    ranks=r.groupby(['band','k','gait_score','endpoint','model']).agg(people=('people','first'),pairs=('pairs','first'),concordance_mean=('concordance','mean'),concordance_min=('concordance','min'),concordance_max=('concordance','max')).reset_index()
    ranks.to_csv(HERE/'within-gait-ranking-aggregate.csv',index=False)
    # All-seed joint criteria, without noninferiority margins or best-seed selection.
    total=e[e.endpoint.eq('total')]
    j=total[total.group.eq('all')].merge(total[total.group.eq('low_sppb')],on=CELL+['endpoint','model','reference'],suffixes=('_all','_low'),validate='one_to_one')
    j['both']=j.mae_change_all.le(0)&j.mae_change_low.lt(0)
    j['all_three']=j.both&j.p90_change_low.lt(0)
    joint=j.groupby(['band','k','draw','model','reference']).agg(overall_change=('mae_change_all','mean'),low_change=('mae_change_low','mean'),low_p90_change=('p90_change_low','mean'),overall_nonworse_seeds=('mae_change_all',lambda x:int((x<=0).sum())),joint_seeds=('both','sum'),joint_p90_seeds=('all_three','sum')).reset_index()
    joint.to_csv(HERE/'joint-criteria.csv',index=False)
    # Paired person bootstrap: seed-averaged ERRORS, never seed-averaged predictions.
    intervals=[];rng=np.random.default_rng(20261031)
    for (band,k),cell in p[p.draw.eq(0)].groupby(['band','k']):
        for group,d in groups(cell):
            if group not in ['all','low_sppb','known_gait'] and not group.startswith('gait_'):continue
            for endpoint,(actual,models) in ENDPOINTS.items():
                chosen=[('DS_components','direct6'),('RF_components','direct6'),('RF_components','RF_direct')] if endpoint=='total' else ([('RF','gait_lookup'),('DS','gait_lookup')] if endpoint!='gait' else [])
                if endpoint=='total' and group not in ['all','low_sppb']:continue
                if endpoint!='total' and group=='all':continue
                for model,reference in chosen:
                    d0=d[d[[actual,models[model],models[reference]]].notna().all(axis=1)].copy()
                    if not len(d0):continue
                    d0['delta']=abs(d0[models[model]]-d0[actual])-abs(d0[models[reference]]-d0[actual])
                    delta=d0.groupby('participantid').delta.mean().to_numpy()
                    boot=rng.integers(0,len(delta),size=(2000,len(delta)))
                    lo,hi=np.quantile(delta[boot].mean(axis=1),[.025,.975])
                    intervals.append({'band':band,'k':int(k),'group':group,'endpoint':endpoint,'model':model,'reference':reference,'people':len(delta),'mae_change':float(delta.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    save('paired-bootstrap.json',intervals)
    ledgers=[json.loads(path.read_text()) for path in (HERE/'folds').glob('*-DS.json')]
    selections=pd.DataFrame([{c:d[c] for c in ['band','k','fold','seed','width','epochs','parameters']} for d in ledgers]);selections.to_csv(HERE/'DS-selections.csv',index=False)
    out_of_range={c:int(((p['DS_'+c]<0)|(p['DS_'+c]>4)).sum()) for c in COMPONENTS}
    save('summary.json',{'primary':'raw predictions, draw0, equal-person metrics; five-seed mean, not ensemble','component_labels':dict(zip(COMPONENTS,LABELS)),'score_lookup_limit':'uses measured clinical gait score; diagnostic comparator, not a gait-video-only model','raw_DS_out_of_range_prediction_rows':out_of_range,'epochs_max':int(selections.epochs.max()),'epochs_at_600':int(selections.epochs.eq(600).sum()),'total_primary':agg[agg.draw.eq(0)&agg.endpoint.eq('total')&agg.group.isin(['all','low_sppb'])].to_dict(orient='records'),'joint_primary':joint[joint.draw.eq(0)].to_dict(orient='records'),'uncertainty':'bootstrap conditions on fixed exploratory OOF fits; repeated seeds and draws do not add independent people','artifacts':{name:ref.sha(HERE/name) for name in ['seed-metrics.csv','paired-seed-metrics.csv','aggregate-metrics.csv','paired-aggregate.csv','joint-criteria.csv','within-gait-ranking.csv','paired-bootstrap.json']}})
    print(joint[joint.draw.eq(0)&joint.reference.eq('direct6')].round(6).to_string(index=False),flush=True)
    print('ANALYSIS_FINISHED',flush=True)

if __name__=='__main__':
    import sys
    with threadpool_limits(limits=1):
        if '--baseline' in sys.argv: baseline_only()
        else: run()
