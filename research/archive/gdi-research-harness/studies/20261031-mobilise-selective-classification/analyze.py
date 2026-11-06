"""Report every seed and condition; keep missed, falsely warned and deferred separate."""
from study import *
from sklearn.metrics import roc_auc_score,average_precision_score

def decisions(y,d,ids):
    pos=y==1;neg=~pos;answered=d>=0
    result={'visits':len(y),'people':len(np.unique(ids)),'positive_people':len(np.unique(ids[pos])),
            'positive_visits':int(pos.sum()),'warning_visits':int((d==1).sum()),'deferred_visits':int((d==-1).sum()),
            'positive_people_any_missed_visit':len(np.unique(ids[pos&(d==0)])),
            'positive_people_any_detected_visit':len(np.unique(ids[pos&(d==1)])),
            'positive_people_any_deferred_visit':len(np.unique(ids[pos&(d==-1)])),
            'negative_people_any_false_warning':len(np.unique(ids[neg&(d==1)])),
            'warning_rate':mean(d==1,ids),'defer_rate':mean(d==-1,ids),
            'miss_rate':mean(d[pos]==0,ids[pos]),'detect_rate':mean(d[pos]==1,ids[pos]),'positive_defer_rate':mean(d[pos]==-1,ids[pos]),
            'fpr':mean(d[neg]==1,ids[neg]),'negative_defer_rate':mean(d[neg]==-1,ids[neg]),
            'miss_among_answered_positive':mean(d[pos&answered]==0,ids[pos&answered]),
            'fpr_among_answered_negative':mean(d[neg&answered]==1,ids[neg&answered]),
            'answered_error':mean(d[answered]!=y[answered],ids[answered])}
    w=weights(ids);warn=d==1
    result['false_discovery_rate']=float(np.sum(w[warn]*(1-y[warn]))/w[warn].sum()) if warn.any() else None
    assert abs(result['miss_rate']+result['detect_rate']+result['positive_defer_rate']-1)<1e-12
    return result

def probabilities(y,p,ids):
    w=weights(ids);bins=np.minimum((p*10).astype(int),9);ece=0.;cal=[]
    for b in range(10):
        ix=bins==b
        if not ix.any():continue
        pred=float(np.average(p[ix],weights=w[ix]));actual=float(np.average(y[ix],weights=w[ix]));mass=float(w[ix].sum()/w.sum())
        ece+=mass*abs(pred-actual);cal.append({'bin':b,'visits':int(ix.sum()),'mass':mass,'predicted':pred,'observed':actual})
    return {'brier':float(np.average((p-y)**2,weights=w)),'log_loss':float(np.average(-y*np.log(np.clip(p,1e-15,1))-(1-y)*np.log(np.clip(1-p,1e-15,1)),weights=w)),
            'roc_auc':float(roc_auc_score(y,p,sample_weight=w)),'average_precision':float(average_precision_score(y,p,sample_weight=w)),
            'ece10':ece,'prevalence':float(np.average(y,weights=w))},cal

def load():
    p=pd.read_csv(HERE/'predictions.csv.gz')
    old=pd.read_csv(BASE/'predictions.csv');old=old[old.draw.eq(0)]
    keys=['target','band','k']+KEY
    old=old[keys+['actual','baseline','Ridge','SVR','RF']].rename(columns={'actual':'old_actual','RF':'RFreg'})
    p=p.merge(old,on=keys,validate='many_to_one');np.testing.assert_array_equal(p.actual,p.old_actual)
    parts=[]
    for c in pd.read_csv(STABILITY/'predictions.csv',chunksize=200000):
        parts.append(c[c.draw.eq(0)&c.budget.eq(600)][keys+['seed','actual','prediction']])
    ds=pd.concat(parts).rename(columns={'actual':'ds_actual','prediction':'DeepSets600'})
    p=p.merge(ds,on=keys+['seed'],validate='one_to_one');np.testing.assert_array_equal(p.actual,p.ds_actual)
    reg=pd.read_csv(HERE/'regression/predictions.csv.gz')
    cols=[c for c in reg if c.startswith('SVRreg_') or c.startswith('RFreg_')]
    p=p.merge(reg[keys+cols],on=keys,validate='many_to_one')
    np.testing.assert_allclose(p.SVR,p.SVRreg_prediction,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(p.RFreg,p.RFreg_prediction,rtol=1e-12,atol=1e-12)
    return p

def run():
    p=load();metrics=[];pm=[];calbins=[];boot=[]
    for (target,band,k,seed),d in p.groupby(['target','band','k','seed'],sort=True):
        info={'target':target,'band':band,'k':int(k),'seed':int(seed)};y=d.positive.to_numpy();ids=d.participantid.to_numpy()
        ds={}
        for name in ['baseline','Ridge','SVR','RFreg','DeepSets600']:ds[name+'_original']=label(d[name].to_numpy(),target)
        for col in d:
            if '_r' in col and col.rsplit('_r',1)[-1] in ['0','10','20']:ds[col]=d[col].to_numpy()
        for name,dec in ds.items():metrics.append(dict(info,method=name,**decisions(y,dec,ids)))
        for name in ['prior','LR_raw','LR_cal','RF_raw','RF_cal','SVRreg_cal','RFreg_cal']:
            a,b=probabilities(y,d[name].to_numpy(),ids);pm.append(dict(info,method=name,**a))
            calbins.extend(dict(info,method=name,**r) for r in b)
        if band=='long' and k==5:
            for name in ['SVRreg_cal_fpr10_r0','RF_cal_fpr10_r0','SVRreg_cal_sens80_r0','RF_cal_sens80_r0','RF_cal_sens80_r20','RF_cal_fpr10_r20']:
                dec=ds[name]
                for true in [0,1]:
                    ix=y==true;df=pd.DataFrame({'id':ids[ix],'warn':dec[ix]==1,'miss':dec[ix]==0,'defer':dec[ix]==-1}).groupby('id').mean().reset_index()
                    df['target']=target;df['seed']=seed;df['method']=name;df['true']=true;boot.append(df)
    m=pd.DataFrame(metrics);q=pd.DataFrame(pm)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);q.to_csv(HERE/'probability-by-seed.csv',index=False)
    pd.DataFrame(calbins).to_csv(HERE/'calibration-bins.csv',index=False)
    keys=['target','band','k','method']
    m.groupby(keys)[['detect_rate','miss_rate','fpr','defer_rate','positive_defer_rate','negative_defer_rate','false_discovery_rate','answered_error']].agg(['mean','min','max']).to_csv(HERE/'decision-summary.csv')
    q.groupby(keys)[['brier','roc_auc','average_precision','ece10']].agg(['mean','min','max']).to_csv(HERE/'probability-summary.csv')
    b=pd.concat(boot);b.to_csv(HERE/'primary-person-rates.csv.gz',index=False)
    ci=[];rng=np.random.default_rng(20261031)
    for target in ['sppb','tug']:
        for policy in ['fpr10','sens80']:
            a='RF_cal_'+policy+'_r0';ref='SVRreg_cal_'+policy+'_r0'
            for true,metric in [(1,'warn'),(0,'warn')]:
                sub=b[(b.target==target)&(b.true==true)&b.method.isin([a,ref])]
                wide=sub.groupby(['id','method'])[metric].mean().unstack();v=(wide[a]-wide[ref]).to_numpy();n=len(v)
                means=np.array([rng.choice(v,n,replace=True).mean() for _ in range(2000)])
                seedwide=sub.groupby(['seed','method'])[metric].mean().unstack();delta=seedwide[a]-seedwide[ref]
                ci.append({'target':target,'policy':policy,'comparison':a+' minus '+ref,'metric':'sensitivity' if true else 'FPR','people':n,
                           'difference':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'seed_differences':delta.to_dict()})
    save(HERE/'paired-bootstrap.json',{'scope':'conditional paired participant bootstrap of fixed OOF predictions; seeds averaged within person; no refitting or multiplicity adjustment','comparisons':ci})
    primary=m[m.band.eq('long')&m.k.eq(5)].groupby(['target','method']).mean(numeric_only=True)
    wanted=['SVR_original','RFreg_original','DeepSets600_original','LR_cal_fpr10_r0','SVRreg_cal_fpr10_r0','RFreg_cal_fpr10_r0','RF_cal_fpr10_r0','SVRreg_cal_sens80_r0','RF_cal_sens80_r0','RF_cal_sens80_r10','RF_cal_sens80_r20']
    compact=primary.loc[primary.index.get_level_values('method').isin(wanted),['detect_rate','miss_rate','fpr','defer_rate','positive_defer_rate','negative_defer_rate','false_discovery_rate']].reset_index()
    save(HERE/'summary.json',{'primary':'long five bouts, selected before new results using prior experiment convention','decision_rows':len(m),'probability_rows':len(q),'primary_metrics':compact.astype(object).where(pd.notna(compact),None).to_dict('records'),
                             'seed_notes':'RF classifiers use five seeds. LR and saved original regressors deterministic/reused, not five independent training runs. DeepSets600 uses five stored seeds.',
                             'scope':'no clinical cutoff validation, no video validation, no OOD claim; previous exploratory reuse of outer evaluation subjects'})
    print(compact.to_string(index=False));print('ANALYSIS_DONE',flush=True)

if __name__=='__main__':run()
