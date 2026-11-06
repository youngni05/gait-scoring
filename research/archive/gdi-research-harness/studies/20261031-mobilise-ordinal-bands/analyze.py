"""Four-class probability, ordinal-distance and low-function error analyses."""
from study import *

def load():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==300
    assert json.loads((HERE/'regression-completion.json').read_text())['tasks']==60
    paths=sorted((STORE/'folds').glob('*.csv.gz'));assert len(paths)==300
    p=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
    r=pd.concat([pd.read_csv(f) for f in sorted((STORE/'regression').glob('*.csv.gz'))],ignore_index=True)
    keys=['target','band','k','fold']+KEY
    r=r.drop(columns=['seed','cohort']).rename(columns={'actual':'reg_actual','class':'reg_class'})
    p=p.merge(r,on=keys,validate='many_to_one');np.testing.assert_array_equal(p.actual,p.reg_actual);np.testing.assert_array_equal(p['class'],p.reg_class)
    assert json.loads((HERE/'regression-five-completion.json').read_text())['tasks']==300
    paths=sorted((STORE/'regression-five').glob('*.csv.gz'));assert len(paths)==300
    q=pd.concat([pd.read_csv(f) for f in paths],ignore_index=True)
    keep=keys+['seed']+[c for c in q if c.startswith('RFreg5_')]
    p=p.merge(q[keep],on=keys+['seed'],validate='one_to_one')
    assert not p.duplicated(keys+['seed']).any()
    return p
def avg(a,ids):return float(np.average(a,weights=weights(ids))) if len(a) else None
def class_metrics(y,pred,ids,low):
    distance=abs(pred-y);pos=y==low;neg=~pos
    return {'accuracy':avg(pred==y,ids),'balanced_recall':float(np.mean([avg(pred[y==j]==j,ids[y==j]) for j in range(4)])),
            'band_mae':avg(distance,ids),'opposite_extreme':avg(distance==3,ids),'two_or_more_bands':avg(distance>=2,ids),
            'low_miss':avg(pred[pos]!=low,ids[pos]),'low_to_best':avg(distance[pos]==3,ids[pos]),'low_two_or_more':avg(distance[pos]>=2,ids[pos]),
            'argmax_fpr':avg(pred[neg]==low,ids[neg])}
def probability_metrics(y,p,ids):
    truth=np.eye(4)[y];e=(p-truth)**2
    brier=e.sum(1);rps=((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1)
    nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
    return {'brier':avg(brier,ids),'rps':avg(rps,ids),'log_loss':avg(nll,ids)}, {'brier':brier,'rps':rps,'log_loss':nll}
def operating(y,decision,ids,low):
    pos=y==low;neg=~pos;w=weights(ids);warn=decision==1
    return {'sensitivity':avg(decision[pos]==1,ids[pos]),'miss':avg(decision[pos]==0,ids[pos]),'fpr':avg(decision[neg]==1,ids[neg]),
            'warning_rate':avg(warn,ids),'false_discovery':float(np.sum(w[warn]*(y[warn]!=low))/w[warn].sum()) if warn.any() else None}
def calibration_bins(y,p,ids):
    w=weights(ids);rows=[];classwise=[]
    for j in range(4):
        ece=0.;b=np.minimum((p[:,j]*10).astype(int),9)
        for i in range(10):
            ix=b==i
            if not ix.any():continue
            mass=float(w[ix].sum()/w.sum());pp=float(np.average(p[ix,j],weights=w[ix]));obs=float(np.average(y[ix]==j,weights=w[ix]));ece+=mass*abs(pp-obs)
            rows.append({'class':j,'bin':i,'visits':int(ix.sum()),'mass':mass,'predicted':pp,'observed':obs})
        classwise.append({'class':j,'brier':avg((p[:,j]-(y==j))**2,ids),'ece10':ece,'prevalence':avg(y==j,ids)})
    return rows,classwise

def run():
    p=load();records=[];ops=[];conf=[];cb=[];classscores=[];perperson=[]
    for (target,band,k,seed),d in p.groupby(['target','band','k','seed'],sort=True):
        info={'target':target,'band':band,'k':int(k),'seed':int(seed)};y=d['class'].to_numpy();ids=d.participantid.to_numpy();low=lowclass(target)
        methods=['Prior']+[n+'_'+v for n in ['SVRreg','RFreg','RFreg5','RF','OrdinalRF'] for v in ['raw','cal']]
        for method in methods:
            a=d[[f'{method}_p{j}' for j in range(4)]].to_numpy();pred=a.argmax(1)
            assert np.isfinite(a).all() and (a>=0).all() and (a<=1+1e-12).all();np.testing.assert_allclose(a.sum(1),1,atol=1e-12)
            pm,loss=probability_metrics(y,a,ids);cm=class_metrics(y,pred,ids,low)
            records.append(dict(info,method=method,**pm,**cm))
            bins,cls=calibration_bins(y,a,ids);cb.extend(dict(info,method=method,**b) for b in bins);classscores.extend(dict(info,method=method,**c) for c in cls)
            for j in range(4):
                ix=y==j
                for h in range(4):conf.append(dict(info,method=method,actual_class=j,predicted_class=h,rate=avg(pred[ix]==h,ids[ix]),people=len(np.unique(ids[ix]))))
            if method!='Prior':
                for policy in ['p50','fpr10','sens80']:
                    dec=d[f'{method}_{policy}'].to_numpy();ops.append(dict(info,method=method,policy=policy,**operating(y,dec,ids,low)))
            if band=='long' and k==5 and method!='Prior':
                vectors=dict(loss,opposite_extreme=(abs(pred-y)==3).astype(float),low_miss=np.where(y==low,(pred!=low).astype(float),np.nan),
                             low_to_best=np.where(y==low,(abs(pred-y)==3).astype(float),np.nan),
                             low_fpr10_miss=np.where(y==low,(d[f'{method}_fpr10'].to_numpy()==0).astype(float),np.nan),
                             low_fpr10_false_warning=np.where(y!=low,(d[f'{method}_fpr10'].to_numpy()==1).astype(float),np.nan))
                frame=pd.DataFrame(dict(id=ids,**vectors)).groupby('id').mean().reset_index()
                frame['target']=target;frame['seed']=seed;frame['method']=method;perperson.append(frame)
        for name in ['SVRreg','RFreg','RFreg5']:
            records.append(dict(info,method=name+'_point',**class_metrics(y,d[name+'_point'].to_numpy(),ids,low)))
    m=pd.DataFrame(records);o=pd.DataFrame(ops);c=pd.DataFrame(classscores)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);o.to_csv(HERE/'low-function-by-seed.csv',index=False);c.to_csv(HERE/'class-probabilities-by-seed.csv',index=False)
    pd.DataFrame(conf).to_csv(HERE/'confusion-by-seed.csv',index=False);pd.DataFrame(cb).to_csv(STORE/'calibration-bins.csv.gz',index=False)
    keys=['target','band','k','method'];m.groupby(keys).agg({col:['mean','min','max'] for col in m if col not in keys+['seed']}).to_csv(HERE/'summary-by-condition.csv')
    o.groupby(keys+['policy']).agg({col:['mean','min','max'] for col in o if col not in keys+['policy','seed']}).to_csv(HERE/'low-function-summary.csv')
    b=pd.concat(perperson,ignore_index=True);b.to_csv(STORE/'primary-person-losses.csv.gz',index=False)
    paired=[];rng=np.random.default_rng(20261031)
    for target in ['sppb','tug']:
        for variant in ['raw','cal']:
            for baseline in ['RF','RFreg','RFreg5','SVRreg']:
                method='OrdinalRF_'+variant;ref=baseline+'_'+variant
                sub=b[b.target.eq(target)&b.method.isin([method,ref])]
                for metric in ['brier','rps','opposite_extreme','low_miss','low_to_best','low_fpr10_miss','low_fpr10_false_warning']:
                    wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna();v=(wide[method]-wide[ref]).to_numpy()
                    samples=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
                    ss=sub.groupby(['seed','method'])[metric].mean().unstack();ds=ss[method]-ss[ref]
                    paired.append({'target':target,'variant':variant,'baseline':baseline,'metric':metric,'difference':float(v.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
                                   'lower_error_seeds':int((ds<0).sum()),'seed_differences':ds.to_dict(),'people':len(v)})
    save(HERE/'paired-bootstrap.json',{'scope':'conditional paired person bootstrap, 5 seeds averaged within person; no retraining/multiple-comparison correction','comparisons':paired})
    primary=m[m.band.eq('long')&m.k.eq(5)].groupby(['target','method']).mean(numeric_only=True).reset_index()
    save(HERE/'summary.json',{'primary':'long5, draw0, all five seeds','metrics_rows':len(m),'primary_metrics':primary.astype(object).where(pd.notna(primary),None).to_dict('records'),
                            'data_paths':{'large':str(STORE),'small':str(HERE)},'analysis_sha256':sha(__file__)})
    print(primary[['target','method','brier','rps','accuracy','opposite_extreme','low_miss','low_to_best']].to_string(index=False));print('ANALYSIS_DONE',flush=True)

if __name__=='__main__':run()
