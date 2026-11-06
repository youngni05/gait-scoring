"""Stream person-held-out predictions and compare boundary/decision changes."""
from run import *

def mean(a,ids):return float(np.average(a,weights=weights(ids))) if len(a) else np.nan
def cells():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==300
    for target,band,n,seed in itertools.product(['sppb','tug'],['short','long'],[1,3,5],SEEDS):
        frames=[pd.read_csv(STORE/'folds'/f'{target}-{band}-{n}-f{fold}-s{seed}.csv.gz') for fold in range(5)]
        d=pd.concat(frames,ignore_index=True);assert not d.duplicated(KEY+['family','scheme']).any()
        yield {'target':target,'band':band,'n':n,'seed':seed},d
def probability(y,p,prior,ids):
    truth=np.eye(p.shape[1])[y];b=((p-truth)**2).sum(1);base=((prior-truth)**2).sum(1)
    v=mean(b,ids);bv=mean(base,ids)
    return {'brier':v,'prior_brier':bv,'brier_skill':1-v/bv,'rps':mean(((p.cumsum(1)[:,:-1]-truth.cumsum(1)[:,:-1])**2).mean(1),ids)},b
def run():
    metrics=[];probs=[];policies=[];conf=[];losses=[]
    for info,d in cells():
        for (family,scheme),g in d.groupby(['family','scheme']):
            k=int(g.classes.iloc[0]);assert g.classes.eq(k).all();target=info['target'];ids=g.participantid.to_numpy();y=g['class'].to_numpy();lc=low(target,k);pos=y==lc;neg=~pos
            assert len(g)==(7356 if target=='sppb' else 5688) and len(np.unique(ids))==(2174 if target=='sppb' else 1939)
            p=g[[f'p{j}' for j in range(k)]].to_numpy();prior=g[[f'prior{j}' for j in range(k)]].to_numpy();pr,_=probability(y,p,prior,ids)
            base=dict(info,family=family,scheme=scheme,classes=k);lowb=(p[:,lc]-pos)**2;probs.append(dict(base,**pr,low_brier=mean(lowb,ids)))
            widths=np.array([np.diff(json.loads(c)).tolist() for c in g.cuts])
            for rule in ['mode','selected','prior_class']:
                pred=g[rule].to_numpy();st=stats(y,pred,ids,k,target);recalls=st.pop('recalls');mid=(pred>0)&(pred<k-1)
                wm=mean(widths[np.flatnonzero(mid),pred[mid]-1],ids[mid]) if mid.any() else np.nan
                truthmid=(y>0)&(y<k-1);wt=mean(widths[np.flatnonzero(truthmid),y[truthmid]-1],ids[truthmid])
                metrics.append(dict(base,decision=rule,**st,balanced_above_chance=st['balanced_accuracy']-1/k,
                                    prediction_middle_width=wm,true_middle_width=wt,**{f'recall{j}':recalls[j] for j in range(k)}))
                for j in range(k):
                    mask=y==j
                    for h in range(k):conf.append(dict(base,decision=rule,actual_class=j,predicted_class=h,rate=mean(pred[mask]==h,ids[mask]),people=len(np.unique(ids[mask]))))
                if info['band']=='long' and info['n']==5 and rule!='prior_class':
                    frame=pd.DataFrame({'id':ids,'error':(pred!=y).astype(float),'extreme':(abs(pred-y)==k-1).astype(float),
                                        'miss':np.where(pos,pred!=lc,np.nan),'fpr':np.where(neg,pred==lc,np.nan),
                                        'low_to_best':np.where(pos,abs(pred-y)==k-1,np.nan),'low_brier':lowb,
                                        'fpr10_miss':np.where(pos,g.fpr10.eq(0),np.nan),'fpr10_false_warning':np.where(neg,g.fpr10.eq(1),np.nan)})
                    frame=frame.groupby('id').mean().reset_index()
                    for key,val in dict(base,decision=rule).items():frame[key]=val
                    losses.append(frame)
            for policy in ['p50','fpr10','sens80']:
                dec=g[policy].to_numpy();policies.append(dict(base,policy=policy,miss=mean(dec[pos]==0,ids[pos]),fpr=mean(dec[neg]==1,ids[neg]),warning_rate=mean(dec==1,ids)))
    m=pd.DataFrame(metrics);p=pd.DataFrame(probs);o=pd.DataFrame(policies)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);p.to_csv(HERE/'probabilities-by-seed.csv',index=False);o.to_csv(HERE/'warnings-by-seed.csv',index=False)
    pd.DataFrame(conf).to_csv(HERE/'confusion-by-seed.csv',index=False)
    m.groupby(['target','band','n','family','scheme','decision']).mean(numeric_only=True).to_csv(HERE/'summary-by-condition.csv')
    b=pd.concat(losses,ignore_index=True);b.to_csv(STORE/'primary-person-losses.csv.gz',index=False)
    selections=[]
    for path in sorted((STORE/'folds').glob('*.json')):
        a=json.loads(path.read_text());target,band,n,fold,seed=a['task']
        for family,meta in a['families'].items():
            for scheme,sel in meta['selection'].items():
                c=meta['candidates'][sel['candidate']]
                if scheme=='adjusted4':
                    ref=meta['selection']['fixed4']['metrics'];options=[r for r in meta['rules'] if len(meta['candidates'][r['candidate']]['cuts'])==3]
                else:
                    ci=0 if scheme=='fixed4' else len(meta['candidates'])-1
                    ref=next(r['metrics'] for r in meta['rules'] if r['candidate']==ci and r['rule']=='mode');options=[r for r in meta['rules'] if r['candidate']==ci]
                selections.append({'target':target,'band':band,'n':n,'fold':fold,'seed':seed,'family':family,'scheme':scheme,'cuts':json.dumps(c['cuts']),
                                   'rule':sel['rule'],'eligible_candidates':sum(x['eligible'] for x in meta['candidates']),
                                   'guard_feasible_rules':sum(guards(r['metrics'],ref) for r in options),
                                   'higher_balanced_accuracy_rejected':sum(r['metrics']['balanced_accuracy']>ref['balanced_accuracy'] and not guards(r['metrics'],ref) for r in options),
                                   **{key:value for key,value in sel['metrics'].items() if key!='recalls'}})
    pd.DataFrame(selections).to_csv(HERE/'inner-selections.csv',index=False)
    rng=np.random.default_rng(20261101);paired=[]
    for target,family in itertools.product(['sppb','tug'],FAMILIES):
        sub=b[b.target.eq(target)&b.family.eq(family)]
        pairs=[('adjusted4','selected','fixed4','selected'),('merged3','selected','fixed4','selected')]+[(s,'selected',s,'mode') for s in ['fixed4','adjusted4','merged3']]
        for ns,nd,rs,rd in pairs:
            aa=sub[sub.scheme.eq(ns)&sub.decision.eq(nd)];rr=sub[sub.scheme.eq(rs)&sub.decision.eq(rd)]
            for metric in ['extreme','miss','fpr','low_to_best','low_brier','fpr10_miss','fpr10_false_warning']:
                x=aa.groupby('id')[metric].mean().to_frame('new').join(rr.groupby('id')[metric].mean().rename('ref')).dropna();v=(x.new-x.ref).to_numpy()
                samples=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
                sd=aa.groupby('seed')[metric].mean()-rr.groupby('seed')[metric].mean()
                paired.append({'target':target,'family':family,'new':ns+'_'+nd,'reference':rs+'_'+rd,'metric':metric,'difference':float(v.mean()),
                               'ci95':np.quantile(samples,[.025,.975]).tolist(),'lower_error_seeds':int((sd < -1e-12).sum()),'equal_seeds':int((abs(sd)<1e-12).sum()),'seed_differences':sd.to_dict(),'people':len(v)})
    save(HERE/'paired-bootstrap.json',{'scope':'conditional fixed-model paired participant bootstrap; average seeds per person; 2000 samples; no multiple comparison correction','comparisons':paired})
    historical=pd.read_csv(PREVIOUS/'metrics-by-seed.csv');historical=historical[historical.method.isin(['RFreg5_cal','RF_cal','OrdinalRF_cal'])]
    historical.to_csv(HERE/'historical-four-class-models.csv',index=False)
    outputs=[HERE/f for f in ['metrics-by-seed.csv','probabilities-by-seed.csv','warnings-by-seed.csv','confusion-by-seed.csv','inner-selections.csv','paired-bootstrap.json']]+[STORE/'primary-person-losses.csv.gz']
    save(HERE/'analysis-completion.json',{'metric_rows':len(m),'probability_rows':len(p),'warning_rows':len(o),'selection_rows':len(selections),'output_sha256':{str(f):sha(f) for f in outputs},
                                        'sha256':sha(__file__),'primary':'long5 selected; fixed4 common-distribution control differs from historical direct four-class forests'})
    print(m[m.band.eq('long')&m.n.eq(5)&m.decision.eq('selected')].groupby(['target','family','scheme'])[['accuracy','balanced_accuracy','extreme','miss','fpr']].mean().to_string(),flush=True)
    print('ANALYSIS_DONE',flush=True)

if __name__=='__main__':run()
