from study import *
spec=importlib.util.spec_from_file_location('match_reference',ref.REFERENCE/'analyze.py');matchref=importlib.util.module_from_spec(spec);spec.loader.exec_module(matchref)

def avg(v,ids):return float(np.average(v,weights=weights(ids)))
def metrics(d,name):
    y=d.is_low.to_numpy();ids=d.participantid.to_numpy();p=d[name+'_probability'].to_numpy();warning=d[name+'_fpr10'].to_numpy()
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    low=y==1;high=~low
    vectors={'brier':(p-y)**2,'logloss':-(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1))),
      'miss':np.where(low,warning==0,np.nan),'fpr':np.where(high,warning==1,np.nan),'warning_rate':warning.astype(float)}
    scores={k:avg(v[np.isfinite(v)],ids[np.isfinite(v)]) for k,v in vectors.items()}
    scores.update(auroc=float(roc_auc_score(y,p,sample_weight=weights(ids))),average_precision=float(average_precision_score(y,p,sample_weight=weights(ids))))
    bins=np.minimum((p*10).astype(int),9);ece=0.;w=weights(ids)
    for b in range(10):
        ix=bins==b
        if ix.any():ece+=float(w[ix].sum()/w.sum())*abs(float(np.average(p[ix],weights=w[ix]))-float(np.average(y[ix],weights=w[ix])))
    scores['ece10']=ece
    return scores,vectors

def run():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==50
    paths=sorted((STORE/'folds').glob('*.csv.gz'));assert len(paths)==50
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    assert not data.duplicated(['target','seed']+KEY).any()
    rows=[];matches=[];losses=[]
    for target,group in data.groupby('target'):
        baseline=group[group.seed.eq(SEEDS[0])].sort_values(['fold']+KEY).reset_index(drop=True)
        pairs={label:matchref.matched_pairs(baseline,target,flag) for label,flag in [('speed',False),('speed_cohort',True)]}
        for label,p in pairs.items():p.to_csv(STORE/f'matches-{target}-{label}.csv.gz',index=False)
        for seed,d in group.groupby('seed'):
            d=d.sort_values(['fold']+KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(d[KEY+['class','is_low','speed','fold','cohort']],baseline[KEY+['class','is_low','speed','fold','cohort']])
            for name in METHODS:
                scores,vectors=metrics(d,name);rows.append(dict(target=target,seed=int(seed),method=name,**scores))
                frame=pd.DataFrame({'id':d.participantid,**vectors}).groupby('id').mean().reset_index()
                frame['target']=target;frame['seed']=seed;frame['method']=name;losses.append(frame)
                prob=d[name+'_probability'].to_numpy()
                for label,p in pairs.items():
                    a=p.positive_index.to_numpy();b=p.negative_index.to_numpy();difference=prob[a]-prob[b];ids=p.positive_person.to_numpy()
                    matches.append(dict(target=target,seed=int(seed),method=name,matching=label,
                        pair_order_accuracy=avg((difference>0).astype(float)+.5*(difference==0),ids),
                        matched_low_people=int(p.positive_person.nunique()),control_people=int(p.negative_person.nunique()),pairs=len(p),
                        mean_speed_gap=avg(p.speed_gap.to_numpy(),ids)))
    m=pd.DataFrame(rows);mt=pd.DataFrame(matches);l=pd.concat(losses,ignore_index=True)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False);l.to_csv(STORE/'person-losses.csv.gz',index=False)
    comparisons=[];rng=np.random.default_rng(20261101)
    for target in ['sppb','tug']:
        for a,b in [('OriginalRF','MultiRF'),('MultiRF','BinaryRF'),('MultiHGB','BinaryHGB'),('OriginalRF','BinaryRF'),('OriginalRF','BinaryHGB')]:
            sub=l[l.target.eq(target)&l.method.isin([a,b])]
            for metric in ['brier','miss','fpr']:
                wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna();difference=(wide[b]-wide[a]).to_numpy()
                boot=[rng.choice(difference,len(difference),replace=True).mean() for _ in range(2000)]
                ss=m[m.target.eq(target)&m.method.isin([a,b])].pivot(index='seed',columns='method',values=metric)
                comparisons.append(dict(target=target,baseline=a,method=b,metric=metric,difference=float(difference.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),improved_seeds=int((ss[b]<ss[a]).sum())))
    save(HERE/'paired-bootstrap.json',{'scope':'fixed-prediction conditional person bootstrap, five seeds averaged per person; no retraining or multiplicity correction','comparisons':comparisons})
    primary=m.groupby(['target','method']).mean(numeric_only=True).reset_index();pm=mt.groupby(['target','method','matching']).mean(numeric_only=True).reset_index()
    save(HERE/'summary.json',{'metrics':primary.to_dict('records'),'matched':pm.to_dict('records'),'metric_rows':len(m),'matched_rows':len(mt),'analysis_sha256':sha(__file__)})
    print(primary.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
        time.sleep(.1)
    run()
