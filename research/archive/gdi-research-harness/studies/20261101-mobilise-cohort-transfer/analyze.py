from study import *
spec=importlib.util.spec_from_file_location('metric_reference',REFERENCE/'analyze.py');metricsref=importlib.util.module_from_spec(spec);spec.loader.exec_module(metricsref)

def run():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==40
    paths=sorted((STORE/'folds').glob('*.csv.gz'));assert len(paths)==40
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert not data.duplicated(['target','seed']+KEY).any()
    rows=[];cohort_rows=[];matched=[];losses=[]
    for target,group in data.groupby('target'):
        reference=group[group.seed.eq(SEEDS[0])].sort_values(['cohort']+KEY).reset_index(drop=True)
        # Original fold restriction ensures reference models share their calibration within a pair.
        matcher=reference.copy();matcher['fold']=matcher.original_fold
        pairs=metricsref.matched_pairs(matcher,target,True);pairs.to_csv(STORE/f'matches-{target}.csv.gz',index=False)
        for seed,d in group.groupby('seed'):
            d=d.sort_values(['cohort']+KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(d[KEY+['class','cohort','fold','original_fold','speed']],reference[KEY+['class','cohort','fold','original_fold','speed']])
            for name in ['Prior']+[m+'_'+v for m in METHODS for v in ['raw','cal']]:
                scores,vectors=metricsref.metrics(d,name,target);rows.append(dict(target=target,seed=int(seed),method=name,**scores))
                person=pd.DataFrame({'id':d.participantid,**vectors}).groupby('id').mean().reset_index();person['target']=target;person['seed']=seed;person['method']=name;losses.append(person)
                for cohort,sub in d.groupby('cohort'):
                    cs,_=metricsref.metrics(sub,name,target);low=sub['class'].eq(old.lowclass(target))
                    cohort_rows.append(dict(target=target,seed=int(seed),method=name,cohort=cohort,people=int(sub.participantid.nunique()),low_people=int(sub.loc[low,'participantid'].nunique()),low_visits=int(low.sum()),**cs))
                if name=='Prior' or not name.endswith('_cal'):continue
                a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy();risk=d[f'{name}_p{old.lowclass(target)}'].to_numpy();delta=risk[a]-risk[b]
                matched.append(dict(target=target,seed=int(seed),method=name,matching='same_cohort_original_fold',pair_order_accuracy=metricsref.avg((delta>0).astype(float)+.5*(delta==0),pairs.positive_person.to_numpy()),pairs=len(pairs),matched_low_people=int(pairs.positive_person.nunique()),control_people=int(pairs.negative_person.nunique())))
    m=pd.DataFrame(rows);cm=pd.DataFrame(cohort_rows);mt=pd.DataFrame(matched);l=pd.concat(losses,ignore_index=True)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);cm.to_csv(HERE/'cohort-metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False);l.to_csv(STORE/'person-losses.csv.gz',index=False)
    rng=np.random.default_rng(20261101);comparisons=[]
    for target in ['sppb','tug']:
        for a,b in [('MixedCoreRF_cal','LocoCoreRF_cal'),('MixedTurnRF_cal','LocoTurnRF_cal'),('LocoCoreRF_cal','LocoTurnRF_cal'),('MixedCoreRF_cal','MixedTurnRF_cal')]:
            sub=l[l.target.eq(target)&l.method.isin([a,b])]
            for metric in ['brier','rps','fpr10_miss','fpr10_fpr']:
                wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna();difference=(wide[b]-wide[a]).to_numpy();boot=[rng.choice(difference,len(difference),replace=True).mean() for _ in range(2000)]
                ss=m[m.target.eq(target)&m.method.isin([a,b])].pivot(index='seed',columns='method',values=metric)
                comparisons.append(dict(target=target,baseline=a,method=b,metric=metric,difference=float(difference.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),lower_error_seeds=int((ss[b]<ss[a]).sum())))
    save(HERE/'paired-bootstrap.json',{'scope':'fixed prediction person-cluster bootstrap, five seeds averaged, no retraining/multiplicity correction; training size and pathology are confounded in cohort holdout','comparisons':comparisons})
    primary=m.groupby(['target','method']).mean(numeric_only=True).reset_index();cohort=cm.groupby(['target','cohort','method']).mean(numeric_only=True).reset_index();pm=mt.groupby(['target','method']).mean(numeric_only=True).reset_index()
    clean=lambda df:df.astype(object).where(pd.notna(df),None).to_dict('records')
    save(HERE/'summary.json',{'metrics':clean(primary),'cohort_metrics':clean(cohort),'matched':clean(pm),'metric_rows':len(m),'cohort_metric_rows':len(cm),'matched_rows':len(mt),'analysis_sha256':sha(__file__)})
    print(primary[['target','method','brier','fpr10_miss','fpr10_fpr']].to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
        time.sleep(.1)
    run()
