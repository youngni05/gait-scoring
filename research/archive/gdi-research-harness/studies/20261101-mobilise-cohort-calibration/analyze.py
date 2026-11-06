from study import *
spec=importlib.util.spec_from_file_location('gait_metric_reference',REFERENCE.parent/'20261101-mobilise-duration-turns/analyze.py');metricsref=importlib.util.module_from_spec(spec);spec.loader.exec_module(metricsref)

def run():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==40
    paths=sorted((STORE/'folds').glob('*.csv.gz'));assert len(paths)==40
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert not data.duplicated(['target','seed']+KEY).any()
    rows=[];cohort_rows=[];matched=[];losses=[]
    for target,group in data.groupby('target'):
        reference=group[group.seed.eq(SEEDS[0])].sort_values(['cohort']+KEY).reset_index(drop=True)
        matcher=reference.copy();matcher['fold']=matcher.original_fold
        pairs=metricsref.matched_pairs(matcher,target,True);pairs.to_csv(STORE/f'matches-{target}.csv.gz',index=False)
        for seed,d in group.groupby('seed'):
            d=d.sort_values(['cohort']+KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(d[KEY+['class','cohort','fold','original_fold','speed']],reference[KEY+['class','cohort','fold','original_fold','speed']])
            for name in ['Prior']+METHODS:
                scores,vectors=metricsref.metrics(d,name,target)
                if name!='Prior':scores['warning_rate']=metricsref.avg(d[f'{name}_fpr10'].to_numpy(),d.participantid.to_numpy())
                rows.append(dict(target=target,seed=int(seed),method=name,**scores))
                person=pd.DataFrame({'id':d.participantid,**vectors}).groupby('id').mean().reset_index();person['target']=target;person['seed']=seed;person['method']=name;losses.append(person)
                for cohort,sub in d.groupby('cohort'):
                    cs,_=metricsref.metrics(sub,name,target);low=sub['class'].eq(old.lowclass(target))
                    if name!='Prior':cs['warning_rate']=metricsref.avg(sub[f'{name}_fpr10'].to_numpy(),sub.participantid.to_numpy())
                    cohort_rows.append(dict(target=target,seed=int(seed),method=name,cohort=cohort,people=int(sub.participantid.nunique()),low_people=int(sub.loc[low,'participantid'].nunique()),**cs))
                if name=='Prior':continue
                aa=pairs.positive_index.to_numpy();bb=pairs.negative_index.to_numpy();risk=d[f'{name}_p{old.lowclass(target)}'].to_numpy();delta=risk[aa]-risk[bb]
                matched.append(dict(target=target,seed=int(seed),method=name,pair_order_accuracy=metricsref.avg((delta>0).astype(float)+.5*(delta==0),pairs.positive_person.to_numpy()),pairs=len(pairs),matched_low_people=int(pairs.positive_person.nunique())))
    m=pd.DataFrame(rows);cm=pd.DataFrame(cohort_rows);mt=pd.DataFrame(matched);l=pd.concat(losses,ignore_index=True)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);cm.to_csv(HERE/'cohort-metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False);l.to_csv(STORE/'person-losses.csv.gz',index=False)
    rng=np.random.default_rng(20261101);comparisons=[]
    for target,method,metric in itertools.product(['sppb','tug'],['CohortCal','CohortGuard'],['brier','rps','fpr10_miss','fpr10_fpr']):
        sub=l[l.target.eq(target)&l.method.isin(['PersonCal',method])];wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna();delta=(wide[method]-wide.PersonCal).to_numpy()
        boot=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
        ss=m[m.target.eq(target)&m.method.isin(['PersonCal',method])].pivot(index='seed',columns='method',values=metric)
        comparisons.append(dict(target=target,method=method,baseline='PersonCal',metric=metric,difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),lower_error_seeds=int((ss[method]<ss.PersonCal).sum())))
    save(HERE/'paired-bootstrap.json',{'scope':'person-cluster fixed predictions, seeds averaged, no multiplicity correction or external validation','comparisons':comparisons})
    clean=lambda df:df.astype(object).where(pd.notna(df),None).to_dict('records')
    save(HERE/'summary.json',{'metrics':clean(m.groupby(['target','method']).mean(numeric_only=True).reset_index()),'cohort_metrics':clean(cm.groupby(['target','cohort','method']).mean(numeric_only=True).reset_index()),'matched':clean(mt.groupby(['target','method']).mean(numeric_only=True).reset_index()),'metric_rows':len(m),'cohort_metric_rows':len(cm),'matched_rows':len(mt),'analysis_sha256':sha(__file__)})
    print('ANALYZED',len(m),len(cm),len(mt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
    run()
