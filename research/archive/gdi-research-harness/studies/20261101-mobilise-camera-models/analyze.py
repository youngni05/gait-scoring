from study import *
spec=importlib.util.spec_from_file_location('gait_metric_helpers',ref.REFERENCE/'analyze.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)

def run():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==50
    paths=[STORE/'folds'/f'{t}-f{f}-s{s}.csv.gz' for t,f,s in tasks('main')];assert all(p.exists() for p in paths)
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);assert not data.duplicated(['target','seed']+KEY).any()
    rows=[];matched=[];losses=[]
    for target,g in data.groupby('target'):
        reference=g[g.seed.eq(SEEDS[0])].sort_values(['fold']+KEY).reset_index(drop=True);pairs=metric.matched_pairs(reference,target,True);pairs.to_csv(STORE/f'matches-{target}.csv.gz',index=False)
        for seed,d in g.groupby('seed'):
            d=d.sort_values(['fold']+KEY).reset_index(drop=True);pd.testing.assert_frame_equal(d[KEY+['fold','cohort','class','speed']],reference[KEY+['fold','cohort','class','speed']])
            for name in ['Prior']+[n+'_'+v for n in METHODS for v in ['raw','cal']]:
                scores,vectors=metric.metrics(d,name,target);rows.append(dict(target=target,seed=int(seed),method=name,**scores))
                person=pd.DataFrame({'id':d.participantid,**vectors}).groupby('id').mean().reset_index();person['target']=target;person['seed']=seed;person['method']=name;losses.append(person)
                if name=='Prior' or not name.endswith('_cal'):continue
                a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy();risk=d[f'{name}_p{old.lowclass(target)}'].to_numpy();delta=risk[a]-risk[b]
                matched.append(dict(target=target,seed=int(seed),method=name,pair_order_accuracy=metric.avg((delta>0).astype(float)+.5*(delta==0),pairs.positive_person.to_numpy()),pairs=len(pairs),matched_low_people=int(pairs.positive_person.nunique())))
    m=pd.DataFrame(rows);mt=pd.DataFrame(matched);l=pd.concat(losses,ignore_index=True);m.to_csv(HERE/'metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False);l.to_csv(STORE/'person-losses.csv.gz',index=False)
    rng=np.random.default_rng(20261101);comparisons=[]
    for target,(a,b),key in itertools.product(['sppb','tug'],[('SpeedFixedRF_cal','SpeedAccRF_cal'),('CadenceFixedRF_cal','CadenceAccRF_cal'),('SpeedAccRF_cal','SpeedDS_cal'),('CadenceAccRF_cal','CadenceDS_cal'),('SpeedFixedRF_cal','CadenceFixedRF_cal'),('SpeedAccRF_cal','CadenceAccRF_cal'),('SpeedDS_cal','CadenceDS_cal')],['accuracy','brier','extreme_error','fpr10_miss','fpr10_fpr']):
        sub=l[l.target.eq(target)&l.method.isin([a,b])];wide=sub.groupby(['id','method'])[key].mean().unstack().dropna();delta=(wide[b]-wide[a]).to_numpy();boot=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
        ss=m[m.target.eq(target)&m.method.isin([a,b])].pivot(index='seed',columns='method',values=key);win=ss[b]>ss[a] if key=='accuracy' else ss[b]<ss[a]
        comparisons.append(dict(target=target,baseline=a,method=b,metric=key,difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),improved_seeds=int(win.sum())))
    save(HERE/'paired-bootstrap.json',{'scope':'same development people reused,person clusters,fixed predictions across5seeds,no retraining/multiplicity correction','comparisons':comparisons})
    clean=lambda df:df.astype(object).where(pd.notna(df),None).to_dict('records')
    save(HERE/'summary.json',{'metrics':clean(m.groupby(['target','method']).mean(numeric_only=True).reset_index()),'matched':clean(mt.groupby(['target','method']).mean(numeric_only=True).reset_index()),'metric_rows':len(m),'matched_rows':len(mt),'analysis_sha256':sha(__file__)});print('ANALYZED',len(m),len(mt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
    run()
