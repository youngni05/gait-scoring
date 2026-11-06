from study import *

def run():
    assert json.loads((HERE/'main-completion.json').read_text())['tasks']==40
    files=[STORE/f'{t}-{c}-s{s}.csv.gz' for t,c,s in tasks('main')];assert len(files)==40 and all(f.exists() for f in files)
    data=pd.concat([pd.read_csv(f) for f in files],ignore_index=True);rows=[];groups=[];losses=[]
    for (target,seed),d in data.groupby(['target','seed']):
        for name in METHODS:
            value=vectors(d,target,name);rows.append(dict(target=target,seed=int(seed),method=name,paired_people=int(d.participantid.nunique()),pairs=len(d),**{k:mean(x,d.participantid.to_numpy()) for k,x in value.items()}))
            person=pd.DataFrame({'id':d.participantid,**value}).groupby('id').mean().reset_index();person['target']=target;person['seed']=seed;person['method']=name;losses.append(person)
            for cohort,g in d.groupby('held_cohort'):
                groups.append(dict(target=target,seed=int(seed),method=name,cohort=cohort,paired_people=int(g.participantid.nunique()),pairs=len(g),**{k:mean(x,g.participantid.to_numpy()) for k,x in vectors(g,target,name).items()}))
    m=pd.DataFrame(rows);cm=pd.DataFrame(groups);loss=pd.concat(losses,ignore_index=True)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);cm.to_csv(HERE/'cohort-metrics-by-seed.csv',index=False);loss.to_csv(STORE/'person-losses.csv.gz',index=False)
    comparisons=[];rng=np.random.default_rng(20261101)
    for target,baseline,metric in itertools.product(['sppb','tug'],['Speed','CoreRF','NoChange'],['direction','class_delta_mae']):
        if baseline=='Speed' and metric=='class_delta_mae':continue
        sub=loss[loss.target.eq(target)&loss.method.isin([baseline,'TurnRF'])];wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna();delta=(wide.TurnRF-wide[baseline]).to_numpy();boot=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
        ss=m[m.target.eq(target)&m.method.isin([baseline,'TurnRF'])].pivot(index='seed',columns='method',values=metric)
        improved=ss.TurnRF>ss[baseline] if metric=='direction' else ss.TurnRF<ss[baseline]
        comparisons.append(dict(target=target,baseline=baseline,method='TurnRF',metric=metric,difference=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),improved_seeds=int(improved.sum())))
    save(HERE/'paired-bootstrap.json',{'scope':'person clusters,all within-person visitpairs retained,seeds averaged,conditional on frozen models,no multiplicity correction','comparisons':comparisons})
    clean=lambda x:x.astype(object).where(pd.notna(x),None).to_dict('records')
    save(HERE/'summary.json',{'metrics':clean(m.groupby(['target','method']).mean(numeric_only=True).reset_index()),'cohort_metrics':clean(cm.groupby(['target','cohort','method']).mean(numeric_only=True).reset_index()),'metric_rows':len(m),'cohort_metric_rows':len(cm),'analysis_sha256':sha(__file__)});print('ANALYZED',len(m),len(cm),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--follow',action='store_true');args=ap.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
    run()
