"""Person-weighted probabilities, low-function errors, and speed-matched diagnostics."""
from study import *

def avg(x, ids):
    return float(np.average(x, weights=weights(ids))) if len(x) else None

def metrics(d, name, target):
    y = d['class'].to_numpy(); ids = d.participantid.to_numpy()
    p = d[[f'{name}_p{j}' for j in range(4)]].to_numpy()
    assert np.isfinite(p).all() and p.min() >= 0
    np.testing.assert_allclose(p.sum(1), 1, rtol=0, atol=1e-12)
    truth = np.eye(4)[y]; pred = p.argmax(1); low = y == old.lowclass(target)
    vectors = {'brier': ((p-truth)**2).sum(1),
               'rps': ((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1),
               'log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)),
               'accuracy': (pred==y).astype(float), 'extreme_error': (abs(pred-y)==3).astype(float),
               'low_miss': np.where(low, pred!=old.lowclass(target), np.nan),
               'fpr': np.where(~low, pred==old.lowclass(target), np.nan)}
    if name != 'Prior':
        warn = d[f'{name}_fpr10'].to_numpy()
        vectors['fpr10_miss'] = np.where(low, warn==0, np.nan)
        vectors['fpr10_fpr'] = np.where(~low, warn==1, np.nan)
    result = {}
    for key, value in vectors.items():
        ix = np.isfinite(value); result[key] = avg(value[ix], ids[ix])
    for j in range(4): result[f'recall_{j}'] = avg(pred[y==j]==j, ids[y==j])
    recalls = [result[f'recall_{j}'] for j in range(4)]
    result['balanced_recall'] = float(np.mean(recalls)) if all(v is not None for v in recalls) else None
    return result, vectors

def matched_pairs(d, target, same_cohort):
    """Evaluation-only matching with fixed caliper; no score probability used to match."""
    low = d['class'].to_numpy() == old.lowclass(target)
    rows = []
    for fold, frame in d.groupby('fold'):
        pos = frame[low[frame.index]]
        neg = frame[~low[frame.index]].sort_values(KEY)
        for i, r in pos.iterrows():
            controls = neg[neg.participantid.ne(r.participantid)]
            if same_cohort: controls = controls[controls.cohort.eq(r.cohort)]
            if not len(controls): continue
            gap = abs(controls.speed.to_numpy()-r.speed)
            j = int(np.argmin(gap))
            if gap[j] > .05 + 1e-12: continue
            index = int(controls.index[j])
            rows.append({'positive_index': int(i), 'negative_index': index,
                         'positive_person': int(r.participantid), 'negative_person': int(d.loc[index,'participantid']),
                         'speed_gap': float(gap[j]), 'fold': int(fold)})
    return pd.DataFrame(rows)

def load():
    completion = json.loads((HERE/'main-completion.json').read_text())
    assert completion['tasks'] == 50
    paths = sorted((STORE/'folds').glob('*.csv.gz')); assert len(paths)==50
    p = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    assert not p.duplicated(['target','band','k','seed']+KEY).any()
    return p

def run():
    started = time.time(); p = load(); rows=[]; matched=[]; strata=[]; person_losses=[]
    for (target, band, k), group in p.groupby(['target','band','k'], sort=True):
        reference = group[group.seed.eq(SEEDS[0])].sort_values(['fold']+KEY).reset_index(drop=True)
        pairs = {label: matched_pairs(reference, target, flag) for label,flag in [('speed',False),('speed_cohort',True)]}
        for label, frame in pairs.items(): frame.to_csv(STORE/f'matches-{target}-{band}-{k}-{label}.csv.gz',index=False)
        for seed, d in group.groupby('seed', sort=True):
            d=d.sort_values(['fold']+KEY).reset_index(drop=True)
            pd.testing.assert_frame_equal(d[KEY+['fold','class','speed','cohort']],reference[KEY+['fold','class','speed','cohort']])
            info={'target':target,'band':band,'k':int(k),'seed':int(seed)}
            ids=d.participantid.to_numpy(); y=d['class'].to_numpy(); islow=y==old.lowclass(target)
            for name in ['Prior']+[m+'_'+v for m in METHODS for v in ['raw','cal']]:
                score, vectors=metrics(d,name,target); rows.append(dict(info,method=name,**score))
                if band=='long' and k==5:
                    losses=pd.DataFrame({'id':ids,**vectors}).groupby('id').mean().reset_index()
                    for key,value in dict(target=target,seed=seed,method=name).items(): losses[key]=value
                    person_losses.append(losses)
                if name=='Prior' or not name.endswith('_cal'): continue
                risk=d[f'{name}_p{old.lowclass(target)}'].to_numpy()
                for label, frame in pairs.items():
                    a=frame.positive_index.to_numpy(); b=frame.negative_index.to_numpy(); pair_ids=frame.positive_person.to_numpy()
                    delta=risk[a]-risk[b]
                    rank=(delta>0).astype(float)+.5*(delta==0)
                    matched.append(dict(info,method=name,matching=label,
                         pair_order_accuracy=avg(rank,pair_ids),mean_risk_gap=avg(delta,pair_ids),
                         matched_low_miss=avg(d[f'{name}_fpr10'].to_numpy()[a]==0,pair_ids),
                         matched_control_warning=avg(d[f'{name}_fpr10'].to_numpy()[b]==1,pair_ids),
                         pairs=len(frame),low_people=len(np.unique(ids[islow])),matched_low_people=len(np.unique(pair_ids)),
                         unique_control_people=frame.negative_person.nunique(),
                         low_visit_coverage=len(frame)/int(islow.sum()),mean_speed_gap=avg(frame.speed_gap.to_numpy(),pair_ids)))
                # Fixed physical bins, no data-driven speed cut selection.
                bins=np.floor(d.speed.to_numpy()/.1).astype(int)
                for speedbin in np.unique(bins):
                    ix=bins==speedbin; sub=d.loc[ix]
                    m,_=metrics(sub,name,target)
                    strata.append(dict(info,method=name,speed_lower=float(speedbin/10),people=sub.participantid.nunique(),
                                       low_people=sub[sub['class'].eq(old.lowclass(target))].participantid.nunique(),
                                       brier=m['brier'],fpr10_miss=m['fpr10_miss'],fpr10_fpr=m['fpr10_fpr']))
    m=pd.DataFrame(rows); mt=pd.DataFrame(matched); st=pd.DataFrame(strata)
    m.to_csv(HERE/'metrics-by-seed.csv',index=False);mt.to_csv(HERE/'matched-by-seed.csv',index=False)
    st.to_csv(HERE/'speed-strata-by-seed.csv',index=False)
    keys=['target','band','k','method']; numerical=[c for c in m if c not in keys+['seed']]
    summary=m.groupby(keys)[numerical].agg(['mean','min','max']);summary.to_csv(HERE/'summary-by-condition.csv')
    match_summary=mt.groupby(keys+['matching'])[[c for c in mt if c not in keys+['matching','seed']]].agg(['mean','min','max'])
    match_summary.to_csv(HERE/'matched-summary.csv')
    losses=pd.concat(person_losses,ignore_index=True);losses.to_csv(STORE/'primary-person-losses.csv.gz',index=False)
    comparisons=[]; rng=np.random.default_rng(20261031)
    for target in ['sppb','tug']:
        for base,method in [('RF_cal','OrdinalRF_cal')]:
            sub=losses[losses.target.eq(target)&losses.method.isin([base,method])]
            for metric in ['brier','rps','accuracy','extreme_error','low_miss','fpr10_miss','fpr10_fpr']:
                wide=sub.groupby(['id','method'])[metric].mean().unstack().dropna()
                diff=(wide[method]-wide[base]).to_numpy()
                boot=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)])
                seedwise=sub.groupby(['seed','method'])[metric].mean().unstack()
                sd=seedwise[method]-seedwise[base]
                comparisons.append(dict(target=target,baseline=base,method=method,metric=metric,
                    difference=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                    better_seeds=int((sd>0).sum() if metric=='accuracy' else (sd<0).sum()),seed_differences=sd.to_dict()))
    save(HERE/'paired-bootstrap.json',{'scope':'paired person bootstrap of fixed OOF errors, average five seeds within each person; no retraining or multiplicity correction; matching is descriptive only','comparisons':comparisons})
    primary=m[m.band.eq('long')&m.k.eq(5)].groupby(['target','method'])[numerical].mean().reset_index()
    pm=mt[mt.band.eq('long')&mt.k.eq(5)].groupby(['target','method','matching']).mean(numeric_only=True).reset_index()
    save(HERE/'summary.json',{'primary':'long5, draw0, five-seed mean (not ensemble)',
        'metrics':primary.astype(object).where(pd.notna(primary),None).to_dict('records'),
        'matched':pm.astype(object).where(pd.notna(pm),None).to_dict('records'),
        'metrics_rows':len(m),'matched_rows':len(mt),'analysis_sha256':sha(__file__),'seconds':time.time()-started})
    print(primary[['target','method','brier','accuracy','low_miss','fpr10_miss','fpr10_fpr']].to_string(index=False))
    print(pm[['target','method','matching','pair_order_accuracy','matched_low_people','mean_speed_gap']].to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--follow',action='store_true')
    args=parser.parse_args()
    if args.follow:
        while not (HERE/'main-completion.json').exists():time.sleep(1)
        time.sleep(.1)
    run()
