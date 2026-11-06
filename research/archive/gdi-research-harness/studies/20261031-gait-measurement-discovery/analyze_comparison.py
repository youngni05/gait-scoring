"""Summarize held-out predictions, selection stability and descriptive associations."""
from common import *
from scipy.stats import spearmanr
import warnings


def read_predictions(name):
    return pd.concat([pd.read_csv(p,float_precision='round_trip',dtype={'id':str}) for p in sorted((OUT/name).glob('r*f*.predictions.csv.gz'))],ignore_index=True)


def paired_ci(delta):
    rng = np.random.default_rng(20261031)
    means = []
    # Batches bound memory without changing the sampling definition.
    for _ in range(20):
        means.extend(delta[rng.integers(0,len(delta),(100,len(delta)))].mean(1).tolist())
    return {'difference':float(delta.mean()),'conditional95':np.quantile(means,[.025,.975]).tolist()}


def correlations(d, meta):
    records = []
    populations = [('all',np.ones(len(d['y']),bool))]
    if len(np.unique(d['cohort'])) > 1:
        populations += [(str(c),d['cohort']==c) for c in np.unique(d['cohort'])]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for group, mask in populations:
            group_rows = []
            for j, name in enumerate(meta['features']):
                valid = mask & np.isfinite(d['x'][:,j])
                if valid.sum() >= 5 and len(np.unique(d['x'][valid,j])) > 1 and len(np.unique(d['y'][valid])) > 1:
                    result = spearmanr(d['x'][valid,j],d['y'][valid])
                    rho,p = float(result.statistic),float(result.pvalue)
                else:
                    rho,p = np.nan,np.nan
                group_rows.append({'cohort':group,'feature':name,'people':int(valid.sum()),'rho':rho,'p':p})
            valid_p = [i for i,r in enumerate(group_rows) if np.isfinite(r['p'])]
            order = sorted(valid_p,key=lambda i:group_rows[i]['p'])
            q = 1.
            for rank,i in reversed(list(enumerate(order,1))):
                q = min(q,group_rows[i]['p']*len(order)/rank)
                group_rows[i]['BH_q'] = q
            records.extend(group_rows)
    return pd.DataFrame(records)


def analyze(name):
    folder = OUT/name
    execution = json.loads((folder/'execution.json').read_text())
    assert execution['status'] == 'complete'
    d, meta = load(name)
    p = read_predictions(name)
    n = len(d['y'])
    assert p.prediction.map(np.isfinite).all()
    p['error'] = abs(p.prediction-p.target)
    p['signed_error'] = p.prediction-p.target
    run_rows = []
    for (repeat,variant),g in p.groupby(['repeat','variant']):
        assert len(g) == g['sample'].nunique() == n
        assert set(g['sample']) == set(range(n))
        run_rows.append({'dataset':name,'repeat':repeat,'variant':variant,'people':n,'MAE':g.error.mean(),'RMSE':np.sqrt((g.signed_error**2).mean()),'bias':g.signed_error.mean(),'p90_absolute_error':g.error.quantile(.9),'out_of_target_range':int(((g.prediction<min(d['y']))|(g.prediction>max(d['y']))).sum())})
    runs = pd.DataFrame(run_rows)
    summary = runs.groupby('variant',as_index=False).agg(MAE=('MAE','mean'),MAE_min=('MAE','min'),MAE_max=('MAE','max'),RMSE=('RMSE','mean'),bias=('bias','mean'),repeats=('repeat','nunique'))
    summary = summary.sort_values(['MAE','variant'])
    summary.to_csv(HERE/f'{name}-metrics.csv',index=False)
    runs.to_csv(folder/'repeat-metrics.csv',index=False)
    base = p[p.variant=='Median|baseline:overall'].groupby('sample').error.mean().sort_index()
    errors = {variant:g.groupby('sample').error.mean().sort_index().to_numpy() for variant,g in p.groupby('variant')}
    interesting = set(summary.head(10).variant) | set(['selected_single','selected_small','selected_family','selected_any'])
    contrasts = {v:paired_ci(errors[v]-base.to_numpy()) for v in sorted(interesting)}
    policy_contrasts = {candidate+'_minus_selected_single':paired_ci(errors[candidate]-errors['selected_single']) for candidate in ['selected_small','selected_family','selected_any']}
    if 'CohortMedian|baseline:diagnosis_only' in errors:
        policy_contrasts['selected_any_minus_diagnosis_only'] = paired_ci(errors['selected_any']-errors['CohortMedian|baseline:diagnosis_only'])
    all_group = '+'.join(meta['groups'])
    contributions = []
    for model in ['Ridge','RF']:
        full = model+'|family:'+all_group
        for family in meta['groups']:
            remainder = '+'.join(k for k in meta['groups'] if k != family)
            without = model+'|family:'+remainder
            if without in errors:
                contrast = paired_ci(errors[full]-errors[without])
                contributions.append({'model':model,'added_family':family,**contrast})
    corr = correlations(d,meta)
    corr.to_csv(HERE/f'{name}-correlations.csv',index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        redundancy = pd.DataFrame(d['x'],columns=meta['features']).corr(method='spearman')
    redundancy.to_csv(HERE/f'{name}-measurement-correlations.csv')
    selected = []
    for path in sorted(folder.glob('r*f*.ledger.json')):
        ledger = json.loads(path.read_text(encoding='utf-8'))
        for item in ledger['selections']:
            selected.append({'repeat':ledger['repeat'],'fold':ledger['fold'],**item})
    selection_rows = []
    for policy in ['selected_single','selected_small','selected_family','selected_any']:
        items = [s for s in selected if s['policy'] == policy]
        for feature in meta['features']:
            count = sum(feature in s['feature_names'] for s in items)
            selection_rows.append({'policy':policy,'feature':feature,'selected_outer_cells':count,'total_outer_cells':len(items)})
    pd.DataFrame(selection_rows).to_csv(HERE/f'{name}-selection-frequency.csv',index=False)
    selected_counts = []
    for policy in ['selected_single','selected_small','selected_family','selected_any']:
        items = [s for s in selected if s['policy'] == policy]
        counts = pd.Series([s['variant'] for s in items]).value_counts()
        selected_counts.append({'policy':policy,'choices':[{'variant':v,'outer_cells':int(counts[v])} for v in counts.index]})
    cohort_rows = []
    selected_variants = ['Median|baseline:overall','CohortMedian|baseline:diagnosis_only','selected_any','selected_single','selected_small','selected_family']
    for (cohort,variant),g in p[p.variant.isin(selected_variants)].groupby(['cohort','variant']):
        cohort_rows.append({'cohort':cohort,'variant':variant,'people':g['sample'].nunique(),'MAE':g.error.mean(),'bias':g.signed_error.mean()})
    pd.DataFrame(cohort_rows).to_csv(HERE/f'{name}-cohort-metrics.csv',index=False)
    scope_rows = []
    for variant,g in p[p.variant.isin(selected_variants)].groupby('variant'):
        if name.endswith('tug'):
            masks = {'under10':g.target<10,'10to20':(g.target>=10)&(g.target<20),'20plus':g.target>=20,'at_most60':g.target<=60}
        else:
            masks = {str(int(score)):g.target==score for score in sorted(g.target.unique())}
        for label,mask in masks.items():
            s = g[mask]
            if len(s):
                scope_rows.append({'variant':variant,'target_group':label,'people':int(s['sample'].nunique()),'MAE':float(s.error.mean()),'bias':float(s.signed_error.mean()),'mean_underprediction':float(np.maximum(s.target-s.prediction,0).mean())})
    pd.DataFrame(scope_rows).to_csv(HERE/f'{name}-target-group-metrics.csv',index=False)
    top_single = summary[summary.variant.str.contains('|single:',regex=False)].head(8).to_dict('records')
    top_family = summary[summary.variant.str.contains('|family:',regex=False)].head(8).to_dict('records')
    policies = summary[summary.variant.isin(selected_variants)].to_dict('records')
    result = {'dataset':name,'people':n,'scope':meta['scope'],'target_range':meta['target_range'],'baseline_MAE':float(base.mean()),
        'selection_policy_metrics':policies,'top_fixed_single_exploratory':top_single,'top_fixed_families_exploratory':top_family,
        'paired_against_median':contrasts,'policy_contrasts':policy_contrasts,'full_minus_family_removed':contributions,'selection_choices':selected_counts,
        'limits':'OOF candidate ranking is exploratory. Selection policies alone evaluate choosing features without outer-test access. Paired intervals condition on fitted models and observed people; no new-site/clinical/measurement-accuracy validation. No winning threshold selected.'}
    save(HERE/f'{name}-summary.json',result)
    save(folder/'analysis-completion.json',{'predictions_sha256':{str(p):sha(p) for p in sorted(folder.glob('r*f*.predictions.csv.gz'))},'analysis_code_sha256':sha(HERE/'analyze_comparison.py'),'summary_sha256':sha(HERE/f'{name}-summary.json'),'complete':True})
    print(name,'baseline',round(base.mean(),4),'policies',[(r['variant'],round(r['MAE'],4)) for r in policies],flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=DATASETS)
    analyze(parser.parse_args().dataset)
