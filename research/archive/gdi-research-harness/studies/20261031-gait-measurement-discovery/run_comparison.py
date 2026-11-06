"""Fit candidates and choose policies using only each outer training population."""
from common import *
import argparse
import time


def run(name):
    started = time.time()
    prep = json.loads((HERE/'comparison-preparation.json').read_text(encoding='utf-8'))
    for path, digest in prep['source_sha256'].items():
        assert sha(path) == digest, path
    d, meta = load(name)
    assert sha(OUT/name/'input.npz') == meta['input_sha256']
    x, y, cohort = d['x'], d['y'], d['cohort']
    n, p = x.shape
    folder = OUT/name
    save(folder/'execution.json', {'status':'running','source_preparation_sha256':sha(HERE/'comparison-preparation.json'), 'device':'local CPU','people':n})
    for repeat in range(3):
        assignments = d['folds'][repeat]
        for fold in range(5):
            prefix = folder/f'r{repeat}f{fold}'
            marker = prefix.with_suffix('.complete.json')
            if marker.exists():
                previous = json.loads(marker.read_text())
                for path, digest in previous['outputs'].items():
                    assert sha(path) == digest
                print(f'{name} repeat={repeat} fold={fold} already complete',flush=True)
                continue
            tr, te = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
            inner_seed = 7000+100*repeat+fold
            seed = 9000+100*repeat+fold
            inner = [(tr[a],tr[b]) for a,b in KFold(3,shuffle=True,random_state=inner_seed).split(tr)]
            cache = {}
            candidates = []

            def evaluate(label, category, columns, model, flags=False):
                columns = tuple(sorted(columns))
                key = (columns,model,flags)
                if key not in cache:
                    z = np.isnan(x[:,columns]).astype(float) if flags else x[:,columns]
                    pred = np.full(n,np.nan)
                    for a,b in inner:
                        if model == 'CohortMedian':
                            lookup = {c:np.median(y[a][cohort[a]==c]) for c in np.unique(cohort[a])}
                            pred[b] = [lookup.get(c,np.median(y[a])) for c in cohort[b]]
                        else:
                            pred[b] = fit_predict(model,z,y,a,b,seed,n)
                    assert np.isfinite(pred[tr]).all() and np.isnan(pred[te]).all()
                    cache[key] = {'inner_mae':float(abs(pred[tr]-y[tr]).mean()), 'z':z}
                item = {'variant':model+'|'+category+':'+label,'name':label,'category':category,'columns':list(columns),'model':model,'flags_only':flags,'inner_mae':cache[key]['inner_mae']}
                candidates.append(item)
                return item

            evaluate('overall','baseline',[], 'Median')
            if len(np.unique(cohort)) > 1:
                evaluate('diagnosis_only','baseline',[], 'CohortMedian')
            for i, feature in enumerate(meta['features']):
                for model in ['Ridge','RF']:
                    evaluate(feature,'single',[i],model)
            for label, columns in family_sets(meta['groups']):
                for model in ['Ridge','RF']:
                    evaluate(label,'family',columns,model)
            for model in ['Ridge','RF']:
                evaluate('all_missingness','control',list(range(p)),model,True)

            # Greedy forward search is repeated from scratch inside every outer training set.
            first = min((c for c in candidates if c['category']=='single' and c['model']=='Ridge'), key=lambda c:(c['inner_mae'],c['variant']))
            chosen = first['columns'][:]
            for count in [2,3]:
                options = []
                for col in range(p):
                    if col in chosen:
                        continue
                    columns = sorted(chosen+[col])
                    item = evaluate('+'.join(meta['features'][i] for i in columns), 'search', columns, 'Ridge')
                    options.append(item)
                winner = min(options,key=lambda c:(c['inner_mae'],c['variant']))
                chosen = winner['columns'][:]
                candidates.append({**winner,'variant':f'Ridge|forward:{count}','category':'forward','name':f'{count}_measurements'})
            # Keep forward search trials in the ledger, but only the selected paths are evaluated outside.
            tested = [c for c in candidates if c['category'] != 'search']
            policies = {
                'selected_single':[c for c in tested if c['category']=='single'],
                'selected_small':[c for c in tested if c['category'] in ['single','forward']],
                'selected_family':[c for c in tested if c['category']=='family'],
                'selected_any':[c for c in tested if c['category'] in ['single','forward','family'] or c['model']=='Median'],
            }
            selections = []
            for policy, options in policies.items():
                winner = min(options,key=lambda c:(c['inner_mae'],len(c['columns']),c['variant']))
                selections.append({'policy':policy,**winner,'feature_names':[meta['features'][i] for i in winner['columns']]})
            predictions, outer_cache = [], {}
            tasks = [(c['variant'], c) for c in tested]+[(s['policy'],s) for s in selections]
            for variant, c in tasks:
                key = (tuple(c['columns']),c['model'],c['flags_only'])
                if key not in outer_cache:
                    if c['model'] == 'CohortMedian':
                        lookup = {v:np.median(y[tr][cohort[tr]==v]) for v in np.unique(cohort[tr])}
                        pred = np.array([lookup.get(v,np.median(y[tr])) for v in cohort[te]])
                    else:
                        pred = fit_predict(c['model'],cache[key]['z'],y,tr,te,seed,n)
                    assert np.isfinite(pred).all()
                    outer_cache[key] = pred
                predictions.append(pd.DataFrame({'repeat':repeat,'fold':fold,'variant':variant,'model':c['model'],'selected_key':c['variant'],
                    'sample':te,'id':d['ids'][te],'cohort':cohort[te],'target':y[te],'prediction':outer_cache[key]}))
            pred_path = prefix.with_suffix('.predictions.csv.gz')
            ledger_path = prefix.with_suffix('.ledger.json')
            pd.concat(predictions,ignore_index=True).to_csv(pred_path,index=False)
            save(ledger_path,{'repeat':repeat,'fold':fold,'seed':seed,'inner_seed':inner_seed,'train':tr.tolist(),'test':te.tolist(),
                'inner':[{'train':a.tolist(),'validation':b.tolist()} for a,b in inner],
                'candidates':candidates,'selections':selections,'outer_unique_fits':len(outer_cache),'inner_unique_candidates':len(cache)})
            save(marker,{'outputs':{str(pred_path):sha(pred_path),str(ledger_path):sha(ledger_path)},'seconds_since_start':time.time()-started})
            print(f'{name} repeat={repeat} fold={fold} candidates={len(tested)} inner_unique={len(cache)} seconds={time.time()-started:.1f}',flush=True)
    save(folder/'execution.json',{'status':'complete','source_preparation_sha256':sha(HERE/'comparison-preparation.json'),'device':'local CPU','people':n,'outer_cells':15,'seconds':time.time()-started,'models_saved':False})
    print(f'{name} TRAINING_FINISHED seconds={time.time()-started:.1f}',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=DATASETS)
    args = parser.parse_args()
    run(args.dataset)
