"""Independently check linkage, nested choices, prediction arithmetic and fit replays."""
from common import *
from analyze_comparison import read_predictions
from sklearn.pipeline import Pipeline
import argparse


def replay(model, x, y, train, test, seed, n):
    if model == 'Median':
        return np.repeat(np.median(y[train]),len(test))
    regressor = (Ridge(alpha=10) if model == 'Ridge' else RandomForestRegressor(n_estimators=128,min_samples_leaf=10 if n>=500 else 2,max_features=1.,random_state=seed,n_jobs=2))
    pipeline = Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler()),('model',regressor)])
    kwargs = {'model__sample_weight':np.ones(len(train))} if model=='RF' else {}
    pipeline.fit(x[train],y[train],**kwargs)
    return pipeline.predict(x[test])


def main(name):
    prep = json.loads((HERE/'comparison-preparation.json').read_text(encoding='utf-8'))
    assert sha(HERE/'comparison-protocol.json') == prep['protocol_sha256']
    for path,digest in prep['source_sha256'].items():
        assert sha(path) == digest,path
    d,meta = load(name)
    assert sha(OUT/name/'input.npz') == meta['input_sha256']
    n = len(d['y'])
    assert n == len(set(d['ids'])) == meta['people']
    p = read_predictions(name)
    selection_count, inner_replays, outer_replays = 0,0,0
    replay_keys = set()
    for repeat in range(3):
        cohort = d['cohort']
        if len(set(cohort))>1 and min(pd.Series(cohort).value_counts())>=5:
            splitter = StratifiedKFold(5,shuffle=True,random_state=[17,41,89][repeat])
            expected = list(splitter.split(np.zeros(n),cohort))
        else:
            expected = list(KFold(5,shuffle=True,random_state=[17,41,89][repeat]).split(np.arange(n)))
        for fold,(train,test) in enumerate(expected):
            prefix = OUT/name/f'r{repeat}f{fold}'
            marker = json.loads(prefix.with_suffix('.complete.json').read_text())
            for path,digest in marker['outputs'].items():
                assert sha(path) == digest
            ledger = json.loads(prefix.with_suffix('.ledger.json').read_text(encoding='utf-8'))
            assert ledger['train'] == train.tolist() and ledger['test'] == test.tolist()
            assert set(train).isdisjoint(test) and len(train)+len(test)==n
            valid = []
            for item in ledger['inner']:
                a,b = np.array(item['train']),np.array(item['validation'])
                assert set(a).isdisjoint(b) and set(a)|set(b)==set(train)
                assert set(test).isdisjoint(a) and set(test).isdisjoint(b)
                valid.extend(b.tolist())
            assert sorted(valid) == sorted(train)
            expected_inner = [(train[a].tolist(),train[b].tolist()) for a,b in KFold(3,shuffle=True,random_state=7000+100*repeat+fold).split(train)]
            assert expected_inner == [(v['train'],v['validation']) for v in ledger['inner']]
            c = ledger['candidates']
            assert len([v for v in c if v['category']=='single']) == 2*len(meta['features'])
            assert len([v for v in c if v['category']=='family']) == 2*len(family_sets(meta['groups']))
            for item in c:
                assert np.isfinite(item['inner_mae']) and item['inner_mae']>=0
                assert len(item['columns'])==len(set(item['columns']))
                assert all(0<=i<len(meta['features']) for i in item['columns'])
            choices = {
                'selected_single':[v for v in c if v['category']=='single'],
                'selected_small':[v for v in c if v['category'] in ['single','forward']],
                'selected_family':[v for v in c if v['category']=='family'],
                'selected_any':[v for v in c if v['category'] in ['single','forward','family'] or v['model']=='Median'],
            }
            cell = p[(p['repeat']==repeat)&(p.fold==fold)]
            for item in ledger['selections']:
                winner = sorted(choices[item['policy']],key=lambda v:(v['inner_mae'],len(v['columns']),v['variant']))[0]
                assert item['variant']==winner['variant'] and item['columns']==winner['columns'] and item['inner_mae']==winner['inner_mae']
                assert item['feature_names']==[meta['features'][i] for i in item['columns']]
                chosen = cell[cell.variant==item['policy']].sort_values('sample')
                reference = cell[cell.variant==winner['variant']].sort_values('sample')
                np.testing.assert_array_equal(chosen.prediction,reference.prediction)
                selection_count += 1
                if repeat==0 and fold==0 and item['policy'] in ['selected_single','selected_any']:
                    key = (tuple(item['columns']),item['model'])
                    if key in replay_keys:
                        continue
                    replay_keys.add(key)
                    z = d['x'][:,item['columns']]
                    values = []
                    for s in ledger['inner']:
                        a,b = np.array(s['train']),np.array(s['validation'])
                        pred = replay(item['model'],z,d['y'],a,b,ledger['seed'],n)
                        values.extend(abs(pred-d['y'][b]).tolist())
                        inner_replays += 1
                    assert abs(np.mean(values)-item['inner_mae'])<1e-10
                    pred = replay(item['model'],z,d['y'],train,test,ledger['seed'],n)
                    np.testing.assert_allclose(pred,chosen.prediction,atol=1e-9,rtol=0)
                    outer_replays += 1
            for variant,g in cell.groupby('variant'):
                assert sorted(g['sample'])==test.tolist() and not g['sample'].duplicated().any()
                rows = g['sample'].to_numpy(int)
                np.testing.assert_array_equal(g.id.astype(str),d['ids'][rows])
                np.testing.assert_array_equal(g.target,d['y'][rows])
    metrics = pd.read_csv(HERE/f'{name}-metrics.csv')
    arithmetic = []
    for variant,g in p.groupby('variant'):
        mae = []
        for repeat,s in g.groupby('repeat'):
            assert s['sample'].nunique()==len(s)==n
            # Arithmetic is computed from raw arrays, not the analysis result table.
            mae.append(sum(abs(float(a)-float(b)) for a,b in zip(s.prediction,s.target))/n)
        assert len(mae)==3
        saved = metrics[metrics.variant==variant].iloc[0]
        arithmetic.append(abs(sum(mae)/3-saved.MAE))
        assert arithmetic[-1]<1e-10
    analysis = json.loads((OUT/name/'analysis-completion.json').read_text())
    assert sha(HERE/f'{name}-summary.json') == analysis['summary_sha256']
    for path,digest in analysis['predictions_sha256'].items():
        assert sha(path)==digest
    result = {'passed':True,'people':n,'source_hashes_checked':len(prep['source_sha256']),'outer_cells':15,'nested_policy_choices_checked':selection_count,
        'inner_fit_replays':inner_replays,'outer_fit_replays':outer_replays,'all_prediction_ids_targets_and_coverage_checked':True,'max_metric_difference':max(arithmetic),'prediction_rows':len(p),
        'limits':'Representative fits replayed, not every forest refitted. Clinical source truth, external generalization and measurement accuracy remain unverified.'}
    save(HERE/f'{name}-verification.json',result)
    print(name,json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=DATASETS)
    main(parser.parse_args().dataset)
