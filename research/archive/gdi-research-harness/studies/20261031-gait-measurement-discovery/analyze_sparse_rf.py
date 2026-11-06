"""Audit the supplementary RF search and summarize its held-out predictions."""
from common import *
from analyze_comparison import read_predictions, paired_ci
from verify_comparison import replay
from collections import Counter
import argparse


def main(name):
    folder = OUT/name/'sparse-rf'
    execution = json.loads((folder/'execution.json').read_text())
    assert execution['complete']
    protocol = json.loads((HERE/f'{name}-sparse-rf-protocol.json').read_text())
    for path,digest in protocol['sources'].items():
        assert sha(path)==digest
    d,meta = load(name)
    n = len(d['y'])
    p = pd.concat([pd.read_csv(path,float_precision='round_trip',dtype={'id':str}) for path in sorted(folder.glob('r*f*.predictions.csv.gz'))],ignore_index=True)
    original_predictions = read_predictions(name)
    counts = {'1':Counter(),'2':Counter(),'3':Counter(),'selected':Counter()}
    fit_replays = 0
    for repeat in range(3):
        for fold in range(5):
            prefix = folder/f'r{repeat}f{fold}'
            for path,digest in json.loads(prefix.with_suffix('.complete.json').read_text())['outputs'].items():
                assert sha(path)==digest
            ledger = json.loads(prefix.with_suffix('.ledger.json').read_text())
            original_path = OUT/name/f'r{repeat}f{fold}.ledger.json'
            assert sha(original_path)==ledger['original_ledger_sha256']
            prior = json.loads(original_path.read_text(encoding='utf-8'))
            first = min((c for c in prior['candidates'] if c['category']=='single' and c['model']=='RF'),key=lambda c:(c['inner_mae'],c['variant']))
            assert ledger['path'][0]['columns']==first['columns'] and ledger['path'][0]['inner_mae']==first['inner_mae']
            for size in [2,3]:
                prev = set(ledger['path'][size-2]['columns'])
                options = [s for s in ledger['search'] if s['size']==size]
                assert len(options)==len(meta['features'])-len(prev)
                assert all(prev.issubset(s['columns']) for s in options)
                best = min(options,key=lambda c:(c['inner_mae'],c['columns']))
                assert best==ledger['path'][size-1]
            chosen = min(ledger['path'],key=lambda c:(c['inner_mae'],c['size'],c['columns']))
            assert chosen==ledger['selected']
            cell = p[(p['repeat']==repeat)&(p.fold==fold)]
            basecell = original_predictions[(original_predictions['repeat']==repeat)&(original_predictions.fold==fold)&(original_predictions.variant==first['variant'])].sort_values('sample')
            np.testing.assert_allclose(cell[cell.variant=='RF_forward_1'].sort_values('sample').prediction,basecell.prediction,atol=1e-10,rtol=0)
            for step in ledger['path']:
                counts[str(step['size'])].update([' + '.join(meta['features'][i] for i in step['columns'])])
            counts['selected'].update([' + '.join(meta['features'][i] for i in chosen['columns'])])
            selected = cell[cell.variant=='RF_selected_up_to3'].sort_values('sample')
            reference = cell[cell.variant==f'RF_forward_{chosen["size"]}'].sort_values('sample')
            np.testing.assert_array_equal(selected.prediction,reference.prediction)
            for variant,g in cell.groupby('variant'):
                idx = g['sample'].to_numpy(int)
                assert sorted(idx)==prior['test'] and len(idx)==len(set(idx))
                np.testing.assert_array_equal(g.id.astype(str),d['ids'][idx])
                np.testing.assert_array_equal(g.target,d['y'][idx])
                assert np.isfinite(g.prediction).all()
            if repeat==0 and fold==0:
                for step in ledger['path'][1:]:
                    z = d['x'][:,step['columns']]
                    errors = []
                    for split in prior['inner']:
                        a,b = np.array(split['train']),np.array(split['validation'])
                        v = replay('RF',z,d['y'],a,b,prior['seed'],n)
                        errors.extend(abs(v-d['y'][b]).tolist())
                        fit_replays += 1
                    assert abs(np.mean(errors)-step['inner_mae'])<1e-10
                    v = replay('RF',z,d['y'],np.array(prior['train']),np.array(prior['test']),prior['seed'],n)
                    saved = cell[cell.variant==f'RF_forward_{step["size"]}'].sort_values('sample')
                    np.testing.assert_allclose(v,saved.prediction,atol=1e-9,rtol=0)
                    fit_replays += 1
    rows,errors = [],{}
    for variant,g in p.groupby('variant'):
        values = []
        for repeat,s in g.groupby('repeat'):
            assert len(s)==s['sample'].nunique()==n
            values.append(float(abs(s.prediction-s.target).mean()))
        assert len(values)==3
        rows.append({'variant':variant,'MAE':float(np.mean(values)),'MAE_min':min(values),'MAE_max':max(values)})
        errors[variant] = g.assign(error=abs(g.prediction-g.target)).groupby('sample').error.mean().sort_index().to_numpy()
    old = original_predictions.assign(error=abs(original_predictions.prediction-original_predictions.target))
    full_errors = old[old.variant=='selected_family'].groupby('sample').error.mean().sort_index().to_numpy()
    result = {'dataset':name,'people':n,'metrics':rows,
        'choices':{key:[{'features':k,'outer_cells':v} for k,v in count.most_common()] for key,count in counts.items()},
        'up_to3_minus_single':paired_ci(errors['RF_selected_up_to3']-errors['RF_forward_1']),
        'up_to3_minus_selected_family':paired_ci(errors['RF_selected_up_to3']-full_errors),
        'limits':'Exploratory extension after initial analysis; same reused development people. Greedy forward search, not exhaustive all triples. Paired intervals conditional on fitted models.'}
    save(HERE/f'{name}-sparse-rf-summary.json',result)
    pd.DataFrame(rows).to_csv(HERE/f'{name}-sparse-rf-metrics.csv',index=False)
    save(HERE/f'{name}-sparse-rf-verification.json',{'passed':True,'outer_cells':15,'source_hashes_checked':True,'all_nested_choices_and_predicted_people_checked':True,'representative_fit_replays':fit_replays,'predictions_rows':len(p),'summary_sha256':sha(HERE/f'{name}-sparse-rf-summary.json')})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['mobilise_sppb','mobilise_tug'])
    main(parser.parse_args().dataset)
