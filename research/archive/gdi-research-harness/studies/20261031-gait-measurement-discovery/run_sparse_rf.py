"""Exploratory RF forward selection, reusing frozen training-only single losses."""
from common import *
import argparse
import time


def run(name):
    assert name in ['mobilise_sppb','mobilise_tug']
    started = time.time()
    folder = OUT/name/'sparse-rf'
    folder.mkdir(exist_ok=True)
    d,meta = load(name)
    x,y = d['x'],d['y']
    n,p = x.shape
    protocol = {'status':'Exploratory extension after initial fixed-candidate analysis, not independent confirmation.',
        'data':'same original input, all15 outer cells and exact inner3folds; no outer test used in choices',
        'method':'Start with the RF single feature with minimum saved inner MAE. Greedily add one feature at a time up to3 using original128tree RF settings. Choose among1/2/3 using inner MAE.',
        'sources':{str(HERE/'run_sparse_rf.py'):sha(HERE/'run_sparse_rf.py'),str(HERE/'common.py'):sha(HERE/'common.py'),str(OUT/name/'input.npz'):sha(OUT/name/'input.npz')}}
    protocol_path = HERE/f'{name}-sparse-rf-protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        save(protocol_path,protocol)
    for repeat in range(3):
        for fold in range(5):
            prefix = folder/f'r{repeat}f{fold}'
            marker = prefix.with_suffix('.complete.json')
            if marker.exists():
                for path,digest in json.loads(marker.read_text())['outputs'].items():
                    assert sha(path)==digest
                continue
            original = OUT/name/f'r{repeat}f{fold}.ledger.json'
            prior = json.loads(original.read_text(encoding='utf-8'))
            train,test = np.array(prior['train']),np.array(prior['test'])
            options = [c for c in prior['candidates'] if c['category']=='single' and c['model']=='RF']
            first = min(options,key=lambda c:(c['inner_mae'],c['variant']))
            chosen = first['columns'][:]
            path = [{'size':1,'columns':chosen[:],'inner_mae':first['inner_mae']}]
            search = []
            for size in [2,3]:
                choices = []
                for new in range(p):
                    if new in chosen:
                        continue
                    cols = sorted(chosen+[new])
                    z = x[:,cols]
                    errors = []
                    for split in prior['inner']:
                        a,b = np.array(split['train']),np.array(split['validation'])
                        pred = fit_predict('RF',z,y,a,b,prior['seed'],n)
                        errors.extend(abs(pred-y[b]).tolist())
                    item = {'size':size,'columns':cols,'inner_mae':float(np.mean(errors))}
                    choices.append(item)
                    search.append(item)
                best = min(choices,key=lambda c:(c['inner_mae'],c['columns']))
                chosen = best['columns'][:]
                path.append(best)
            selected = min(path,key=lambda c:(c['inner_mae'],c['size'],c['columns']))
            predictions = []
            for step in path:
                pred = fit_predict('RF',x[:,step['columns']],y,train,test,prior['seed'],n)
                variants = [f'RF_forward_{step["size"]}']
                if step is selected:
                    variants.append('RF_selected_up_to3')
                for variant in variants:
                    predictions.append(pd.DataFrame({'repeat':repeat,'fold':fold,'sample':test,'id':d['ids'][test], 'target':y[test], 'prediction':pred,'variant':variant}))
            pred_path = prefix.with_suffix('.predictions.csv.gz')
            ledger_path = prefix.with_suffix('.ledger.json')
            pd.concat(predictions,ignore_index=True).to_csv(pred_path,index=False)
            save(ledger_path,{'original_ledger_sha256':sha(original),'repeat':repeat,'fold':fold,'path':path,'search':search,'selected':selected,
                'feature_names':{str(step['size']):[meta['features'][j] for j in step['columns']] for step in path}})
            save(marker,{'outputs':{str(pred_path):sha(pred_path),str(ledger_path):sha(ledger_path)}})
            print(f'{name} sparseRF repeat={repeat} fold={fold} selected={len(selected["columns"])} seconds={time.time()-started:.1f}',flush=True)
    save(folder/'execution.json',{'complete':True,'outer_cells':15,'protocol_sha256':sha(protocol_path),'seconds':time.time()-started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',choices=['mobilise_sppb','mobilise_tug'])
    run(parser.parse_args().dataset)
