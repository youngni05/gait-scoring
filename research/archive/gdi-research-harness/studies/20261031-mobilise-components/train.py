"""Person-held-out component prediction; preserve the existing bouts and splits."""
import importlib.util, itertools, json, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
FACTORIAL = HERE.parent / '20261030-mobilise-factorial'
spec = importlib.util.spec_from_file_location('factorial_reference', FACTORIAL / 'train.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
ref = old.ref
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

SEEDS = old.SEEDS
EPOCHS = old.EPOCHS
LEAVES = [5, 20, 50]
COMPONENTS = ['gait', 'balance', 'chair']
LABELS = ['4mwt_speed_score', 'balance_score', '5crt_score']
SOURCE = HERE.parents[2] / 'outputs/sppb-tug-candidates-20261030/All_cohorts_dataset.csv'
KEY = ['band', 'k', 'seed', 'fold', 'draw', 'participantid', 'visit_number']

def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')

def load_data():
    data, arrays, _ = old.load_data()
    labels = pd.read_csv(SOURCE, usecols=['participantid', 'visit.number'] + LABELS)
    labels = labels.rename(columns={'visit.number': 'visit_number'})
    data = data[data.sppb_ok].merge(labels, on=['participantid', 'visit_number'], validate='many_to_one')
    for label in LABELS:
        assert data[label].dropna().between(0, 4).all()
        assert data[label].dropna().mod(1).eq(0).all()
    complete = data[LABELS].notna().all(axis=1)
    assert data.loc[complete, LABELS].sum(axis=1).eq(data.loc[complete, 'total_sppb_score']).all()
    return data, arrays['B']

def median(y, w):
    order = np.argsort(y, kind='stable')
    return float(y[order][min(np.searchsorted(np.cumsum(w[order]), w.sum() / 2), len(y)-1)])

def conditional_baselines(data):
    visits = data[data.draw.eq(0) & data.band.eq('long') & data.k.eq(5)].copy()
    assert len(visits) == 7356 and visits.participantid.nunique() == 2174
    visits[['participantid', 'visit_number', 'fold', 'cohort', 'total_sppb_score'] + LABELS].to_csv(HERE/'labels.csv', index=False)
    rows, ledgers = [], []
    for fold in range(5):
        tr = visits[visits.fold.ne(fold)].copy(); te = visits[visits.fold.eq(fold)].copy()
        assert not set(tr.participantid) & set(te.participantid)
        # A direct median of the other-two sum is also provided, to avoid a weak
        # sum-of-marginal-medians comparator for the remaining-score endpoint.
        tr['remaining'] = tr[LABELS[1:]].sum(axis=1, min_count=2)
        ledger = {'fold': fold, 'train_people': sorted(map(int, tr.participantid.unique())), 'global': {}, 'lookup': {}}
        for name, label in list(zip(COMPONENTS, LABELS)) + [('remaining', 'remaining'), ('total', 'total_sppb_score')]:
            eligible = tr[tr[label].notna()]
            value = median(eligible[label].to_numpy(), ref.weights(eligible.participantid.to_numpy()))
            te['constant_' + name] = value
            ledger['global'][name] = value
            if name not in ['gait', 'total']:
                mapping = {}
                for score in range(5):
                    group = eligible[eligible[LABELS[0]].eq(score)]
                    mapping[score] = median(group[label].to_numpy(), ref.weights(group.participantid.to_numpy())) if len(group) else value
                te['lookup_' + name] = te[LABELS[0]].map(mapping)
                ledger['lookup'][name] = mapping
        te['lookup_gait'] = te[LABELS[0]]
        te['lookup_total'] = te.lookup_gait + te.lookup_balance + te.lookup_chair
        te['lookup_total_direct_remaining'] = te.lookup_gait + te.lookup_remaining
        rows.append(te[['participantid', 'visit_number', 'fold'] + [c for c in te if c.startswith(('constant_', 'lookup_'))]])
        ledgers.append(ledger)
    pd.concat(rows, ignore_index=True).to_csv(HERE/'baselines.csv', index=False)
    save('baseline-ledgers.json', ledgers)
    save('label-audit.json', {'people': 2174, 'visits': len(visits), 'missing': {c: int(visits[c].isna().sum()) for c in LABELS}, 'complete_components': int(visits[LABELS].notna().all(axis=1).sum()), 'no_visit_excluded': True})

class MultiDeepSet(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(6, width), torch.nn.ReLU(), torch.nn.Linear(width, width//2), torch.nn.ReLU(), torch.nn.Linear(width//2, 3))
    def forward(self, x):
        h = self.layers[1](self.layers[0](x)).mean(dim=1)
        return self.layers[4](self.layers[3](self.layers[2](h)))

def ds_predict(model, bundle, x):
    with torch.no_grad():
        pred = model(torch.tensor((x-bundle['x_mean'])/bundle['x_std'], dtype=torch.float32)).numpy()
    return pred * bundle['y_std'] + bundle['y_mean']

def ds_fit(train, width, epochs, seed, arrays, k, valid=None):
    w = ref.weights(train.participantid.to_numpy())
    x = arrays[train.bag_id.to_numpy(), :k]
    flat = x.reshape(-1, 6); fw = np.repeat(w/k, k)
    xm = np.average(flat, axis=0, weights=fw)
    xs = np.maximum(np.sqrt(np.average((flat-xm)**2, axis=0, weights=fw)), 1e-8)
    y = train[LABELS].to_numpy(); mask = np.isfinite(y)
    cw = np.zeros_like(y); ym = np.zeros(3); ys = np.ones(3)
    for j in range(3):
        cw[mask[:, j], j] = ref.weights(train.loc[mask[:, j], 'participantid'].to_numpy())
        ym[j] = np.dot(cw[mask[:, j], j], y[mask[:, j], j])
        ys[j] = max(np.sqrt(np.dot(cw[mask[:, j], j], (y[mask[:, j], j]-ym[j])**2)), 1e-8)
    xt = torch.tensor((x-xm)/xs, dtype=torch.float32)
    yt = torch.tensor(np.where(mask, (y-ym)/ys, 0), dtype=torch.float32)
    wt = torch.tensor(cw, dtype=torch.float32)
    torch.manual_seed(seed); model = MultiDeepSet(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.001)
    bundle = {'x_mean': xm, 'x_std': xs, 'y_mean': ym, 'y_std': ys}
    curve = []
    if valid is not None:
        vx = arrays[valid.bag_id.to_numpy(), :k]
        vw = ref.weights(valid.participantid.to_numpy()); vy = valid.total_sppb_score.to_numpy()
    for epoch in range(1, epochs+1):
        optimizer.zero_grad()
        loss = (torch.nn.functional.smooth_l1_loss(model(xt), yt, beta=.5, reduction='none') * wt).sum()/3
        assert torch.isfinite(loss)
        loss.backward(); optimizer.step()
        if valid is not None and epoch % 10 == 0:
            prediction = ds_predict(model, bundle, vx).sum(axis=1)
            curve.append(float(np.dot(vw, abs(prediction-vy))))
    for j, index in enumerate([0, 2, 4]):
        bundle[f'w{j}'] = model.layers[index].weight.detach().numpy()
        bundle[f'b{j}'] = model.layers[index].bias.detach().numpy()
    return model, bundle, curve

def rf_model(leaf, seed):
    return RandomForestRegressor(n_estimators=128, min_samples_leaf=leaf, max_features=1., random_state=seed, n_jobs=1)

def initialize():
    global DATA, ARRAYS, LIMITS
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    LIMITS = threadpool_limits(limits=1)
    DATA, ARRAYS = load_data()

def task_data(band, k, fold, seed):
    cell = DATA[DATA.band.eq(band) & DATA.k.eq(k)].sort_values(['draw', 'participantid', 'visit_number'])
    tr = cell[cell.draw.eq(0) & cell.fold.ne(fold)].reset_index(drop=True)
    te = cell[cell.fold.eq(fold)].copy()
    tag = f'sppb-{band}-{k}-f{fold}-s{seed}'
    ledger = json.loads((old.REFERENCE/'folds'/f'{tag}.json').read_text())
    assert set(tr.participantid) == set(ledger['train_people'])
    assert set(te.participantid) == set(ledger['test_people'])
    assert not set(tr.participantid) & set(te.participantid)
    splits = []
    for ids in ledger['inner_people']:
        a = tr[tr.participantid.isin(ids['train'])]; b = tr[tr.participantid.isin(ids['valid'])]
        assert not set(a.participantid) & set(b.participantid)
        assert set(a.participantid) | set(b.participantid) == set(tr.participantid)
        splits.append((a, b))
    return tr, te, ledger, splits

def work(task):
    band, k, fold, seed, kind = task
    tag = f'{band}-{k}-f{fold}-s{seed}-{kind}'
    if (HERE/'folds'/f'{tag}.json').exists(): return tag, 'cached'
    start = time.time(); tr, te, old_ledger, splits = task_data(band, k, fold, seed)
    out = te[['participantid', 'visit_number', 'draw', 'fold']].copy()
    out['band'] = band; out['k'] = k; out['seed'] = seed
    ledger = {'band': band, 'k': k, 'fold': fold, 'seed': seed, 'kind': kind, 'train_people': old_ledger['train_people'], 'test_people': old_ledger['test_people'], 'inner_people': old_ledger['inner_people']}
    if kind == 'DS':
        width = old_ledger['selected']['600']['width']; curves, sizes = [], []
        for inner, (a, b) in enumerate(splits):
            _, _, curve = ds_fit(a, width, 600, seed+fold*10+inner, ARRAYS, k, b)
            curves.append(curve); sizes.append(int(b.participantid.nunique()))
        combined = np.average(curves, axis=0, weights=sizes)
        selected = int(np.argmin(combined)); epochs = EPOCHS[selected]
        model, bundle, _ = ds_fit(tr, width, epochs, seed+fold*10+9, ARRAYS, k)
        np.savez_compressed(HERE/'models'/f'{tag}.npz', **bundle)
        predictions = ds_predict(model, bundle, ARRAYS[te.bag_id.to_numpy(), :k])
        for j, c in enumerate(COMPONENTS): out['DS_'+c] = predictions[:, j]
        out['DS_total'] = predictions.sum(axis=1)
        ledger.update(width=width, epochs=epochs, inner_curves=curves, inner_sizes=sizes, combined_curve=combined.tolist(), parameters=sum(p.numel() for p in model.parameters()))
    else:
        trials, chosen, models = {}, {}, {}
        # Also train an RF total-score head on the identical inputs. This separates
        # the effect of component decomposition from changing model family.
        for c, label in list(zip(COMPONENTS, LABELS)) + [('total_direct', 'total_sppb_score')]:
            scores = []
            for leaf in LEAVES:
                fold_scores, sizes = [], []
                for inner, (a, b) in enumerate(splits):
                    a = a[a[label].notna()]; b = b[b[label].notna()]
                    model = rf_model(leaf, seed+fold*10+inner)
                    model.fit(np.median(ARRAYS[a.bag_id.to_numpy(), :k], axis=1), a[label], sample_weight=ref.weights(a.participantid.to_numpy()))
                    pred = model.predict(np.median(ARRAYS[b.bag_id.to_numpy(), :k], axis=1))
                    fold_scores.append(float(np.dot(ref.weights(b.participantid.to_numpy()), abs(pred-b[label].to_numpy()))))
                    sizes.append(int(b.participantid.nunique()))
                scores.append({'leaf': leaf, 'inner_mae': float(np.average(fold_scores, weights=sizes)), 'fold_scores': fold_scores, 'sizes': sizes})
            best = min(scores, key=lambda row: row['inner_mae'])
            eligible = tr[tr[label].notna()]
            model = rf_model(best['leaf'], seed+fold*10+9)
            model.fit(np.median(ARRAYS[eligible.bag_id.to_numpy(), :k], axis=1), eligible[label], sample_weight=ref.weights(eligible.participantid.to_numpy()))
            out['RF_'+c] = model.predict(np.median(ARRAYS[te.bag_id.to_numpy(), :k], axis=1))
            trials[c] = scores; chosen[c] = best['leaf']; models[c] = model
        out['RF_total'] = out[['RF_'+c for c in COMPONENTS]].sum(axis=1)
        joblib.dump(models, HERE/'models'/f'{tag}.joblib', compress=3)
        ledger.update(trials=trials, selected=chosen)
    out.to_csv(HERE/'folds'/f'{tag}.csv.gz', index=False)
    ledger['elapsed_seconds'] = time.time()-start
    save(f'folds/{tag}.json', ledger)
    return tag, round(time.time()-start, 1)

def prepare():
    for directory in ['models', 'folds']: (HERE/directory).mkdir(exist_ok=True)
    data, arrays = load_data(); conditional_baselines(data)
    paths = [Path(__file__), SOURCE, FACTORIAL/'train.py', FACTORIAL/'predictions.csv.gz', FACTORIAL/'verification.json', old.REFERENCE/'train.py', ref.HERE/'train.py', old.BASE/'features.csv', old.BASE/'selected-bouts.csv', old.BASE/'person-folds.csv', FACTORIAL/'audited-bouts.csv']
    protocol = {'source_hashes': {str(p): ref.sha(p) for p in paths}, 'seeds': SEEDS, 'people': 2174, 'visits': 7356, 'bands_seconds': {'short': '[4,10)', 'long': '[30,60)'}, 'k': [1,3,5], 'features': old.FEATURES+['duration','turn_rate_per_second'], 'inputs': 'same expanded six features and original selected bouts; no component score, cohort, clinical metadata or review flags in RF/DS inputs', 'splits': 'reuse exact existing outer 5 person folds and inner 3 person folds; every visit of a person stays together', 'draws': 'train draw0; evaluate draw0/1/2 separately, never average draws or seed predictions', 'baselines': 'outer-train person-weighted marginal medians; gait-score conditional medians for balance/chair and separately their sum; no random initialization, identical baseline predictions aligned to each seed and condition', 'missing': 'keep all visits; ignore only the unavailable supervised component in its loss/metric; never infer missing labels from total. Lookup predictions unavailable at 5 visits with missing observed gait score; matched comparisons explicitly restrict to available labels', 'DS': {'architecture': 'same six-input mean-pool encoder and hidden width as factorial B; three linear outputs instead of one', 'width': 'reuse per-fold/seed/condition internally selected original width', 'loss': 'mean of 3 person-weighted SmoothL1 beta .5 component losses; each observed component gets equal-person training weights and training-only standardization', 'optimizer': 'AdamW lr .01 weight_decay .001, full batch', 'epochs': EPOCHS, 'selection': 'minimum total-score inner person MAE of summed component predictions; earlier epoch breaks ties'}, 'RF': {'input': 'per-feature median of the same k bouts, six values', 'n_estimators': 128, 'max_features': 1., 'leaf_grid': LEAVES, 'selection': 'each component separately minimizes its inner person MAE; an additional direct-total RF uses the same grid and input'}, 'seed_schedule': 'inner seed+fold*10+inner, outer seed+fold*10+9', 'primary_predictions': 'raw continuous regression, no clipping or rounding, consistent with original experiments; bounded predictions are a separately marked sensitivity analysis', 'references': 'existing factorial A four-feature and B six-feature direct-total Deep Sets; B is the input-matched main comparator', 'metrics': 'equal-person MAE, bias, RMSE, weighted visit absolute-error P90; reweight within each target/subgroup', 'within_gait': 'report each observed gait score 0..4 separately for balance, chair and remaining sum; compare rich-input models against gait-only lookup on the same eligible rows', 'low_sppb': 'actual total<=3, unchanged from prior experiments; success requires overall MAE nonincrease plus lower tail MAE, separately tail P90 and all5-seed stability', 'uncertainty': 'paired person bootstrap of five-seed mean errors, conditional on fixed OOF fits; reused exploratory evaluation folds, not independent final validation', 'device': 'CPU, six single-thread worker processes', 'torch': torch.__version__}
    if (HERE/'protocol.json').exists(): assert json.loads((HERE/'protocol.json').read_text()) == protocol
    save('protocol.json', protocol)
    print('PREPARED', len(data), arrays.shape, flush=True)

def assemble():
    ds = pd.concat([pd.read_csv(p) for p in sorted((HERE/'folds').glob('*-DS.csv.gz'))], ignore_index=True)
    rf = pd.concat([pd.read_csv(p) for p in sorted((HERE/'folds').glob('*-RF.csv.gz'))], ignore_index=True)
    assert len(list((HERE/'folds').glob('*-DS.json'))) == len(list((HERE/'folds').glob('*-RF.json'))) == 150
    p = ds.merge(rf, on=KEY, validate='one_to_one')
    assert len(p) == len(ds) == len(rf) == 7356*6*3*5
    previous = pd.read_csv(FACTORIAL/'predictions.csv.gz'); previous = previous[previous.target.eq('sppb')]
    p = p.merge(previous[KEY+['actual','A','B']], on=KEY, validate='one_to_one')
    p = p.merge(pd.read_csv(HERE/'labels.csv'), on=['participantid','visit_number','fold'], validate='many_to_one')
    np.testing.assert_array_equal(p.actual, p.total_sppb_score)
    p = p.merge(pd.read_csv(HERE/'baselines.csv'), on=['participantid','visit_number','fold'], validate='many_to_one')
    p.to_csv(HERE/'predictions.csv.gz', index=False)
    save('completion.json', {'DS_outer_models':150, 'RF_outer_forests':600, 'DS_inner_fits':450, 'RF_inner_fits':5400, 'predictions_per_method':len(p), 'people':2174, 'visits':7356, 'predictions_sha256':ref.sha(HERE/'predictions.csv.gz')})

def run():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--smoke', action='store_true'); args = parser.parse_args()
    if args.prepare:
        prepare(); return
    if args.smoke:
        initialize(); tr, te, ledger, splits = task_data('long',5,0,SEEDS[0])
        start=time.time(); model,bundle,curve=ds_fit(splits[0][0],32,20,SEEDS[0],ARRAYS,5,splits[0][1])
        pred=ds_predict(model,bundle,ARRAYS[te.bag_id.to_numpy(),:5]); assert pred.shape==(len(te),3) and np.isfinite(pred).all()
        start_rf=time.time(); m=rf_model(20,SEEDS[0]); m.fit(np.median(ARRAYS[tr.bag_id.to_numpy(),:5],axis=1),tr.balance_score,sample_weight=ref.weights(tr.participantid.to_numpy()))
        save('preflight.json',{'output_shape':list(pred.shape),'DS_20_epochs_seconds':start_rf-start,'RF_fit_seconds':time.time()-start_rf,'finite':True,'missing_component_mask':True})
        print(json.dumps(json.loads((HERE/'preflight.json').read_text())),flush=True); return
    protocol=json.loads((HERE/'protocol.json').read_text())
    for path,digest in protocol['source_hashes'].items(): assert ref.sha(path)==digest, path
    started=time.time()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[name]='1'
    tasks=list(itertools.product(['short','long'],[1,3,5],range(5),SEEDS,['DS','RF']))
    with ProcessPoolExecutor(max_workers=6, initializer=initialize) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,future in enumerate(as_completed(futures),1):
            tag,seconds=future.result()
            print(f'{n}/300 finished; {tag}; {seconds}s; elapsed {time.time()-started:.0f}s',flush=True)
    assemble(); print('ALL_TRAINING_FINISHED',flush=True)

if __name__=='__main__': run()
