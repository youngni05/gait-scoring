"""Verify returned CNN records and independently compare matched person-level OOF errors."""
from pathlib import Path
import csv
import hashlib
import json
import math
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
OUT = HERE / 'analysis'
RECEIVED = HERE / 'received'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csvsave(p, rows):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    RECEIVED.mkdir(exist_ok=True)
    archive = HERE / 'main-review.tar.gz'
    with tarfile.open(archive) as t:
        members = t.getmembers()
        assert len(members) == len(set(m.name for m in members)) and sum(m.size for m in members) < 30_000_000
        for m in members:
            assert m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts
            dest = (RECEIVED / m.name).resolve()
            assert dest.is_relative_to(RECEIVED.resolve())
            content = t.extractfile(m).read()
            if dest.exists():
                assert dest.read_bytes() == content, 'Existing received artifact changed'
            else:
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(content)
    remote = RECEIVED / 'results'
    manifest = read(BUNDLE / 'manifest.json')
    assert sha(RECEIVED / 'bundle/manifest.json') == sha(BUNDLE / 'manifest.json')
    for name, expected in manifest.items():
        assert sha(BUNDLE / name) == expected, name
    for name in ('run.py', 'model.py', 'protocol.json'):
        assert sha(RECEIVED / 'bundle' / name) == manifest[name]
    c = read(BUNDLE / 'protocol.json')
    with np.load(BUNDLE / 'data.npz', allow_pickle=False) as z:
        d = {k: z[k].copy() for k in ('y', 'person_id', 'video_id', 'fold', 'cohort')}
    with np.load(HERE.parent / 'expansion695/train_only.npz', allow_pickle=False) as z:
        for k in d:
            assert np.array_equal(d[k], z[k])
    y, ids = d['y'], d['person_id']
    people = np.unique(ids)
    def identity(outer, inner, epochs, purpose):
        te = d['fold'] == outer
        va = np.zeros(len(ids), dtype=bool) if inner is None else d['fold'] == inner
        tr = ~(te | va)
        a, b, e = (np.unique(ids[m]).tolist() for m in (tr, va, te))
        assert not (set(a)&set(b) or set(a)&set(e) or set(b)&set(e))
        return {'bundle_sha256': sha(BUNDLE / 'manifest.json'), 'purpose': purpose,
                'outer_fold': outer, 'inner_fold': inner, 'epochs': epochs,
                'fit_seed': c['seed'] + outer*100 + (50 if inner is None else inner),
                'train_people': a, 'validation_people': b, 'outer_evaluation_people': e}
    cnn = np.full_like(y, np.nan)
    curves = []; steps = 0; training_seconds = 0.; folds = []
    for f in range(5):
        folder = remote / ('fold%d' % f)
        done = read(folder / 'complete.json')
        assert done['device'] == 'cuda' and done['torch'] == '2.4.0+cu121'
        assert sha(folder / 'selection.json') == done['selection_sha256']
        assert sha(folder / 'predictions.npz') == done['prediction_sha256']
        inner_curves = []
        for j in range(5):
            if j == f: continue
            q = folder / ('inner%d' % j)
            complete = read(q / 'fit_complete.json')
            assert complete['identity'] == identity(f, j, 100, 'inner_epoch_selection') == read(q / 'identity.json')
            assert complete['epochs'] == 100 and sha(q / 'curve.json') == complete['curve_sha256']
            hist = read(q / 'curve.json')
            assert [h['epoch'] for h in hist] == list(range(1, 101))
            for h in hist:
                assert all(np.isfinite(h[k]) for k in ('loss','scaled_mse','train_person_mae','validation_person_mae','seconds'))
                assert math.isclose(h['lr'], c['learning_rate'] * c['lr_drop_factor'] ** (h['epoch']//10), rel_tol=1e-12)
            expected_steps = 100 * math.ceil(int(((d['fold'] != f)&(d['fold'] != j)).sum()) / 32)
            assert complete['optimizer_steps'] == expected_steps
            steps += expected_steps; training_seconds += sum(h['seconds'] for h in hist)
            inner_curves.append(hist)
        mean_val = np.mean([[h['validation_person_mae'] for h in hist] for hist in inner_curves], axis=0)
        mean_train = np.mean([[h['train_person_mae'] for h in hist] for hist in inner_curves], axis=0)
        epoch = int(np.argmin(mean_val)) + 1
        selection = read(folder / 'selection.json')
        assert selection['selected_epoch'] == done['selected_epoch'] == epoch
        np.testing.assert_allclose(mean_val, selection['inner_mean_person_mae_by_epoch'], rtol=0, atol=1e-12)
        assert not selection['outer_evaluation_used_for_selection']
        fit = read(folder / 'refit/fit_complete.json')
        assert fit['identity'] == done['identity'] == identity(f, None, epoch, 'outer_refit') == read(folder / 'refit/identity.json')
        assert fit['checkpoint_sha256'] == done['checkpoint_sha256']
        assert sha(folder / 'refit/curve.json') == fit['curve_sha256']
        refit = read(folder / 'refit/curve.json')
        assert [h['epoch'] for h in refit] == list(range(1, epoch+1)) and fit['epochs'] == epoch
        assert all('validation_person_mae' not in h for h in refit)
        expected_steps = epoch*math.ceil(int((d['fold'] != f).sum())/32)
        assert fit['optimizer_steps'] == expected_steps
        steps += expected_steps; training_seconds += sum(h['seconds'] for h in refit)
        with np.load(folder / 'predictions.npz', allow_pickle=False) as z:
            idx = np.flatnonzero(d['fold'] == f)
            assert np.array_equal(idx, z['row_index']) and z['predictions'].shape == (len(idx),1,2)
            for k in d: assert np.array_equal(z[k], d[k][idx]), k
            cnn[idx] = z['predictions'][:,0]
        for e in range(100):
            curves.append({'outer_fold':f,'epoch':e+1,'inner_train_mae':float(mean_train[e]),'inner_validation_mae':float(mean_val[e])})
        folds.append({'fold':f,'selected_epoch':epoch,'inner_train_at_selected':float(mean_train[epoch-1]),
                      'inner_validation_at_selected':float(mean_val[epoch-1]),
                      'validation_at_25':float(mean_val[24]),'validation_at_50':float(mean_val[49]),
                      'validation_at_75':float(mean_val[74]),'validation_at_100':float(mean_val[99]),
                      'refit_train_mae':refit[-1]['train_person_mae'],
                      'regularizer_loss_at_end':refit[-1]['loss']-refit[-1]['scaled_mse'],
                      'scaled_mse_at_end':refit[-1]['scaled_mse']})
    summary = read(remote / 'summary.json')
    assert summary['optimizer_steps'] == steps and summary['fits'] == 25
    assert summary['bundle_sha256'] == sha(BUNDLE / 'manifest.json') and sha(remote / 'oof_predictions.npz') == summary['oof_sha256']
    with np.load(remote / 'oof_predictions.npz', allow_pickle=False) as z:
        for k in d: assert np.array_equal(z[k], d[k])
        assert np.array_equal(cnn, z['predictions'][:,0])
    assert np.isfinite(cnn).all()
    with (remote/'predictions.csv').open(encoding='utf-8',newline='') as f:
        csvrows=list(csv.DictReader(f))
    assert len(csvrows)==2128
    for row, (i,j) in zip(csvrows, ((i,j) for i in range(1064) for j in range(2))):
        assert row['method']==c['name'] and int(row['seed'])==c['seed'] and row['role']=='outer_evaluation'
        assert row['person_id']==ids[i] and row['video_id']==d['video_id'][i] and int(row['fold'])==int(d['fold'][i])
        assert row['side']==['left','right'][j] and float(row['actual_gdi'])==y[i,j] and float(row['predicted_gdi'])==cnn[i,j]
    with np.load(BUNDLE / 'reference_predictions.npz', allow_pickle=False) as z:
        predictions = {k:z[k].copy() for k in z.files}
    cpu = HERE.parent.parent / '20261030-cpu-head-study/followup/expanded695-pls64-study'
    meta = read(cpu / 'oof.metadata.json')
    assert sha(cpu / 'oof_predictions.npz') == meta['predictions_sha256'] == read(BUNDLE / 'reference_metrics.json')['cpu_predictions_sha256']
    with np.load(cpu / 'oof_predictions.npz', allow_pickle=False) as z:
        for k in ('person_id','video_id','fold'): assert np.array_equal(z[k],d[k])
        for key, method in [('ridge','ridge'),('pls','pls64')]:
            assert np.array_equal(predictions[key],z['predictions'][z['methods'].tolist().index(method)])
    with np.load(HERE.parent / 'expansion695/analysis/gpu/last_stage_oof.npz',allow_pickle=False) as z:
        for k in ('person_id','video_id','fold'): assert np.array_equal(z[k],d[k])
        assert np.array_equal(predictions['motionbert_last_stage'],z['predictions'][:,0])
    predictions['cnn'] = cnn
    def aggregate(p, mask):
        local_ids = ids[mask]; yy = y[mask]; pp = p[mask]
        unique = np.unique(local_ids)
        def macro(v): return float(np.mean([v[local_ids==k].mean() for k in unique]))
        bins = np.abs(np.floor(pp/10)-np.floor(yy/10))
        return {'people':len(unique),'records':int(mask.sum()),'mae':macro(np.abs(pp-yy)),
                'bias':macro(pp-yy),'exact_bin_pct':100*macro(bins==0),'two_or_more_bin_pct':100*macro(bins>=2),
                'person_prediction_sd':float(np.std([pp[local_ids==k].mean() for k in unique])),
                'prediction_min':float(pp.min()),'prediction_max':float(pp.max())}
    metrics=[]; foldmetrics=[]; bands=[]
    for name,p in predictions.items():
        for group,mask in [('all695',np.ones(len(ids),bool)),('old175',d['cohort']==0),('new520',d['cohort']==1)]:
            metrics.append({'model':name,'group':group,**aggregate(p,mask)})
        for f in range(5): foldmetrics.append({'model':name,'fold':f,**aggregate(p,d['fold']==f)})
        for low in range(30,120,10):
            mask=(y>=low)&(y<low+10)
            local_people=[k for k in people if mask[ids==k].any()]
            if not local_people: continue
            err=p-y
            bands.append({'model':name,'score_band':str(low)+'-'+str(low+9),'people':len(local_people),
                          'sides_records':int(mask.sum()),
                          'mae':float(np.mean([np.abs(err[ids==k][mask[ids==k]]).mean() for k in local_people])),
                          'bias':float(np.mean([err[ids==k][mask[ids==k]].mean() for k in local_people]))})
    allcnn=next(m for m in metrics if m['model']=='cnn' and m['group']=='all695')
    for rec in summary['metrics']:
        m=next(m for m in metrics if m['model']=='cnn' and m['group']==rec['group'])
        for k in ('mae','bias','prediction_min','prediction_max'): assert abs(rec[k]-m[k])<1e-10
        assert abs(100*rec['exact_10point_bin_rate']-m['exact_bin_pct'])<1e-10
        assert abs(100*rec['two_or_more_bins_rate']-m['two_or_more_bin_pct'])<1e-10
    for rec in summary['fold_metrics']:
        m=next(m for m in foldmetrics if m['model']=='cnn' and m['fold']==rec['fold'])
        for k in ('mae','bias','prediction_min','prediction_max'): assert abs(rec[k]-m[k])<1e-10
    person_errors={name:np.array([np.abs(p[ids==k]-y[ids==k]).mean() for k in people]) for name,p in predictions.items()}
    draws=np.random.default_rng(20261029).integers(0,len(people),(10000,len(people)))
    comparisons=[]
    for name in predictions:
        if name=='cnn':continue
        delta=person_errors['cnn']-person_errors[name]
        fold_better=sum(person_errors['cnn'][np.array([d['fold'][ids==k][0]==f for k in people])].mean()<person_errors[name][np.array([d['fold'][ids==k][0]==f for k in people])].mean() for f in range(5))
        comparisons.append({'reference':name,'cnn_minus_reference_mae':float(delta.mean()),
                            'relative_error_increase_pct':100*float(delta.mean()/person_errors[name].mean()),
                            'exploratory_bootstrap95':np.percentile(delta[draws].mean(1),[2.5,97.5]).tolist(),
                            'people_cnn_better':int((delta<0).sum()),'people_cnn_worse':int((delta>0).sum()),
                            'folds_cnn_better':int(fold_better)})
    diagnostics=[]
    for f in range(5):
        p=cnn[d['fold']==f]
        for side in range(2):
            values,counts=np.unique(p[:,side],return_counts=True)
            diagnostics.append({'fold':f,'side':['left','right'][side], 'records':len(p),'unique_predictions':len(values),
                                'most_repeated_prediction':float(values[np.argmax(counts)]),'repeat_count':int(counts.max())})
    env=[read(p) for p in remote.glob('environment-*.json')]
    result={'metrics':metrics,'comparisons':comparisons,'training_diagnostics':folds,
            'duplicate_prediction_diagnostics':diagnostics,'optimizer_steps':steps,
            'sum_epoch_seconds_excludes_checkpoint_and_setup':training_seconds,'environment':env,
            'independent_final_evaluation':False,'bootstrap_note':'person resampling of fixed OOF predictions; no retraining or search correction'}
    save(OUT/'summary.json',result)
    for name,rows in [('metrics',metrics),('fold_metrics',foldmetrics),('score_bands',bands),('learning_curves',curves)]:
        csvsave(OUT/(name+'.csv'),rows)
    np.savez_compressed(OUT/'cnn_oof.npz',predictions=cnn[:,None,:],**d)
    proof={'passed':True,'archive_sha256':sha(archive),'bundle_sha256':sha(BUNDLE/'manifest.json'),
           'completed_fits':25,'inner_fits':20,'outer_refits':5,'real_gpu_optimizer_steps':steps,
           'matched_people':695,'matched_records':1064,'code_and_protocol_match':True,
           'all_identity_and_split_records_match':True,'all_prediction_and_curve_hashes_match':True,
           'epoch_selection_recomputed':True,'oof_and_fold_predictions_match':True,
           'metrics_independently_recomputed':True,'checkpoints_downloaded':False,
           'independent_model_reinference':False,'protected_final_people_used':False}
    save(OUT/'verification.json',proof)
    print(json.dumps({'metrics':[m for m in metrics if m['group']=='all695'],'comparisons':comparisons,
                      'training_diagnostics':folds,'duplicate_predictions':diagnostics,
                      'epoch_seconds':training_seconds,'verification':proof},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
