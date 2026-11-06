"""Independent sklearn pipelines for all nested fits and leakage audits."""
import run as a
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

d, old_heads, baseline, sources = a.diag.m.source_data()
s = a.diag.m.load_json(a.HERE / 'summary.json')
protocol = a.diag.m.load_json(a.HERE / 'protocol.json')
assert s['protocol_sha256'] == a.diag.m.digest(a.HERE / 'protocol.json') and s['source_hashes'] == sources
with np.load(a.HERE / 'fold_predictions.npz', allow_pickle=False) as z:
    cube = z['pca_ridge'].copy(); ridge = z['ridge'].copy()
    for key in ['person_id', 'video_id', 'fold']:
        assert np.array_equal(z[key], d[key])
with np.load(a.HERE / 'oof_predictions.npz', allow_pickle=False) as z:
    oof = z['predictions'].copy()
    assert np.array_equal(z['baseline'], baseline)
    assert np.array_equal(oof, cube[d['fold'], np.arange(203)])
assert float(np.max(abs(ridge[d['fold'], np.arange(203)] - baseline))) < 1e-10
independent_rows, audit_rows = [], []
inner_max, outer_max, replay_max = 0.0, 0.0, 0.0
with threadpool_limits(limits=2):
    for fold in range(5):
        outer_tr = np.flatnonzero(d['fold'] != fold)
        outer_te = np.flatnonzero(d['fold'] == fold)
        with np.load(a.HERE / f'models/fold{fold}_inner.npz', allow_pickle=False) as z:
            saved = {k: z[k].copy() for k in z.files}
        assert np.array_equal(saved['outer_train_indices'], outer_tr)
        inner_oof = np.full((4, len(outer_tr), 2), np.nan)
        validation_coverage = []
        for inner in range(4):
            prefix = f'inner{inner}__'
            tr, va = saved[prefix + 'train_indices'], saved[prefix + 'validation_indices']
            assert set(tr) | set(va) == set(outer_tr) and not set(tr) & set(va)
            assert not (set(tr) | set(va)) & set(outer_te)
            trpeople, vapeople, outerpeople = set(d['person_id'][tr]), set(d['person_id'][va]), set(d['person_id'][outer_te])
            assert len(trpeople) == 105 and len(vapeople) == 35
            assert not trpeople & vapeople and not (trpeople | vapeople) & outerpeople
            validation_coverage.extend(va.tolist())
            assignment_local = np.isin(outer_tr, va)
            assert np.all(saved['inner_assignment'][assignment_local] == inner)
            ref_scale = StandardScaler().fit(d['features'][tr, 0])
            assert np.allclose(ref_scale.mean_, saved[prefix + 'scaler_mean'], atol=1e-12, rtol=1e-12)
            assert np.allclose(ref_scale.scale_, saved[prefix + 'scaler_scale'], atol=1e-12, rtol=1e-12)
            components = saved[prefix + 'components']
            assert np.max(abs(components @ components.T - np.eye(64))) < 1e-10
            for ci, k in enumerate(protocol['candidate_components']):
                pipe = make_pipeline(StandardScaler(), PCA(n_components=k, whiten=False, svd_solver='full'),
                                     Ridge(alpha=1, solver='svd'))
                pipe.fit(d['features'][tr, 0], d['y'][tr])
                pred = pipe.predict(d['features'][va, 0])
                inner_oof[ci, assignment_local] = pred
                saved_pred = saved['inner_oof_predictions'][ci, assignment_local]
                error = float(np.max(abs(pred - saved_pred)))
                inner_max = max(inner_max, error)
                assert error < 1e-8
                z = (d['features'][va, 0] - saved[prefix + 'scaler_mean']) / saved[prefix + 'scaler_scale']
                replay = ((z - saved[prefix + 'pca_mean']) @ components[:k].T) @ saved[prefix + f'k{k}_coef'].T + saved[prefix + f'k{k}_intercept']
                assert np.max(abs(replay - saved_pred)) < 1e-10
                independent_rows.append({'outer_fold': fold, 'inner_fold': inner, 'components': k,
                                         'independent_prediction_max_error': error})
            audit_rows.append({'outer_fold': fold, 'inner_fold': inner, 'outer_held_absent': True,
                               'inner_people_disjoint': True, 'inner_scaler_refit_checked': True})
        assert sorted(validation_coverage) == outer_tr.tolist()
        scores = []
        for ci, k in enumerate(protocol['candidate_components']):
            person_errors = [abs(inner_oof[ci, d['person_id'][outer_tr] == pid] - d['y'][outer_tr][d['person_id'][outer_tr] == pid]).mean()
                             for pid in np.unique(d['person_id'][outer_tr])]
            score = float(np.mean(person_errors))
            stored_score = next(r['inner_front_person_MAE'] for r in s['inner_candidate_scores'] if r['outer_fold'] == fold and r['components'] == k)
            assert abs(score - stored_score) < 1e-9
            scores.append((k, score))
        best = min(score for k, score in scores)
        selected = min(k for k, score in scores if score <= best + 1e-12)
        assert selected == s['selected_dimensions'][fold]['selected_components']
        with np.load(a.HERE / f'models/fold{fold}_selected.npz', allow_pickle=False) as z:
            model = {k: z[k].copy() for k in z.files}
        assert int(model['selected_k']) == selected
        assert np.array_equal(model['train_indices'], outer_tr) and np.array_equal(model['evaluation_indices'], outer_te)
        # Independent full pipeline refitted AFTER the training-only dimension selection.
        pipe = make_pipeline(StandardScaler(), PCA(n_components=selected, whiten=False, svd_solver='full'),
                             Ridge(alpha=1, solver='svd')).fit(d['features'][outer_tr, 0], d['y'][outer_tr])
        pred = pipe.predict(d['features'].reshape(-1, 512)).reshape(203, 2, 2)
        error = float(np.max(abs(pred - cube[fold])))
        assert error < 1e-8
        outer_max = max(outer_max, error)
        z = (d['features'] - model['scaler_mean']) / model['scaler_scale']
        replay = z @ model['combined_coef'].T + model['combined_intercept']
        error = float(np.max(abs(replay - cube[fold])))
        assert error < 1e-10
        replay_max = max(replay_max, error)
        assert abs(pipe[1].explained_variance_ratio_.sum() - float(model['retained_feature_variance_ratio'])) < 1e-10

# Independently check all train/held metrics with explicit record-side weights.
for name, values in [('ridge', ridge), ('pca_ridge_nested', cube)]:
    for role in ['train_people', 'held_people']:
        views = 4 if role == 'train_people' else 1
        weights = np.zeros((5, 203, 2))
        for fold in range(5):
            chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
            for i in np.flatnonzero(chosen):
                n = np.count_nonzero(d['person_id'] == d['person_id'][i])
                weights[fold, i] = 1 / (175 * views * n * 2)
        assert abs(weights.sum() - 1) < 1e-12
        for wi, window in enumerate(['front', 'back']):
            error = values[:, :, wi] - d['y'][None]
            gap = abs(np.floor(values[:, :, wi] / 10) - np.floor(d['y'][None] / 10))
            row = next(r for r in s['metrics'] if r['model'] == name and r['role'] == role and r['window'] == window)
            for key, answer in [('mae', np.sum(weights * abs(error))), ('bias', np.sum(weights * error)),
                                ('mse', np.sum(weights * error**2)), ('exact_bin_rate', np.sum(weights * (gap == 0)))]:
                assert abs(row[key] - answer) < 1e-10
            for band in [r for r in s['score_bands'] if r['model'] == name and r['role'] == role and r['window'] == window]:
                mask = (d['y'] >= band['actual_bin_low']) & (d['y'] < band['actual_bin_high_exclusive'])
                present = [pid for pid in np.unique(d['person_id']) if (mask & (d['person_id'] == pid)[:, None]).any()]
                bw = np.zeros_like(weights)
                for fold in range(5):
                    chosen = d['fold'] != fold if role == 'train_people' else d['fold'] == fold
                    for pid in present:
                        n = np.count_nonzero(mask & (d['person_id'] == pid)[:, None])
                        use = mask & (d['person_id'] == pid)[:, None] & chosen[:, None]
                        bw[fold][use] = 1 / (len(present) * views * n)
                assert abs(bw.sum() - 1) < 1e-12
                assert abs(band['mae'] - np.sum(bw * abs(error))) < 1e-10
                assert abs(band['bias'] - np.sum(bw * error)) < 1e-10

# Independently check evaluation spread in the original OOF record order.
for name, p in [('ridge', baseline), ('pca_ridge_nested', oof)]:
    for wi, window in enumerate(['front', 'back']):
        people = np.unique(d['person_id'])
        per = np.array([abs(p[d['person_id'] == pid, wi] - d['y'][d['person_id'] == pid]).mean() for pid in people])
        row = next(r for r in s['metrics'] if r['model'] == name and r['role'] == 'held_people' and r['window'] == window)
        assert abs(row['mae'] - per.mean()) < 1e-12
        predmeans = np.array([p[d['person_id'] == pid, wi].mean() for pid in people])
        assert abs(row['prediction_person_sd_ddof0'] - predmeans.std(ddof=0)) < 1e-12
        for band in [r for r in s['score_bands'] if r['model'] == name and r['role'] == 'held_people' and r['window'] == window]:
            mask = (d['y'] >= band['actual_bin_low']) & (d['y'] < band['actual_bin_high_exclusive'])
            values = []
            for pid in people:
                use = mask & (d['person_id'] == pid)[:, None]
                if use.any():
                    values.append([abs(p[:, wi] - d['y'])[use].mean(), (p[:, wi] - d['y'])[use].mean()])
            values = np.array(values).mean(0)
            assert abs(band['mae'] - values[0]) < 1e-12 and abs(band['bias'] - values[1]) < 1e-12
for relative, fingerprint in a.diag.m.load_json(a.HERE.parent / 'preserved_primary_files.json').items():
    assert a.diag.m.digest(a.diag.m.HERE / relative) == fingerprint
for dirname in ['train_eval_diagnostic', 'ridge-alpha-study']:
    folder = a.HERE.parent / dirname
    for relative, fingerprint in a.diag.m.load_json(folder / 'artifact_manifest.json').items():
        assert a.diag.m.digest(folder / relative) == fingerprint
assert a.diag.m.source_data()[3] == sources

# Export to the existing OOF evaluation contract, without running any additional model.
sys = __import__('sys')
sys.path.insert(0, str(a.HERE.parent))
import evaluate as e
os = __import__('os')
os.environ['MPLCONFIGDIR'] = str(a.HERE / 'mplconfig')
models = [a.HERE / f'models/fold{k}_selected.npz' for k in range(5)]
meta = e.make_metadata(a.HERE / 'oof_predictions.npz', d, sources, models, 'pca_ridge_nested')
meta['selection'] = 'nested person-wise PCA dimension selection, alpha1 fixed'
a.write_json(a.HERE / 'oof.metadata.json', meta)
accepted, _ = e.validate_artifact(a.HERE / 'oof_predictions.npz', a.HERE / 'oof.metadata.json', d, sources)
assert np.array_equal(accepted, oof)
result = {'passed': True, 'outer_fold_models': 5, 'independent_inner_pipeline_checks': 80,
    'inner_independent_max_prediction_error': inner_max, 'outer_independent_max_prediction_error': outer_max,
    'combined_linear_head_replay_max_error': replay_max, 'inner_audits': audit_rows,
    'outer_held_people_absent_from_every_inner_fit': True, 'inner_scaler_PCA_and_Ridge_refitted': True,
    'inner_dimension_choice_independently_reproduced': True, 'all_outer_models_saved_and_replayed': True,
    'outer_metrics_and_bands_independently_checked': True,
    'all_8_train_held_rows_and_64_bands_independently_weighted': True,
    'existing_OOF_contract_accepted': True,
    'source_and_previous_artifacts_unchanged': True, 'source_hashes': sources,
    'protected_cohorts_used': False, 'encoder_reinference': False, 'GPU_used': False,
    'independent_final_evaluation': False}
a.write_json(a.HERE / 'verification.json', result)
print(__import__('json').dumps({k: v for k, v in result.items() if k != 'inner_audits'}, ensure_ascii=False))
