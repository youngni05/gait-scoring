"""Check every added checkpoint against stored held-out predictions and fold fusion."""
import json
import engine
import comparison as c

np = engine.np
BIG = engine.Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
study = engine.ROOT / c.MANIFEST['sourceStudy']
completion = study / 'completion-verification.json'
assert engine.sha(completion) == c.MANIFEST['sourceCompletionSHA256']
assert json.loads(completion.read_text())['passed']
baseline = json.loads((study/'bundle/baseline-models.json').read_text())
for m in engine.MANIFEST['members']:
    original = next(v for v in baseline if v['family']==m['family'] and v['fold']==m['fold']
                    and (v['tag']=='refit' if m['family']=='bilateral' else m['id']==f"{v['family']}_{v['fold']}_{v['tag']}"))
    assert m['source_checkpoint_sha256'] == original['checkpoint_sha256']
    assert m['calibration'] == original['calibration']
assert engine.MANIFEST['selections'] == c.MANIFEST['selections']
with np.load(BIG/'data.npz') as data:
    x, owner = data['x'], data['window_record']
report = {'clinicalValidation':False, 'tolerance':1e-4, 'members':[], 'folds':[]}
for f in range(5):
    members = [m for m in c.MANIFEST['members'] if m['fold']==f]
    folder = BIG/f'received/data/runs/mixed/CNN/fold{f}'
    with np.load(folder/'predictions.npz') as z:
        rows = z['row_index'][:3]
    actual = [dict() for _ in rows]
    for m in members:
        folder = engine.Path(m['path']).parent
        with np.load(folder/'replay.npz') as z:
            indices = [int(np.flatnonzero(z['evaluation_row']==row)[0]) for row in rows]
            expected = z['evaluation_raw'][indices]
        observed = np.array([c.raw_member(m, x[owner==row]).mean(0) for row in rows])
        difference = float(np.max(np.abs(observed-expected)))
        assert difference <= 1e-4, (m['id'],difference)
        calibrated = observed*m['calibration'][0]+m['calibration'][1]
        for a, value in zip(actual, calibrated): a[m['id']]=value
        report['members'].append({'id':m['id'],'maxDifference':difference})
    expected = {}
    for family in ('CNN','STT','bilateral'):
        folder = BIG/f'received/data/runs/mixed/{family}/fold{f}'
        done = json.loads((folder/'complete.json').read_text())
        assert engine.sha(folder/'predictions.npz') == done['prediction_sha256']
        with np.load(folder/'predictions.npz') as z:
            assert np.array_equal(z['row_index'][:3],rows)
            expected[family] = {key:z[key][:3] for key in ('refit','bag')}
    s = c.MANIFEST['selections'][f]
    base = s['cnn_weight']*expected['CNN']['bag']+(1-s['cnn_weight'])*expected['STT']['bag']
    bil = expected['bilateral']['refit']
    delta = (1-s['fraction'])*(base[:,0]-base[:,1])+s['fraction']*(bil[:,0]-bil[:,1])
    target = {'CNN':expected['CNN']['refit'], 'STT':expected['STT']['refit'],
              'fusion':np.stack([base.mean(1)+delta/2,base.mean(1)-delta/2],axis=1)}
    scores = [c.fold_scores(v,c.MANIFEST,f) for v in actual]
    errors = {kind:float(np.max(np.abs(np.array([v[kind] for v in scores])-target[kind]))) for kind in target}
    assert max(errors.values()) <= 1e-4, errors
    report['folds'].append({'fold':f,'records':rows.tolist(),'maxDifference':errors})
    print('Fold',f,'passed',errors,flush=True)
report['passed'] = True
report['manifestSHA256'] = engine.sha(c.MANIFEST_PATH)
(engine.HERE/'comparison-verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('All 55 checkpoints and all 5 fold combinations passed.',flush=True)
