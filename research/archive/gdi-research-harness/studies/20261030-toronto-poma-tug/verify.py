"""Independent file/metric/split checks and adapter failure-case regression checks."""
from collections import Counter
import csv
import io
import json
import zipfile

from prepare import HERE, ROOT, DATA, adapt, encoder, functions, labels, sha, save
import numpy as np


def main():
    checks = []
    result = json.loads((HERE/'results.json').read_text())
    receipt = json.loads((HERE/'feature_receipt.json').read_text())
    source = json.loads((HERE/'source_receipt.json').read_text())
    for proof, base in [(result, HERE), (receipt, HERE), (source, ROOT)]:
        assert all(sha(base/path) == expected for path, expected in proof['sha256'].items())
    checks.append('source, configuration, code, weight and output fingerprints unchanged')
    with np.load(HERE/'features.npz', allow_pickle=False) as z:
        X, Y, people, record_features, record_people = [z[k].copy() for k in ['X','Y','person','record_features','record_person']]
    with np.load(HERE/'inputs.npz', allow_pickle=False) as z:
        inputs, input_people = z['input'].copy(), z['person'].copy()
    np.testing.assert_array_equal(input_people, record_people)
    assert len(people) == result['people'] and len(inputs) == result['windows']
    assert inputs.shape[1:] == (124,17,3) and record_features.shape == (len(inputs),512)
    assert np.isfinite(inputs).all() and np.isfinite(record_features).all()
    clinical = labels()
    for i, p in enumerate(people):
        np.testing.assert_array_equal(Y[i], clinical[int(p)])
        np.testing.assert_array_equal(X[i], record_features[record_people == p].astype(np.float64).mean(0))
    checks.append('all person labels and feature averaging independently rejoined')
    split = json.loads((HERE/'folds.json').read_text())['outer']
    assert len(split) == len(people) and {x['evaluation'] for x in split} == set(people)
    for outer in split:
        assert len(outer['training']) == len(people)-1 and outer['evaluation'] not in outer['training']
        assert set(outer['training']) | {outer['evaluation']} == set(people)
        assert {x['evaluation'] for x in outer['inner']} == set(outer['training'])
        for inner in outer['inner']:
            assert len(inner['training']) == len(people)-2
            assert set(inner['training']) | {inner['evaluation']} == set(outer['training'])
            assert not set(inner['training']) & {inner['evaluation'], outer['evaluation']}
    trace = json.loads((HERE/'fit_trace.json').read_text())
    for fit in trace:
        assert not set(fit['train_people']) & set(fit['evaluation_people'])
        if fit['stage'] == 'inner':
            assert fit['outer'] not in fit['train_people']+fit['evaluation_people']
    checks.append('all outer and inner fits have disjoint participant sets')
    with (HERE/'predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(people)*2*4
    with np.load(HERE/'predictions.npz', allow_pickle=False) as pred:
        np.testing.assert_array_equal(pred['truth'], Y)
        for j, target in enumerate(['POMA_Gait','TUG_seconds']):
            for method in ['mean','median','ridge','pls']:
                selected = [x for x in rows if x['target'] == target and x['method'] == method]
                assert [int(x['person']) for x in selected] == people.tolist()
                actual = np.asarray([float(x['actual']) for x in selected])
                values = np.asarray([float(x['predicted']) for x in selected])
                np.testing.assert_array_equal(actual, Y[:,j])
                np.testing.assert_array_equal(values, pred[method][:,j])
                expected = result['metrics'][target][method]
                error = values-actual
                assert abs(np.abs(error).mean()-expected['MAE']) < 1e-12
                assert abs(np.sqrt(np.mean(error**2))-expected['RMSE']) < 1e-12
                assert abs(1-np.sum(error**2)/np.sum((actual-actual.mean())**2)-expected['R2']) < 1e-12
                if method in ['mean','median']:
                    for i in range(len(people)):
                        other = np.delete(actual,i)
                        baseline = other.mean() if method == 'mean' else np.median(other)
                        assert values[i] == baseline
    checks.append('every CSV/NPZ prediction, baseline, MAE, RMSE and R2 independently recalculated')
    convert, normalize = functions()
    preparation = json.loads((HERE/'preparation.json').read_text())
    included = [x for x in preparation['records'] if x['status'] == 'included']
    with zipfile.ZipFile(DATA/'PoseTracking.zip') as archive:
        for row in included:
            raw = archive.read(row['member'])
            data = np.asarray(list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))[1:], dtype=np.float64)
            rebuilt, info = adapt(data, convert, normalize)
            np.testing.assert_array_equal(inputs[row['input_row']], rebuilt)
            assert abs(info['output_end_seconds']-info['output_start_seconds']-4.1) < 1e-12
    checks.append('every saved real input rebuilt from original CSV at 30Hz without extrapolation')
    # A missing middle frame must not turn two short runs into one usable walk.
    t = np.arange(241,dtype=float)/30
    motion = np.zeros((len(t),17,3),dtype=float)
    motion[...,:2] = np.arange(34).reshape(17,2)[None]+t[:,None,None]
    motion[...,2] = .8
    fixture = np.column_stack([t,motion.reshape(len(t),51)])
    valid, _ = adapt(fixture,convert,normalize)
    assert valid is not None
    direct = normalize(convert(motion[:124].astype(np.float32),'coco','h36m'))
    np.testing.assert_array_equal(valid,direct)
    for kind in ['nan','time_gap','negative_confidence']:
        bad = fixture.copy()
        if kind == 'nan': bad[120,1] = np.nan
        if kind == 'time_gap': bad[120:,0] += 1/30
        if kind == 'negative_confidence': bad[120,3] = -1
        output, info = adapt(bad,convert,normalize)
        assert output is None and info['status'] == 'excluded_shorter_than_4_1_seconds'
    checks.append('synthetic missing-frame, time-gap and invalid-confidence cases cannot be bridged')
    model, torch = encoder()
    with torch.inference_mode():
        for i in [0,len(inputs)//2,len(inputs)-1]:
            value = model.get_representation(torch.from_numpy(inputs[i:i+1])).mean((1,2)).numpy()[0]
            np.testing.assert_array_equal(value,record_features[i])
    checks.append('three real frozen-encoder feature vectors reproduced exactly')
    assert all(sha(ROOT/path) == expected for path, expected in source['sha256'].items())
    save('verification.json', {'verified': True, 'checks': checks, 'real_people': len(people), 'real_windows': len(inputs),
                               'prediction_rows_checked': len(rows), 'fit_trace_entries_checked': len(trace),
                               'synthetic_tests_are_not_clinical_data': True,
                               'original_sources_and_encoder_unchanged': True,
                               'results_sha256': sha(HERE/'results.json'), 'verification_code_sha256': sha(HERE/'verify.py')})
    print(json.dumps({'verified':True,'checks':checks,'people':len(people),'windows':len(inputs),'prediction_rows':len(rows)},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
