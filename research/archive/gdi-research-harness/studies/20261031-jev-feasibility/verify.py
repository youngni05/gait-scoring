"""Independent output/label/fold/payload and arithmetic checks after the pilot."""
import json
import hashlib
from study import HERE, load, jobs, job_id, payload, sha, save, MAX_REQUESTS, MAX_TOKENS
import numpy as np


def main():
    data=load();planned={job_id(j):j for j in jobs(data)}
    rows=[json.loads(s) for s in (HERE/'responses.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['job'] for r in rows})==len(planned)
    assert len(rows)<=MAX_REQUESTS
    tokens=0
    for r in rows:
        assert r['status']=='ok',r['job']
        j=planned[r['job']];name,i,arm,reverse,model,tag=j
        p=payload(data,name,i,arm,reverse,model)
        assert hashlib.sha256(json.dumps(p,sort_keys=True,allow_nan=False).encode()).hexdigest()==r['payload_sha256']
        assert r['response']['model']==model
        for key,q in p['questions'].items():
            a=r['response']['answers'][key]
            assert a['choice'] in q['criteria'] and set(a['probabilities'])==set(q['criteria'])
            assert abs(sum(a['probabilities'].values())-1)<.01
            assert all(0<=v<=1 for v in a['probabilities'].values())
        tokens+=r['response']['usage']['input_tokens']
    assert tokens<=MAX_TOKENS
    results=json.loads((HERE/'results.json').read_text())
    predictions=json.loads((HERE/'predictions.json').read_text())
    assert set(results)==set(predictions)=={'GDI','TUG'}
    checks=0
    for name,p in predictions.items():
        d=data[name];y=np.asarray(d['y']);f=np.asarray(d['fold'])
        np.testing.assert_array_equal(p['target'],y)
        np.testing.assert_array_equal(p['fold'],f)
        assert p['ids']==d['ids']
        for method,values in p['predictions'].items():
            v=np.asarray(values);assert v.shape==y.shape and np.isfinite(v).all()
            errors=v-y;m=results[name]['metrics'][method]
            assert abs(m['MAE']-sum(abs(errors))/len(y))<1e-10
            assert abs(m['bias']-sum(errors)/len(y))<1e-10
            assert abs(m['RMSE']-(sum(errors**2)/len(y))**.5)<1e-10
            np.testing.assert_allclose(results[name]['fold_MAE'][method],
                                       [np.mean(abs(errors[f==i])) for i in range(5)])
            checks+=1
        for method,comparison in results[name]['paired'].items():
            baseline=method.split('_')[0]+'_numeric'
            delta=np.abs(np.array(p['predictions'][method])-y)-np.abs(np.array(p['predictions'][baseline])-y)
            assert abs(delta.mean()-comparison['MAE_difference_new_minus_baseline'])<1e-10
    receipt=json.loads((HERE/'preparation.json').read_text())
    assert all(sha(path)==digest for path,digest in receipt['source_hashes'].items())
    save('verification.json',{'status':'passed','requests_verified':len(rows),'input_tokens':tokens,
                              'method_metric_sets_recomputed':checks,'payloads_recreated':True,
                              'source_hashes_unchanged':True,'person_ids_and_folds_match':True,
                              'model_retraining_independently_reproduced':False,
                              'results_sha256':sha(HERE/'results.json'),'responses_sha256':sha(HERE/'responses.jsonl')})
    print((HERE/'verification.json').read_text())


if __name__=='__main__':main()
