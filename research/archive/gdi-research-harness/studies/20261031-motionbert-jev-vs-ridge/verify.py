"""Independent split, request, Ridge-solution and error checks for direct-head pilot."""
import json
import hashlib
from run import HERE,checked_load,encode,payload,ALPHAS,save,sha
import numpy as np


def predict_dual(x,y,q,alpha):
    mu=x.mean(0);sd=x.std(0);sd[np.ptp(x,axis=0)==0]=1
    a=(x-mu)/sd;b=(q-mu)/sd
    return y.mean()+b@a.T@np.linalg.solve(a@a.T+alpha*np.eye(len(y)),y-y.mean())


def main():
    data,cases=checked_load();assert len(cases)==62
    rows=json.loads((HERE/'predictions.json').read_text());lookup={r['key']:r for r in rows}
    response=[json.loads(s) for s in (HERE/'responses.jsonl').read_text().splitlines()]
    replies={r['key']:r for r in response};assert len(replies)==62==len(response)
    result=json.loads((HERE/'results.json').read_text())
    for c in cases:
        d=data[c['dataset']];i=c['index'];examples=np.array(c['examples']);fold=c['fold'];tr=np.flatnonzero(d['fold']!=fold)
        assert len(examples)==8==len(set(examples)) and i not in examples and np.all(d['fold'][examples]!=fold)
        z=encode(d,fold);nearest=tr[np.argsort(np.sum((z[tr]-z[i])**2,axis=1),kind='stable')[:8]]
        assert set(nearest)==set(examples)
        pp=payload(c,d);assert 'target' not in pp['state']['query'] and 'person' not in pp['state']['query']
        for k,j in enumerate(examples):
            np.testing.assert_array_equal(np.fromstring(pp['state']['examples'][k]['features_csv'],sep=','),z[j])
            assert pp['state']['examples'][k]['target']==d['y'][j]
        np.testing.assert_array_equal(np.fromstring(pp['state']['query']['features_csv'],sep=','),z[i])
        assert hashlib.sha256(json.dumps(pp,sort_keys=True).encode()).hexdigest()==c['payload_sha256']==replies[c['key']]['payload_sha256']
        row=lookup[c['key']];assert row['target']==d['y'][i]
        alpha=row['matched_selection']['alpha'];assert alpha==ALPHAS[int(np.argmin(row['matched_selection']['inner_MAE']))]
        pred=predict_dual(z[examples],d['y'][examples],z[i:i+1],alpha)[0]
        np.testing.assert_allclose(pred,row['Ridge_same8'],atol=1e-8,rtol=0)
        a=replies[c['key']]['response']['answers'];chosen=a['target']['choice'];fallback=float(np.median(d['y'][examples]))
        direct=fallback if chosen=='unknown' else float(chosen.removeprefix('value_'))
        assert direct==row['Jev_direct'] and (chosen=='unknown')==row['Jev_unknown']
        score=sum(v*(fallback if k=='unknown' else float(k.removeprefix('value_'))) for k,v in a['target']['probabilities'].items())
        np.testing.assert_allclose(score,row['Jev_grid_probability_proxy'],rtol=0,atol=1e-12)
        analogue=sum(v*(fallback if k=='unknown' else float(d['y'][examples[int(k[1:])]])) for k,v in a['analogue']['probabilities'].items())
        np.testing.assert_allclose(analogue,row['Jev_example_probability_proxy'],rtol=0,atol=1e-12)
    for name in data:
        selected=[r for r in rows if r['dataset']==name];y=np.array([r['target'] for r in selected]);entry=result[name]
        assert len(selected)==entry['people']
        answered=[r for r in selected if not r['Jev_unknown']]
        assert entry['coverage']==len(answered)/len(selected)
        assert entry['answered_only']['people']==len(answered)
        for method,m in entry['metrics'].items():
            pred=np.array([r[method] for r in selected]);err=pred-y
            np.testing.assert_allclose([np.mean(abs(err)),np.sqrt(np.mean(err**2)),np.mean(err)],[m['MAE'],m['RMSE'],m['bias']],atol=1e-12,rtol=0)
            for f in range(5):
                ff=[r for r in selected if r['fold']==f]
                np.testing.assert_allclose(entry['per_fold'][str(f)][method],np.mean([abs(r[method]-r['target']) for r in ff]),rtol=0,atol=1e-12)
            if answered:
                err=np.array([r[method]-r['target'] for r in answered]);a=entry['answered_only']['metrics'][method]
                np.testing.assert_allclose([np.mean(abs(err)),np.sqrt(np.mean(err**2)),np.mean(err)],[a['MAE'],a['RMSE'],a['bias']],rtol=0,atol=1e-12)
        samples=np.random.default_rng(20261031).integers(0,len(y),size=(10000,len(y)))
        for key,m in entry['paired'].items():
            method,base=key.split('_minus_');delta=np.array([abs(r[method]-r['target'])-abs(r[base]-r['target']) for r in selected])
            np.testing.assert_allclose(m['difference'],delta.mean(),rtol=0,atol=1e-12)
            np.testing.assert_allclose(m['conditional_bootstrap95'],np.quantile(delta[samples].mean(1),[.025,.975]),rtol=0,atol=1e-12)
    save('verification.json',{'status':'passed','cases':62,'same_512_inputs':True,'same8_training_examples':True,
      'all_examples_exclude_query_fold':True,'payload_fingerprints_match':True,'query_targets_not_in_payload':True,
      'independent_matched_Ridge_dual_solves':62,'all_metrics_and_bootstrap_recomputed':True,
      'full_Ridge_independently_retrained':False,'source_and_protocol_hashes_match':True,'results_sha256':sha(HERE/'results.json'),
      'code_sha256':{name:sha(HERE/name) for name in ['run.py','verify.py']}})
    print((HERE/'verification.json').read_text())


if __name__=='__main__':main()
