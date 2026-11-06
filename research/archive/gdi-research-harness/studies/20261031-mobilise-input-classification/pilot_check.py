from study import *

initialize()
errors=[]
for target in ['sppb','tug']:
    tag=f'{target}-long-5-f0-s{SEEDS[0]}'
    tr,te,inner,_=old.cell_data(target,'long',5,0)
    x,xt=inputs(tr,5),inputs(te,5)
    ledger=json.loads((STORE/'folds'/f'{tag}.json').read_text())
    result=pd.read_csv(STORE/'folds'/f'{tag}.csv.gz')
    previous=pd.read_csv(old.STORE/'folds'/f'{tag}.csv.gz')
    for variant in ['raw','cal']:
        np.testing.assert_allclose(result[[f'MedianRF_{variant}_p{j}' for j in range(4)]],
                                   previous[[f'RF_{variant}_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    bundle=dict(np.load(ledger['models']['DeepSets']['path']))
    raw=predict_ds(bundle,xt['DeepSets'])
    np.testing.assert_allclose(raw,predict_ds(bundle,xt['DeepSets'][:,::-1]),rtol=0,atol=1e-12)
    np.testing.assert_allclose(raw,result[[f'DeepSets_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
    w=weights(tr.participantid.to_numpy());w/=w.sum()
    mean=np.average(x['DeepSets'].reshape(-1,6),axis=0,weights=np.repeat(w/5,5))
    np.testing.assert_allclose(mean,bundle['mean'],rtol=0,atol=1e-12)
    ins=np.load(STORE/'folds'/f'{tag}-inner.npz')
    for i, m in enumerate(ledger['models']['DeepSets']['inner_models']):
        actual=predict_ds(dict(np.load(m['path'])),x['DeepSets'][inner==i])
        error=float(np.max(abs(actual-ins['DeepSets_oof'][inner==i])))
        assert error<=1e-5;errors.append(error)
    assert ledger['models']['DeepSets']['epochs'] in EPOCHS
np.testing.assert_array_equal(old.classes(np.array([0,3,4,6,7,9,10,12]),'sppb'),[0,0,1,1,2,2,3,3])
np.testing.assert_array_equal(old.classes(np.array([10,10.01,15,15.01,20,20.01]),'tug'),[0,1,1,2,2,3])
save(HERE/'pilot-verification.json',{'passed':True,'pilot_tasks':2,'max_float32_numpy_inner_probability_difference':max(errors),
    'checks':['exact prior median RF probabilities','DS set permutation invariance','train-only input normalization','inner DS saved-model probability replay','fixed class boundaries'],
    'tolerance':{'float64':1e-12,'float32_to_numpy':1e-5},'code_sha256':sha(__file__)})
print('PILOT_CHECK_PASSED',max(errors),flush=True)
