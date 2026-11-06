"""Independent sklearn calibration fits and scalar metric verification."""
import importlib.util
from statistics import mean
import run as r
np,p,m,HERE=r.np,r.p,r.m,r.HERE
from sklearn.linear_model import LinearRegression

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
old=module('side_verifier',p.HERE/'verify.py')
scalar=module('scalar_verifier',p.BLEND/'verify.py')

def equal(row,values):
    for k,v in values.items():
        if v is None:assert row[k] is None
        else:assert abs(row[k]-v)<1e-9,(k,row[k],v)

def main():
    sources,d,cube,inner,cnn=r.sources_and_inputs();saved=m.load_json(HERE/'source_hashes.json');assert sources==saved
    s=m.load_json(HERE/'summary.json');o=p.load(HERE/'oof_predictions.npz');folds=p.load(HERE/'fold_predictions.npz')['predictions']
    inputs=p.load(HERE/'calibration_inputs.npz');maxdiff=0.;count=0
    for key in ['y','person_id','video_id','fold','source_group']:np.testing.assert_array_equal(d[key],o[key])
    for model in r.CFG['models']:
        for f in range(5):
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            assert not set(d['person_id'][tr])&set(d['person_id'][te])
            for variant in r.CFG['variants']:
                name=model+'__'+variant;mi=r.METHODS.index(name)
                if variant=='unchanged':expected=cube[model][f]
                else:
                    x=cube[model][f,tr] if variant=='train_affine' else inner[model][f]
                    np.testing.assert_array_equal(x,inputs[name+f'__fold{f}'])
                    fit=LinearRegression().fit(x.reshape(-1,1),d['y'][tr].ravel())
                    expected=fit.predict(cube[model][f].reshape(-1,1)).reshape(1064,2)
                    row=next(v for v in s['coefficients'] if (v['model'],v['variant'],v['fold'])==(model,variant,f))
                    assert abs(row['slope']-fit.coef_[0])<1e-10 and abs(row['intercept']-fit.intercept_)<1e-9
                    count+=1
                maxdiff=max(maxdiff,float(abs(expected-folds[mi,f]).max()))
                np.testing.assert_allclose(expected,folds[mi,f],atol=1e-9,rtol=0)
                np.testing.assert_array_equal(folds[mi,f,te],o['predictions'][mi,te])
    np.testing.assert_array_equal(cnn,o['predictions'][-1])
    checks=0
    for row in s['metrics']:
        mask=np.ones(1064,dtype=bool) if row['group']=='all695' else d['source_group']==row['group']
        pred=o['predictions'][r.METHODS.index(row['method'])]
        equal(row,old.stats(d['y'][mask],pred[mask],d['person_id'][mask]));equal(row,scalar.independent_metrics(d,pred,selected=np.flatnonzero(mask)));checks+=1
    for row in s['bands']:
        equal(row,scalar.independent_metrics(d,o['predictions'][r.METHODS.index(row['method'])],band=row['actual_bin_low']//10));checks+=1
    for row in s['train_held']:
        f=row['fold'];mask=d['fold']==f if row['split']=='held' else d['fold']!=f
        equal(row,old.stats(d['y'][mask],folds[r.METHODS.index(row['method']),f,mask],d['person_id'][mask]));checks+=1
    draws=np.random.default_rng(20261029).integers(0,695,(10000,695))
    for row in s['comparisons']:
        arrays=[]
        for name in [row['method'].split('__')[0]+'__unchanged',row['method']]:
            err=abs(o['predictions'][r.METHODS.index(name)]-d['y'])
            arrays.append(np.array([mean(err[d['person_id']==pid].flatten()) for pid in np.unique(d['person_id'])]))
        delta=arrays[0]-arrays[1]
        assert abs(delta.mean()-row['reduction'])<1e-10
        np.testing.assert_allclose(np.quantile(delta[draws].mean(1),[.025,.975]),row['bootstrap95'],atol=1e-9,rtol=0)
    result=dict(verified=True,people=695,records=1064,independent_calibration_refits=count,max_prediction_difference=maxdiff,
        metric_rows_checked=checks,bootstrap_comparisons_checked=len(s['comparisons']),sources_unchanged=len(sources),
        CNN_calibrated=False,CNN_reason='train predictions absent',GPU_used=False,protected_numeric_data_used=False,independent_final_evaluation=False)
    r.js('verification.json',result);print(result)

if __name__=='__main__':main()
