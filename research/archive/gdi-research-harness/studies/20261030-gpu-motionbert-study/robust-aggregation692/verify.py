"""Independent order-statistic, calibration, split and selection verification."""
import argparse
from common import *

def independent(raw,owner,rows):
    out=np.empty((6,len(rows),2))
    for j,i in enumerate(rows):
        for side in range(2):
            vals=sorted(float(v) for v in raw[owner==i,side]);n=len(vals)
            out[0,j,side]=sum(vals)/n
            out[1,j,side]=(vals[(n-1)//2]+vals[n//2])/2
            for q,p in enumerate([.1,.2]):
                k=int(n*p);out[2+q,j,side]=sum(vals[k:n-k])/(n-2*k)
                out[4+q,j,side]=(k*vals[k]+sum(vals[k:n-k])+k*vals[n-k-1])/n
    return out
def disjoint(train,valid,outer):
    assert set(train).isdisjoint(valid) and set(train).isdisjoint(outer) and set(valid).isdisjoint(outer)
def fixtures():
    raw=np.column_stack([np.r_[5.,np.arange(13.),1000.],np.r_[7.,-np.arange(13.),-1000.]])
    owner=np.r_[11,np.full(14,20)];rows=np.array([20,11]);v=aggregate(raw,owner,rows)
    np.testing.assert_allclose(v,independent(raw,owner,rows),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(v[:,1],np.tile([5,7],(6,1)))
    assert v[1,0,0]<v[0,0,0] and v[2,0,0]<v[0,0,0] and v[4,0,0]<v[0,0,0]
    np.testing.assert_allclose(aggregate(raw[::-1],owner[::-1],rows),v,atol=1e-12,rtol=0)
    try:aggregate(raw,owner,np.array([99]))
    except AssertionError:pass
    else:raise AssertionError('Empty record accepted')
    try:disjoint([1],[2],[1])
    except AssertionError:pass
    else:raise AssertionError('Leak accepted')
    save(O/'fixtures.json',dict(passed=True,odd_even_median=True,single_window=True,tail_count_floor=True,
        extreme_values=True,order_invariance=True,empty_record_rejected=True,person_leak_rejected=True))

def full():
    fixtures();cfg=read(H/'protocol.json');sel=read(O/'selection.json');report=read(O/'comparison.json')
    assert sel['protocol_sha256']==sha(H/'protocol.json') and report['selection_sha256']==sha(O/'selection.json')
    assert sha(Path(report['predictions_path']))==report['predictions_sha256'];z=arrays(Path(report['predictions_path']))
    count=0;hashes={};maxagg=0;constant=[]
    for family in ('CNN','STT','control'):
        for f in range(5):
            js=[None] if family=='control' else [j for j in range(5) if j!=f]+[None]
            for j in js:
                p=CACHE/family/('fold%d_%s.npz'%(f,'refit' if j is None else 'inner%d'%j));v=arrays(p);proof=read(p.with_suffix('.json'))
                assert proof['cache_sha256']==sha(p) and sha(proof['checkpoint'])==proof['checkpoint_sha256']
                for source,digest in proof['source_files'].items():assert sha(source)==digest
                for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(v[key],z[key][v['row_index']])
                tr=v['train_mask'];va=v['validation_mask'];assert np.all(tr^va)
                if j is not None:
                    assert (v['fold']!=f).all() and np.all(v['fold'][va]==j)
                    disjoint(v['person_id'][tr],v['person_id'][va],z['person_id'][z['fold']==f])
                else:disjoint(v['person_id'][tr],v['person_id'][va],[])
                independent_agg=independent(v['window_raw'],v['window_record'],v['row_index'])
                maxagg=max(maxagg,float(np.abs(independent_agg-v['aggregate_raw']).max()))
                np.testing.assert_allclose(independent_agg,v['aggregate_raw'],atol=1e-10,rtol=0)
                for k in range(6):
                    x=independent_agg[k,tr].ravel();y=v['y'][tr].ravel()
                    if np.all(x==x[0]):
                        # Rank-one OLS has infinitely many coefficients. The source
                        # lstsq uses the minimum-norm solution, derived here directly.
                        slope=float(y.mean()*x[0]/(x[0]**2+1));intercept=float(y.mean()/(x[0]**2+1))
                        constant.append(dict(family=family,outer=f,inner=j,method=cfg['methods'][k],raw_constant=float(x[0])))
                    else:
                        slope=float(np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2));intercept=float(y.mean()-slope*x.mean())
                    # Independent centered OLS, compared in prediction space.
                    np.testing.assert_allclose(independent_agg[k]*slope+intercept,v['corrected'][k],atol=1e-8,rtol=0)
                    poisoned=v['y'].copy();poisoned[va]=1e9
                    np.testing.assert_array_equal(fit_calibration(v['aggregate_raw'][k,tr],poisoned[tr]),v['calibration'][k])
                hashes[str(p)]=sha(p);count+=1
    import compare as c
    for sf in sel['folds']:
        v,meta,p,h=c.inner(sf['outer']);w=sf['cnn_weight'];t=sf['difference_fraction']
        pair,scores=choose(p['CNN'],p['STT'],v['bilateral'],v['y'],v['person_id'],w,t);assert list(pair)==sf['pair']
        # Test changes to outer answers cannot alter the inner-only choice.
        y=z['y'].copy();y[z['fold']==sf['outer']]=-1e6
        poisoned_pair,_=choose(p['CNN'],p['STT'],v['bilateral'],y[v['row_index']],v['person_id'],w,t)
        assert pair==poisoned_pair
        reversed_pair,_=choose(p['CNN'][:,::-1],p['STT'][:,::-1],v['bilateral'][::-1],v['y'][::-1],v['person_id'][::-1],w,t)
        assert pair==reversed_pair
        weights=np.array([1/np.sum(v['person_id']==i) for i in v['person_id']]);weights/=weights.sum()
        check=np.array([[np.sum(np.abs(combine(cp,sp,v['bilateral'],w,t)-v['y']).mean(1)*weights) for sp in p['STT']] for cp in p['CNN']])
        np.testing.assert_allclose(check,scores,atol=1e-12,rtol=0)
        te=z['fold']==sf['outer'];cc,_,_=c.cached('CNN',sf['outer'],None);ss,_,_=c.cached('STT',sf['outer'],None)
        baseold=arrays(MD/'analysis/predictions.npz')
        reconstructed=combine(cc['corrected'][pair[0],cc['validation_mask']],ss['corrected'][pair[1],ss['validation_mask']],baseold['bilateral_STT'][te],w,t)
        np.testing.assert_array_equal(reconstructed,z['inner_selected'][te])
    for name,value in report['mae'].items():
        perrecord=np.abs(z[name]-z['y']).mean(1);ids=z['person_id'];weights=np.array([1/np.sum(ids==i) for i in ids]);weights/=weights.sum()
        assert abs(float(np.sum(perrecord*weights))-value)<1e-12
        distance=np.abs(np.floor_divide(z[name],10)-np.floor_divide(z['y'],10))
        for key,flag in [('exact',distance==0),('within_one',distance<=1),('two_or_more',distance>=2)]:
            assert abs(float(np.sum(flag.mean(1)*weights))-report['bands10'][name][key])<1e-12
    control=read(O/'control_diagnostic.json');cp=np.full((6,len(z['y']),2),np.nan)
    for f in range(5):
        v,_,_=c.cached('control',f,None);te=z['fold']==f;cp[:,te]=v['corrected'][:,v['validation_mask']]
    for method,pred in zip(cfg['methods'],cp):
        assert abs(float(np.sum(np.abs(pred-z['y']).mean(1)*weights))-control['mae'][method])<1e-12
        case=control['worst_original_case'];assert pred[case['row'],case['side']]==case['predictions'][method]
    assert count==55
    save(O/'verification.json',dict(passed=True,cached_models=55,independent_aggregations=330,max_aggregation_error=maxagg,
        all_training_only_calibrations=True,person_boundaries=True,selection_recomputed=True,poisoned_labels_invariant=True,
        row_order_invariant=True,person_equal_metric=True,source_and_cache_hashes_checked=True,source_cache_hashes=hashes,
        comparison_sha256=sha(O/'comparison.json'),selection_sha256=sha(O/'selection.json'),optimizer_steps=0,protected_final_opened=False,
        constant_training_prediction_conditions=constant,initial_centered_OLS_verifier_failure_preserved=True,
        bands_and_control_metrics_independently_checked=True,control_report_sha256=sha(O/'control_diagnostic.json'),
        final_script_hashes={n:sha(H/n) for n in ('common.py','export.py','compare.py','verify.py')}))
    print('VERIFICATION_PASSED',count,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fixtures','full']);a=p.parse_args();globals()[a.stage]()
