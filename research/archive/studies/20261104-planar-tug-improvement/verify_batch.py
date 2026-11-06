"""Independent replay, preprocessing/selection checks and paired metrics."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np,joblib
import sensor_loop as s
import projection_audit as p
HERE=s.HERE;HEAVY=s.HEAVY
def metric(y,z):
    err=z-y;ae=abs(err);high=y>=20
    return {'n':len(y),'MAE':float(ae.mean()),'p95':float(np.quantile(ae,.95)),'bias':float(err.mean()),
       'high_n':int(high.sum()),'high_MAE':float(ae[high].mean()),'high_bias':float(err[high].mean())}
def delta(a,b):
    x=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261104)
    boot=np.concatenate([x[rng.integers(0,len(x),(1000,len(x)))].mean(1) for _ in range(5)])
    return {'delta_MAE':float(x.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist()}
def replay(m,X):
    a=np.where(np.isnan(X),m['imputer'].statistics_,X)
    z=np.column_stack([a,1/np.maximum(a[:,0],.1)])
    z=(z-m['scaler'].mean_)/m['scaler'].scale_
    if m['spline'] is not None:z=m['spline'].transform(z)
    return np.maximum(.1,m['model'].predict(z))
def sensor():
    prot=json.loads((HERE/'sensor-protocol.json').read_text());assert s.sha(s.SOURCE)==prot['source_sha256'] and s.sha(HERE/'sensor_loop.py')==prot['code_sha256']
    d,X,sets=s.load();y=d['y'];ids=d['ids'];records=json.loads((HERE/'sensor-receipt.json').read_text());assert len(records)==25
    pred={};maxdiff=0.;nmodel=0;meta_count=0
    for r in records:
        view=r['view'];tr=np.array(r['train']);te=np.array(r['test']);f=r['fold']
        assert np.array_equal(tr,np.flatnonzero(d['fold']!=f)) and np.array_equal(te,np.flatnonzero(d['fold']==f))
        assert not set(ids[tr])&set(ids[te]) and ids[tr].tolist()==r['train_ids'] and ids[te].tolist()==r['test_ids']
        path=HEAVY/'sensor-models'/r['context'];assert s.sha(path)==r['context_sha256'];ctx=np.load(path)
        A=np.full_like(ctx['A'],np.nan);T=np.empty_like(ctx['T'])
        for rec in r['models']:
            file=HEAVY/'sensor-models'/rec['file'];assert s.sha(file)==rec['sha256'];m=joblib.load(file)
            it=np.array(rec['train']);iv=np.array(rec['test']);assert not set(it)&set(iv)
            assert m['names']==sets[view] and m['config']==s.CONFIGS[rec['candidate']] and m['ntrain']==len(it)
            raw=X[view][it];med=np.array([np.nanmedian(raw[:,j]) if np.isfinite(raw[:,j]).any() else 0 for j in range(raw.shape[1])])
            np.testing.assert_allclose(m['imputer'].statistics_,med,atol=1e-12,rtol=0)
            aa=np.where(np.isnan(raw),med,raw);en=np.column_stack([aa,1/np.maximum(aa[:,0],.1)])
            np.testing.assert_allclose(m['scaler'].mean_,en.mean(0),atol=1e-12,rtol=0)
            np.testing.assert_allclose(m['scaler'].var_,en.var(0),atol=1e-10,rtol=1e-10)
            pp=replay(m,X[view][iv]);k=rec['candidate'];nmodel+=1
            if rec['inner'] is None:assert np.array_equal(it,tr) and np.array_equal(iv,te);T[:,k]=pp
            else:
                assert set(it)|set(iv)==set(tr);A[np.searchsorted(tr,iv),k]=pp
        for a,b in [(A,ctx['A']),(T,ctx['T'])]:
            dd=float(abs(a-b).max());assert dd<1e-9;maxdiff=max(maxdiff,dd)
        order=np.argsort(abs(A-y[tr,None]).mean(0),kind='stable');assert order.tolist()==r['order']
        meta=HEAVY/'sensor-models'/r['meta'];assert s.sha(meta)==r['meta_sha256'];ms=joblib.load(meta)
        pp={'selected':T[:,order[0]],'top3':T[:,order[:3]].mean(1)}
        for name,m in ms.items():
            center=A.mean(0);scale=np.maximum(A.std(0),1e-6);np.testing.assert_allclose(m['center'],center,atol=1e-12,rtol=0);np.testing.assert_allclose(m['scale'],scale,atol=1e-12,rtol=0)
            assert abs(m['ymean']-y[tr].mean())<1e-12
            Z=(A-center)/scale;w=np.where(y[tr]>=20,m['tail'],np.where(y[tr]>=12,np.sqrt(m['tail']),1.))
            grad=Z.T@(w*(Z@m['coef']-(y[tr]-y[tr].mean())))+10*m['coef']
            active=m['coef']>1e-10;assert (m['coef']>=0).all()
            assert np.max(abs(grad[active]),initial=0)<1e-6 and np.min(grad[~active],initial=0)>-1e-6
            pp[name]=np.maximum(.1,(T-center)/scale@m['coef']+y[tr].mean());meta_count+=1
        for policy,z in pp.items():
            np.testing.assert_allclose(z,ctx[policy],atol=1e-9,rtol=0)
            key=view+'/'+policy
            if key not in pred:pred[key]=np.full(len(y),np.nan)
            pred[key][te]=z
        print('SENSOR_VERIFIED',view,f,flush=True)
    assert nmodel==1200 and meta_count==50 and all(np.isfinite(a).all() for a in pred.values())
    receipt=json.loads(Path('D:/Projects/D_P/studies/20261104-pose-compatible-tug/run-receipt.json').read_text())
    for view in ['speed','timing','calibrated_spatiotemporal']:
        old=np.empty(len(y))
        for f in range(5):
            path=s.OLD/f'{view}-{f}.json';assert s.sha(path)==receipt['hashes'][str(path)]
            r=json.loads(path.read_text());te=np.flatnonzero(d['fold']==f);assert r['test_ids']==ids[te].tolist()
            old[te]=r['predictions']['stacked']
        pred['previous/'+view]=old
    summary={k:metric(y,z) for k,z in pred.items()};comparisons={}
    for k,z in pred.items():
        policy=k.split('/')[-1];base=pred['speed/'+policy] if policy in ['selected','top3','stacked','tail_stacked'] else pred['speed/stacked']
        comparisons[k]={'matched_speed':delta(abs(z-y),abs(base-y)),
          'previous_metric12':delta(abs(z-y),abs(pred['previous/calibrated_spatiotemporal']-y))}
    s.save(HERE/'sensor-results.json',{'summary':summary,'comparisons':comparisons,
      'candidate_primary':'fixed stacked, declared before fitting; secondary conditions are descriptive','limits':prot['scope']})
    np.savez_compressed(HEAVY/'sensor-verified-predictions.npz',y=y,ids=ids,**{k.replace('/','__'):z for k,z in pred.items()})
    s.save(HERE/'sensor-verification.json',{'passed':True,'models_replayed':nmodel,'meta_models_verified':meta_count,'max_replay_abs':maxdiff,
      'train_only_preprocessing':True,'person_disjoint_cases':25,'results_sha256':s.sha(HERE/'sensor-results.json'),'protocol_sha256':s.sha(HERE/'sensor-protocol.json')})
def projection():
    prot=json.loads((HERE/'projection-protocol.json').read_text());qa=json.loads((HERE/'projection-geometry-audit.json').read_text())
    assert p.sha(HERE/'projection_audit.py')==prot['code_sha256'] and p.sha(p.OLD/'experiment.py')==prot['old_code_sha256']
    assert p.sha(HEAVY/'projection/input.npz')==qa['input_sha256'] and p.sha(HEAVY/'projection/raw-window-cache.npz')==qa['raw_cache_sha256']
    d=np.load(HEAVY/'projection/input.npz');cache=np.load(HEAVY/'projection/raw-window-cache.npz');y=d['y'];ids=d['ids']
    for i,raw in enumerate(cache['poses']):
        for v in p.VIEWS:
            f=p.features(raw,cache['times'][i],v);np.testing.assert_allclose([f[n] for n in qa['view_feature_names'][v]],d[v][i],atol=1e-10,rtol=0,equal_nan=True)
    recs=json.loads((HERE/'projection-receipt.json').read_text());assert len(recs)==300;pred={};maxdiff=0.
    for r in recs:
        tr=np.array(r['train']);te=np.array(r['test']);assert not set(ids[tr])&set(ids[te])
        x=d[r['view']][:,:1] if r['arm']=='speed' else d[r['view']];x=np.column_stack([x,1/np.maximum(x[:,0],.1)])
        file=HEAVY/'projection'/r['model'];assert p.sha(file)==r['sha256'];m=joblib.load(file)
        med=np.array([np.nanmedian(x[tr,j]) if np.isfinite(x[tr,j]).any() else 0 for j in range(x.shape[1])])
        np.testing.assert_allclose(m['imputer'].statistics_,med,atol=1e-12,rtol=0)
        pp=m['model'].predict(np.where(np.isnan(x[te]),med,x[te]));dd=float(abs(pp-r['prediction']).max());assert dd<1e-9;maxdiff=max(maxdiff,dd)
        key=r['view']+'/'+r['arm']
        if key not in pred:pred[key]=np.full((5,len(y)),np.nan)
        pred[key][r['repeat'],te]=pp
    summary={};comparison={}
    for k,z in pred.items():
        assert np.isfinite(z).all();ae=abs(z-y);hi=y>=20
        summary[k]={'n':len(y),'MAE_repeated':float(ae.mean()),'MAE_each_repeat':ae.mean(1).tolist(),'high_MAE':float(ae[:,hi].mean()),'bias':float((z-y).mean())}
        comparison[k]=delta(ae.mean(0),abs(pred['orthographic/features']-y).mean(0))
    q=qa['people'];speedrel=np.array([(r['speed2D']-r['speed3D'])/r['speed3D'] for r in q]);leg=np.array([r['projected_leg_over3D'] for r in q]);knee=np.array([r['knee_mean_difference_degrees'] for r in q])
    result={'summary':summary,'comparisons_to_original_features':comparison,
      'geometry':{'speed_relative_bias_median':float(np.median(speedrel)),'speed_abs_relative_error_p95':float(np.quantile(abs(speedrel),.95)),
         'leg_length_ratio_median':float(np.median(leg)),'leg_length_ratio_p05':float(np.quantile(leg,.05)),
         'knee_angle_mean_abs_difference_median_degrees':float(np.median(knee)),'knee_angle_difference_p95_degrees':float(np.quantile(knee,.95)),
         'original_feature_max_rebuild_difference':qa['original_feature_max_difference']},'limits':prot['limits']}
    p.save(HERE/'projection-results.json',result)
    np.savez_compressed(HEAVY/'projection/verified-predictions.npz',y=y,ids=ids,**{k.replace('/','__'):z for k,z in pred.items()})
    p.save(HERE/'projection-verification.json',{'passed':True,'models_replayed':len(recs),'raw_windows_features_replayed':297*6,
      'max_replay_abs':maxdiff,'results_sha256':p.sha(HERE/'projection-results.json'),'protocol_sha256':p.sha(HERE/'projection-protocol.json')})
    print('PROJECTION_VERIFIED',flush=True)
if __name__=='__main__':{'sensor':sensor,'projection':projection}[sys.argv[1]]()
