"""Independent selected refits, feature construction, choices and metrics."""
import csv, importlib.util
from statistics import mean
import run as r
np,p,m,HERE=r.np,r.p,r.m,r.HERE

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
old=module('previous_verifier',r.PREV/'verify.py')
scalar=module('scalar_verifier',p.BLEND/'verify.py')

def temporal_reference(raw):
    features=[]
    for record in raw:
        xy=record[:,:,:2].astype(float);sides=[];stats=[]
        for indices in [[4,5,6],[1,2,3]]:
            hip,knee,ankle=[xy[:,j] for j in indices]
            a,b=hip-knee,ankle-knee
            den=np.sqrt(np.sum(a*a,1))*np.sqrt(np.sum(b*b,1))
            angle=180-np.arccos(np.clip(np.sum(a*b,1)/den,-1,1))*180/np.pi
            signals=np.column_stack([xy[:,j,k]-xy[:,0,k] for j in indices for k in [0,1]]+[angle])
            sides.append(signals);ss=[]
            for k in range(7):
                first=signals[1:,k]-signals[:-1,k]
                ss.extend([np.mean(abs(first)),np.std(first),np.mean(abs(first[1:]-first[:-1]))])
            stats.append(np.array(ss))
        cc=[]
        for lag in range(-15,16):
            indices=[(t,t-lag) for t in range(124) if 0<=t-lag<124]
            a=sides[0][[i for i,j in indices]];b=sides[1][[j for i,j in indices]]
            a=a-np.mean(a,0);b=b-np.mean(b,0);den=np.sqrt(np.sum(a*a,0)*np.sum(b*b,0))
            cc.append(np.divide(np.sum(a*b,0),den,out=np.zeros(7),where=den>1e-12))
        cc=np.array(cc)
        features.append(np.r_[stats[0],stats[1],stats[0]-stats[1],cc[15],np.max(cc,0),(np.argmax(cc,0)[[2,4,6]]-15)/15])
    return np.array(features)

def equal(row,actual):
    for key,value in actual.items():
        if value is None:assert row[key] is None
        else:assert abs(row[key]-value)<1e-9,(key,row[key],value)

def main():
    s=m.load_json(HERE/'summary.json');d=p.load(p.PARENT/'development695.npz');o=p.load(HERE/'oof_predictions.npz')
    cube=p.load(HERE/'fold_predictions.npz')['predictions'];xs,temp=r.features(d)
    for key in ['person_id','video_id','fold','y','source_group']:np.testing.assert_array_equal(d[key],o[key])
    sources=m.load_json(HERE/'source_hashes.json')
    for path,digest in sources.items():assert m.digest(m.WORKSPACE/path)==digest,path
    reference=temporal_reference(p.load(p.GPU/'expansion695/train_only.npz')['x'][:,0])
    np.testing.assert_allclose(reference,temp,rtol=0,atol=1e-9)
    np.testing.assert_array_equal(temp,p.load(HERE/'temporal_features.npz')['X'])
    assert {k:v.shape[1] for k,v in xs.items()}==dict(mb=512,leg=164,combined=676,position=108,angle=18,length=36,no_length=128,leg_temporal=244,combined_temporal=756)
    print('temporal features verified',flush=True)
    scores=list(csv.DictReader((HERE/'inner_scores.csv').open(encoding='utf-8-sig')))
    refits=0;inner_refits=0;maxdiff=0.;candidatechecks=0
    with p.threadpool_limits(limits=2):
        for f in range(5):
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            assert len(set(d['person_id'][tr]))==556 and len(set(d['person_id'][te]))==139
            assert not set(d['person_id'][tr])&set(d['person_id'][te])
            saved=p.load(HERE/f'models/fold{f}_inner.npz');assignment=saved['inner_assignment']
            np.testing.assert_array_equal(tr,saved['train_indices'])
            guard=[dict(task='previous_leg_gap',candidate='unchanged',score=old.stats(d['y'][tr],saved['previous_predictions'],d['person_id'][tr])['overall_MAE'])]
            for row in [v for v in scores if int(v['fold'])==f]:
                name,key=row['task'],row['candidate'];family,which,kind=r.TASKS[name]
                v=saved[name+'__'+key];y=r.target(d['y'][tr],which)
                per=[mean(abs(y[d['person_id'][tr]==pid]-v[d['person_id'][tr]==pid])) for pid in np.unique(d['person_id'][tr])]
                assert abs(mean(per)-float(row['target_MAE']))<1e-9
                pred=r.reconstruct(name,v,saved['base_predictions'],saved['previous_predictions'])
                assert abs(old.stats(d['y'][tr],pred,d['person_id'][tr])['overall_MAE']-float(row['overall_MAE']))<1e-9
                candidatechecks+=1
                if family in ['mb','leg','combined'] and which=='mean':guard.append(dict(task=name,candidate=key,score=float(row['overall_MAE'])))
            for row in [v for v in s['selections'] if v['fold']==f]:
                name,key,method=row['task'],row['candidate'],row['method']
                if method=='inner_selected_mean':
                    expect=min(guard,key=lambda v:v['score']);assert (name,key)==(expect['task'],expect['candidate'])
                else:
                    candidates=[v for v in scores if int(v['fold'])==f and v['task']==name]
                    assert key==min(candidates,key=lambda v:float(v['target_MAE']))['candidate']
                if name=='previous_leg_gap':np.testing.assert_array_equal(cube[-1,f],cube[1,f]);continue
                family,which,kind=r.TASKS[name];x=xs[family];y=r.target(d['y'],which)
                v=old.ref_fit_predict(x[tr],y[tr],x,key);refits+=1
                expected=r.reconstruct(name,v,cube[0,f],cube[1,f])
                actual=cube[r.METHODS.index(method),f]
                maxdiff=max(maxdiff,float(abs(expected-actual).max()))
                np.testing.assert_allclose(expected,actual,atol=1e-7,rtol=0)
                model=p.load(HERE/f'models/fold{f}_{method}.npz')
                np.testing.assert_array_equal(model['train_indices'],tr);np.testing.assert_array_equal(model['evaluation_indices'],te)
                for j in range(4):
                    ti=tr[assignment!=j];vi=tr[assignment==j]
                    assert len(set(d['person_id'][ti]))==417 and len(set(d['person_id'][vi]))==139
                    assert not set(d['person_id'][ti])&set(d['person_id'][vi])
                    v=old.ref_fit_predict(x[ti],y[ti],x[vi],key);inner_refits+=1
                    np.testing.assert_allclose(v,saved[name+'__'+key][assignment==j],atol=1e-7,rtol=0)
            for family in ['leg_temporal','combined_temporal']:
                a=cube[r.METHODS.index('mean_'+family+'_ridge'),f];b=cube[r.METHODS.index('gap_'+family+'_ridge'),f]
                expected=np.column_stack([a.mean(1)+(b[:,0]-b[:,1])/2,a.mean(1)-(b[:,0]-b[:,1])/2])
                np.testing.assert_array_equal(expected,cube[r.METHODS.index('joint_'+family),f])
            print('selected refits verified fold',f,flush=True)
    prev=p.load(r.PREV/'fold_predictions.npz')
    np.testing.assert_array_equal(cube[0],prev['predictions'][0])
    np.testing.assert_array_equal(cube[1],prev['predictions'][list(prev['methods']).index('leg_gap_ridge')])
    for i,name in enumerate(r.METHODS):np.testing.assert_array_equal(o['predictions'][i],cube[i,d['fold'],np.arange(1064)])
    checks=0
    for row in s['metrics']:
        mask=np.ones(1064,dtype=bool) if row['group']=='all695' else d['source_group']==row['group']
        pred=o['predictions'][r.METHODS.index(row['method'])]
        equal(row,old.stats(d['y'][mask],pred[mask],d['person_id'][mask]))
        equal(row,scalar.independent_metrics(d,pred,selected=np.flatnonzero(mask)));checks+=1
    for row in s['bands']:
        equal(row,scalar.independent_metrics(d,o['predictions'][r.METHODS.index(row['method'])],band=row['actual_bin_low']//10));checks+=1
    for row in s['train_held']:
        f=row['fold'];mask=(d['fold']==f) if row['split']=='held' else (d['fold']!=f)
        equal(row,old.stats(d['y'][mask],cube[r.METHODS.index(row['method']),f,mask],d['person_id'][mask]));checks+=1
    draws=np.random.default_rng(20261029).integers(0,695,(10000,695));ids=np.unique(d['person_id'])
    for row in s['comparisons']:
        arrays=[]
        for name in [row['reference'],row['method']]:
            error=abs(o['predictions'][r.METHODS.index(name)]-d['y'])
            arrays.append(np.array([mean(error[d['person_id']==pid].flatten()) for pid in ids]))
        delta=arrays[0]-arrays[1]
        assert abs(mean(delta)-row['reduction'])<1e-10
        np.testing.assert_allclose(np.quantile(delta[draws].mean(1),[.025,.975]),row['bootstrap95'],atol=1e-9,rtol=0)
    result=dict(verified=True,people=695,records=1064,temporal_values_checked=temp.size,max_temporal_difference=float(abs(temp-reference).max()),
        independent_outer_refits=refits,independent_inner_refits=inner_refits,max_prediction_difference=maxdiff,
        candidate_rows_checked=candidatechecks,metric_rows_checked=checks,bootstrap_comparisons_checked=len(s['comparisons']),
        sources_unchanged=len(sources),GPU_used=False,protected_numeric_data_used=False,independent_final_evaluation=False)
    r.js('verification.json',result);print(result)

if __name__=='__main__':main()
