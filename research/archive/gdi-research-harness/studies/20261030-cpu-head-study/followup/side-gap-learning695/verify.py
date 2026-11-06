"""Verify nested choices, independent selected refits, raw features and held metrics."""
import csv
from statistics import mean,pstdev
import run as r
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits
np,m,HERE=r.np,r.m,r.HERE


def ref_fit_predict(x,y,ev,key):
    if key=='zero':return np.zeros(len(ev))
    if key=='training_record_mean':return np.full(len(ev),y.mean())
    scaler=StandardScaler().fit(x);z=scaler.transform(x)
    model=Ridge(alpha=float(key[6:]),solver='svd') if key.startswith('alpha_') else PLSRegression(n_components=int(key[2:]),scale=False,max_iter=500,tol=1e-6)
    return model.fit(z,y).predict(scaler.transform(ev)).reshape(-1)


def stats(y,p,ids):
    per=[];a=[];b=[];direction=[]
    for pid in np.unique(ids):
        indices=np.flatnonzero(ids==pid);tu=y[indices];pr=p[indices]
        td=(tu[:,0]-tu[:,1]).tolist();pd=(pr[:,0]-pr[:,1]).tolist()
        per.append([mean(abs(pr-tu).flatten()),mean(abs(v-u) for u,v in zip(td,pd)),
                    mean(map(abs,pd)),mean(map(abs,td)),mean(abs(pr.mean(1)-tu.mean(1)))])
        a.append(mean(td));b.append(mean(pd))
        signs=[int(np.sign(v)==np.sign(u)) for u,v in zip(td,pd) if abs(u)>1e-12]
        if signs:direction.append(mean(signs))
    avg=[mean(c) for c in zip(*per)]
    return dict(people=len(per),overall_MAE=avg[0],gap_MAE=avg[1],predicted_gap_magnitude=avg[2],
                actual_gap_magnitude=avg[3],mean_score_MAE=avg[4],direction_accuracy=mean(direction),
                signed_person_gap_correlation=float(np.corrcoef(a,b)[0,1]) if np.ptp(b)>1e-12 else None)


def main():
    s=m.load_json(HERE/'summary.json');d=r.load(r.PARENT/'development695.npz');o=r.load(HERE/'oof_predictions.npz')
    cube=r.load(HERE/'fold_predictions.npz')['predictions'];raw=r.load(HERE/'leg_features.npz')['X']
    for key in ['y','person_id','video_id','fold','source_group']:np.testing.assert_array_equal(d[key],o[key])
    sources=m.load_json(HERE/'source_hashes.json')
    for path,digest in sources.items():assert m.digest(m.WORKSPACE/path)==digest,path
    x=r.load(r.GPU/'expansion695/train_only.npz')['x'][:,0]
    # Independent feature construction per record, using explicit side and joint ordering.
    max_feature=0.
    for i in range(1064):
        xy=x[i,:,:,:2].astype(float);sides=[];valids=[]
        for joints in [[4,5,6],[1,2,3]]:
            a=xy[:,joints[0]]-xy[:,joints[1]];b=xy[:,joints[2]]-xy[:,joints[1]]
            la=np.hypot(a[:,0],a[:,1]);lb=np.hypot(b[:,0],b[:,1]);valid=la*lb>1e-12
            flex=np.full(124,np.nan)
            flex[valid]=180-np.arccos(np.clip(np.sum(a[valid]*b[valid],axis=1)/(la[valid]*lb[valid]),-1,1))*180/np.pi
            signals=[xy[:,j,axis]-xy[:,0,axis] for j in joints for axis in range(2)]+[flex,la,lb]
            flat=[]
            for col in signals:
                use=col[np.isfinite(col)]
                flat.extend(np.quantile(use,[.1,.25,.5,.75,.9]).tolist()+[pstdev(use.tolist())])
            sides.append(np.array(flat));valids.append(valid.mean())
        expected=np.concatenate([sides[0],sides[1],sides[0]-sides[1],valids])
        max_feature=max(max_feature,float(abs(expected-raw[i]).max()))
        np.testing.assert_allclose(expected,raw[i],atol=1e-9,rtol=0)
    Xs={'mb_gap_ridge':d['X'],'mb_gap_pls':d['X'],'leg_gap_ridge':raw,'combined_gap_ridge':np.concatenate([d['X'],raw],axis=1)}
    choices=list(csv.DictReader((HERE/'selections.csv').open(encoding='utf-8-sig')))
    candidates=list(csv.DictReader((HERE/'inner_candidate_scores.csv').open(encoding='utf-8-sig')))
    max_pred=0.;inner_refits=0;outer_refits=0;checked_candidates=0
    with threadpool_limits(limits=2):
        for f in range(5):
            tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            y=d['y'][tr];target=y[:,0]-y[:,1];ids=d['person_id'][tr]
            inner=r.load(HERE/f'models/fold{f}_inner_predictions.npz');assignment=inner['inner_assignment']
            np.testing.assert_array_equal(inner['outer_train_indices'],tr)
            basemean=inner['base_inner_predictions'].mean(1)
            score_rows=[]
            for row in [q for q in candidates if int(q['outer_fold'])==f]:
                delta=inner[row['family']+'__'+row['candidate']]
                candidate=np.column_stack([basemean+delta*.5,basemean-delta*.5])
                metric=stats(y,candidate,ids)
                assert abs(metric['gap_MAE']-float(row['inner_gap_MAE']))<1e-10
                assert abs(metric['overall_MAE']-float(row['inner_overall_MAE']))<1e-10
                score_rows.append(row);checked_candidates+=1
            baseline=stats(y,inner['base_inner_predictions'],ids)
            for row in [q for q in choices if int(q['outer_fold'])==f]:
                family,key,method=row['selected_family'],row['candidate'],row['method']
                if method=='guarded_gap':
                    allowed=[dict(family='joint_pls32',candidate='unchanged',inner_gap_MAE=baseline['gap_MAE'],inner_overall_MAE=baseline['overall_MAE'])]
                    allowed += [q for q in score_rows if float(q['inner_overall_MAE'])<=baseline['overall_MAE']+1e-12]
                    expected=min(allowed,key=lambda q:float(q['inner_gap_MAE']))
                else:expected=min([q for q in score_rows if q['family']==family],key=lambda q:float(q['inner_gap_MAE']))
                assert key==expected['candidate'] and family==expected['family']
                mi=r.METHODS.index(method)
                if family=='joint_pls32':np.testing.assert_array_equal(cube[mi,f],cube[0,f]);continue
                z=Xs[family]
                pred=ref_fit_predict(z[tr],target,z,key);outer_refits+=1
                expected_pred=np.column_stack([cube[0,f].mean(1)+pred/2,cube[0,f].mean(1)-pred/2])
                diff=float(abs(cube[mi,f]-expected_pred).max());max_pred=max(max_pred,diff)
                np.testing.assert_allclose(cube[mi,f],expected_pred,rtol=0,atol=1e-7)
                model=r.load(HERE/f'models/fold{f}_{method}.npz')
                np.testing.assert_array_equal(tr,model['train_indices']);np.testing.assert_array_equal(te,model['evaluation_indices'])
                for j in range(4):
                    ti=np.flatnonzero(assignment!=j);vi=np.flatnonzero(assignment==j)
                    assert len(set(ids[ti]))==417 and len(set(ids[vi]))==139 and not set(ids[ti])&set(ids[vi])
                    delta=ref_fit_predict(z[tr[ti]],target[ti],z[tr[vi]],key);inner_refits+=1
                    np.testing.assert_allclose(delta,inner[family+'__'+key][vi],rtol=0,atol=1e-7)
            # Controls and unchanged mean checked independently in every fold.
            for mi,delta in [(1,0.),(2,target.mean())]:
                np.testing.assert_allclose(cube[mi,f,:,0],cube[0,f].mean(1)+delta/2,rtol=0,atol=1e-12)
                np.testing.assert_allclose(cube[mi,f,:,1],cube[0,f].mean(1)-delta/2,rtol=0,atol=1e-12)
    for i,name in enumerate(r.METHODS):
        np.testing.assert_array_equal(o['predictions'][i],cube[i,d['fold'],np.arange(1064)])
        np.testing.assert_allclose(o['predictions'][i].mean(1),o['predictions'][0].mean(1),rtol=0,atol=2e-14)
    checks=0
    for row in s['metrics']:
        use=np.ones(1064,dtype=bool) if row['group']=='all695' else d['source_group']==row['group']
        actual=stats(d['y'][use],o['predictions'][r.METHODS.index(row['method']),use],d['person_id'][use])
        for k,v in actual.items():
            if v is None:assert row[k] is None
            else:assert abs(v-row[k])<1e-10,(row['method'],k)
        checks+=1
    for row in s['train_held']:
        f=row['fold'];use=(d['fold']==f) if row['split']=='held' else (d['fold']!=f)
        actual=stats(d['y'][use],cube[r.METHODS.index(row['method']),f,use],d['person_id'][use])
        for k,v in actual.items():
            if v is None:assert row[k] is None
            else:assert abs(v-row[k])<1e-10
        checks+=1
    # Match bins and tail bias with a separately grouped scalar implementation from the verified blend study.
    spec=r.importlib.util.spec_from_file_location('scalar_metrics',r.BLEND/'verify.py')
    scalar=r.importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)
    for row in s['score_bands']:
        actual=scalar.independent_metrics(d,o['predictions'][r.METHODS.index(row['method'])],band=row['actual_bin_low']//10)
        for k,v in actual.items():assert abs(v-row[k])<1e-10
        checks+=1
    draws=np.random.default_rng(20261029).integers(0,695,(10000,695));person_ids=np.unique(d['person_id'])
    for row in s['comparisons']:
        i=r.METHODS.index(row['method']);arrays=[]
        for mi in [0,i]:
            pred=o['predictions'][mi]
            e=abs(pred-d['y']) if row['metric']=='overall_MAE' else abs(pred[:,0]-pred[:,1]-d['y'][:,0]+d['y'][:,1])
            arrays.append(np.array([mean(e[d['person_id']==pid].flatten()) for pid in person_ids]))
        delta=arrays[0]-arrays[1];np.testing.assert_allclose(np.quantile(np.mean(delta[draws],axis=1),[.025,.975]),row['bootstrap95'],atol=1e-10,rtol=0)
        assert abs(mean(delta)-row['reduction'])<1e-10
    result=dict(verified=True,people=695,records=1064,raw_feature_values_checked=1064*164,max_raw_feature_difference=max_feature,
        independent_selected_outer_refits=outer_refits,independent_selected_inner_refits=inner_refits,max_prediction_difference=max_pred,
        nested_candidate_rows_checked=checked_candidates,metric_rows_checked=checks,bootstrap_comparisons_checked=len(s['comparisons']),
        mean_predictions_unchanged=True,source_files_unchanged=len(sources),GPU_used=False,protected_numeric_data_used=False,
        independent_final_evaluation=False)
    r.js('verification.json',result);print(result)


if __name__=='__main__':main()
