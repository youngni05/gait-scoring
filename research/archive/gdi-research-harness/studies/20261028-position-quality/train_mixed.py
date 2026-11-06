import csv,json
import numpy as np
from analyze_quality import HERE,H,ROOT,OLD,PREV,load,sha,save

def persons(values,ids):return np.array([values[ids==p].mean() for p in sorted(set(ids))])
def comparison(old,new):
    d=old-new;rng=np.random.default_rng(20261026)
    ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1),[2.5,97.5])
    return dict(reduction=float(d.mean()),paired_bootstrap95=ci.tolist(),improved=int((d>0).sum()),worsened=int((d<0).sum()))

def main():
    assert not (HERE/'mixed_result.json').exists()
    before=load(HERE/'sources.json');assert all(sha(ROOT/p)==d for p,d in before.items())
    packet=H/'jobs/development-baseline/runs/20261025T044558572569Z-71adb0/development_packet.npz'
    expected=load(OLD/'analysis_sources.json')['sha256'][packet.relative_to(ROOT).as_posix()]
    assert sha(packet)==expected
    with np.load(OLD/'raw_features.npz',allow_pickle=False) as f,np.load(packet,allow_pickle=False) as d:
        ids,vid,split=(f[k].copy() for k in ('patient_ids','video_ids','split'))
        for k in ('patient_ids','video_ids','split'):np.testing.assert_array_equal(f[k],d[k])
        x={m:f[m+'_local'].copy() for m in ('front','middle','last')};y=d['Y'].copy()
    train,val=split=='train',split=='validation'
    assert train.sum()==203 and val.sum()==40 and not set(ids[train])&set(ids[val])
    a=np.concatenate([x[m][train] for m in x]);b=np.tile(y[train],(3,1))
    mu=a.mean(axis=0);scale=a.std(axis=0);scale[scale==0]=1
    normalized=(a-mu)/scale;center=normalized.mean(axis=0);ym=b.mean(axis=0)
    aw=(normalized-center)/np.sqrt(3);bw=(b-ym)/np.sqrt(3)
    u,s,vh=np.linalg.svd(aw,full_matrices=False)
    coef=(vh.T@((s/(s*s+1))[:,None]*(u.T@bw))).T
    intercept=ym-coef@center
    direct=np.linalg.solve(aw.T@aw+np.eye(aw.shape[1]),aw.T@bw).T
    np.testing.assert_allclose(coef,direct,rtol=0,atol=1e-7)
    p={m:((xx[val]-mu)/scale)@coef.T+intercept for m,xx in x.items()}
    p['mean3']=np.stack(list(p.values())).mean(axis=0)
    np.savez_compressed(HERE/'mixed_model.npz',scaler_mean=mu,scaler_scale=scale,coef=coef,intercept=intercept)
    with np.load(HERE/'mixed_model.npz',allow_pickle=False) as z:
        for m in x:np.testing.assert_allclose(((x[m][val]-z['scaler_mean'])/z['scaler_scale'])@z['coef'].T+z['intercept'],p[m],rtol=0,atol=1e-10)
    with (PREV/'predictions.csv').open(encoding='utf-8') as f:
        old={(r['video_id'],r['method']):[float(r['prediction_L']),float(r['prediction_R'])] for r in csv.DictReader(f)}
    po={m:np.array([old[(str(v),m)] for v in vid[val]]) for m in p}
    errors={m:persons(np.abs(pp-y[val]).mean(axis=1),ids[val]) for m,pp in p.items()}
    old_errors={m:persons(np.abs(pp-y[val]).mean(axis=1),ids[val]) for m,pp in po.items()}
    r=persons(np.ptp(np.stack([p[m] for m in x]),axis=0).mean(axis=1),ids[val])
    ro=persons(np.ptp(np.stack([po[m] for m in x]),axis=0).mean(axis=1),ids[val])
    with (HERE/'mixed_predictions.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','video_id','method','actual_L','actual_R','prediction_L','prediction_R'])
        for m,values in p.items():
            for i,v in enumerate(vid[val]):w.writerow([ids[val][i],v,m,*y[val][i],*values[i]])
    assert all(sha(ROOT/path)==d for path,d in before.items()) and sha(packet)==expected
    result=dict(train_people=175,train_records=203,train_windows=609,per_window_weight=1/3,
        validation_people=37,validation_records=40,new_models_fitted=1,encoder_fine_tuned=False,
        person_mae={m:float(e.mean()) for m,e in errors.items()},
        same_input_vs_original={m:comparison(old_errors[m],e) for m,e in errors.items()},
        mixed_mean3_vs_original_front=comparison(old_errors['front'],errors['mean3']),
        three_position_range_mean=float(r.mean()),range_reduction=comparison(ro,r),
        coefficient_equation_max_difference=float(np.max(np.abs(coef-direct))),
        source_hashes_unchanged=True,independent_final_evaluation=False,
        note='Adaptive exploratory experiment after initial diagnostics; no hyperparameter search or new labels.')
    save('mixed_result.json',result)
    save('mixed_sources.json',{str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'plan.md',packet)})
    print(json.dumps(result,ensure_ascii=False,indent=2))

from pathlib import Path
if __name__=='__main__':main()
