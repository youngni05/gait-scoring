"""Independent recomputation of all input descriptors and reported associations."""
import csv
import math
import warnings
from statistics import mean, pstdev
import run as r
from scipy.stats import spearmanr
np, m, HERE = r.np, r.m, r.HERE


def rank_average(a):
    order=sorted(range(len(a)),key=lambda i:a[i])
    out=np.zeros(len(a)); start=0
    while start<len(order):
        end=start+1
        while end<len(order) and a[order[end]]==a[order[start]]:
            end+=1
        for i in order[start:end]: out[i]=(start+1+end)/2
        start=end
    return out


def main():
    source=m.load_json(HERE/'source_hashes.json')
    for path,digest in source.items(): assert m.digest(m.WORKSPACE/path)==digest,path
    d=r.load_npz(r.INPUT); p=r.load_npz(r.BLEND/'oof_predictions.npz')
    rec=list(csv.DictReader((HERE/'record_input_features.csv').open(encoding='utf-8-sig')))
    persons=list(csv.DictReader((HERE/'person_input_errors.csv').open(encoding='utf-8-sig')))
    s=m.load_json(HERE/'summary.json')
    worst=0.0
    zero=observed_boundary=missing_boundary=0
    for i,row in enumerate(rec):
        xy=d['x'][i,0,:,:,:2].astype(float); c=d['x'][i,0,:,:,2].astype(float)
        assert row['person_id']==str(d['person_id'][i]) and row['video_id']==str(d['video_id'][i])
        assert int(row['fold'])==d['fold'][i]
        b=np.maximum(abs(xy[:,:,0]),abs(xy[:,:,1]))>=1-1e-6
        obs=c>0; zero+=int((~obs).sum()); observed_boundary+=int((obs&b).sum()); missing_boundary+=int((~obs&b).sum())
        upper=[9,10,11,12,13,14,15,16]
        vals=dict(upper_missing_fraction=np.count_nonzero(c[:,upper]==0)/(124*8),
                  upper_confidence_mean=mean(c[:,upper].flatten()),
                  leg_missing_fraction=np.count_nonzero(c[:,1:7]==0)/(124*6),
                  leg_confidence_mean=mean(c[:,1:7].flatten()),
                  observed_boundary_fraction=np.count_nonzero(obs&b)/np.count_nonzero(obs),
                  missing_boundary_fraction_all_points=np.count_nonzero(~obs&b)/(124*17))
        steps=xy[1:,1:7]-xy[:-1,1:7]
        acc=xy[2:,1:7]-2*xy[1:-1,1:7]+xy[:-2,1:7]
        vals['leg_step_p95']=np.quantile(np.sqrt(np.sum(steps**2,axis=-1)),.95)
        vals['leg_acceleration_p95']=np.quantile(np.sqrt(np.sum(acc**2,axis=-1)),.95)
        ratios=[]
        for a,bone in [(1,2),(2,3),(4,5),(5,6)]:
            lengths=np.hypot(xy[:,a,0]-xy[:,bone,0],xy[:,a,1]-xy[:,bone,1]).tolist()
            ratios.append(pstdev(lengths)/mean(lengths))
        vals['leg_length_cv_mean']=mean(ratios)
        for key,value in vals.items():
            diff=abs(float(row[key])-value);worst=max(worst,diff)
            assert diff<2e-7,(i,key,diff)
    assert zero==s['inventory']['zero_confidence_points']==72097
    assert missing_boundary==s['inventory']['missing_boundary_points']==zero
    assert observed_boundary==s['inventory']['observed_boundary_points']
    person_ids=np.unique(d['person_id'])
    assert [q['person_id'] for q in persons]==person_ids.tolist()
    for row in persons:
        idx=np.flatnonzero(d['person_id']==row['person_id']); f=int(row['fold'])
        for k in r.FEATURES:
            assert abs(float(row[k])-mean(float(rec[i][k]) for i in idx))<1e-12
        errors=[]
        for method,col in [('pls32','pls_MAE'),('gpu_last_stage','gpu_MAE')]:
            pred=p['predictions'][p['methods'].tolist().index(method),idx]
            err=mean(abs(pred-d['y'][idx]).flatten().tolist())
            errors.append(err); assert abs(float(row[col])-err)<1e-10
        assert abs(float(row['common_error'])-min(errors))<1e-10
        assert abs(float(row['actual_mean_GDI'])-mean(d['y'][idx].flatten()))<1e-10
        train=np.unique(d['person_id'][d['fold']!=f]); assert len(train)==556 and row['person_id'] not in train
        train_mean=np.array([mean([mean(d['y'][d['person_id']==pid,side]) for pid in train]) for side in range(2)])
        assert abs(float(row['target_distance_from_outer_training_mean'])-mean(abs(d['y'][idx]-train_mean).flatten()))<1e-10
        assert abs(float(row['actual_left_right_difference'])-mean(abs(d['y'][idx,0]-d['y'][idx,1])))<1e-10
    groups={}
    for i,row in enumerate(persons): groups.setdefault((math.floor(float(row['actual_mean_GDI'])/10),int(row['cohort'])),[]).append(i)
    eligible=[idx for idx in groups.values() if len(idx)>=5]
    for assoc in s['associations']:
        a=[float(q[assoc['feature']]) for q in persons];b=[float(q[assoc['outcome']]) for q in persons]
        if max(a)==min(a):
            assert assoc['spearman'] is None and assoc['within_score_cohort_rank_correlation'] is None
            continue
        assert abs(float(spearmanr(a,b).statistic)-assoc['spearman'])<1e-10
        ra,rb=rank_average(a),rank_average(b)
        residuals=[np.concatenate([ranks[idx]-mean(ranks[idx]) for idx in eligible]) for ranks in [ra,rb]]
        expected=float(np.dot(*residuals)/np.sqrt(sum(residuals[0]**2)*sum(residuals[1]**2)))
        assert abs(expected-assoc['within_score_cohort_rank_correlation'])<1e-10
    review=list(csv.DictReader((HERE/'review_top70.csv').open(encoding='utf-8-sig')))
    expected=sorted(persons,key=lambda q:(-float(q['common_error']),q['person_id']))[:70]
    assert [q['person_id'] for q in review]==[q['person_id'] for q in expected]
    for feature in s['feature_summary']:
        a=[float(q[feature['feature']]) for q in persons]
        assert abs(np.median(a)-feature['median'])<1e-12
        assert abs(mean(float(q[feature['feature']]) for q in review)-feature['top70_mean'])<1e-12
    side=m.load_json(HERE/'side_diagnostic.json')
    for row in side['rows']:
        pred=p['predictions'][p['methods'].tolist().index(row['method'])]
        per_person=[]
        for pid in person_ids:
            values=[]
            for i in np.flatnonzero(d['person_id']==pid):
                yl,yr=map(float,d['y'][i]);pl,pr=map(float,pred[i])
                meanerr=abs((pl+pr-yl-yr)/2);halferr=abs((pl-pr-yl+yr)/2)
                mae=(abs(pl-yl)+abs(pr-yr))/2
                assert abs(mae-max(meanerr,halferr))<1e-12
                values.append([abs(yl-yr),abs(pl-pr),2*halferr,meanerr,mae,int(halferr>meanerr)])
            per_person.append([mean(col) for col in zip(*values)])
        keys=['actual_abs_LR_difference','predicted_abs_LR_difference','LR_difference_MAE','mean_score_MAE',
              'person_MAE','fraction_side_difference_error_dominates']
        for key,col in zip(keys,zip(*per_person)):
            assert abs(mean(col)-row[key])<1e-10,key
    result=dict(verified=True,record_descriptors_checked=1064*9,person_rows_checked=695,
                association_rows_checked=len(s['associations']),review_people_checked=70,within_strata_people=sum(map(len,eligible)),
                max_descriptor_difference=worst,input_identity_and_existing_errors_verified=True,
                source_files_unchanged=len(source),side_diagnostic_metrics_checked=18,
                boundary_precision_correction_verified=True,GPU_used=False,training=False,causal_validation=False)
    r.save('verification.json',result)
    print(result)


if __name__=='__main__': main()
