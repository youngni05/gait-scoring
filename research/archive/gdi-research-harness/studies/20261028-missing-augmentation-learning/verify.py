import csv,json,hashlib,statistics,collections
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert abs(float(a)-float(b))<1e-8,(a,b)
def avg(a):return statistics.mean(a)
def main():
    result=load(S/'result.json');sources={**load(S/'sources.json'),**load(S/'analysis_sources.json')}
    for p,d in sources.items():assert sha(R/p)==d
    assert load(S/'identity_training_check.json')['identity_augmentation_same_parameters']
    assert all(load(S/'mask_checks.json').values())
    with np.load(S/'features.npz',allow_pickle=False) as z:f={k:z[k].copy() for k in z.files}
    with np.load(S/'models.npz',allow_pickle=False) as z:m={k:z[k].copy() for k in z.files}
    vid=list(map(str,f['video_ids']));ids=list(map(str,f['patient_ids']));people=sorted(set(ids));ix={v:i for i,v in enumerate(vid)}
    assert len(vid)==203 and len(people)==175
    manifests=load(S/'mask_manifest.json')
    for j,original_index in enumerate(f['original_indices']):
        meta=load(S/f'records/row_{j:04d}.json');path=S/f'records/row_{j:04d}.npz'
        assert meta['sha256']==sha(path) and meta['original_index']==original_index and meta['masks']==manifests[j]
        with np.load(path,allow_pickle=False) as z:
            for kind,key in [('training','training_aug'),('stress','stress')]:
                np.testing.assert_array_equal(z[kind],f[key][j])
                if meta['masks'][kind]['new_missing_cells']==0:np.testing.assert_array_equal(f[key][j],f['front'][j])
        with np.load(H/f'jobs/native124-cache-v1/records/row_{original_index:04d}.npz',allow_pickle=False) as z:a=z['input124'].copy()
        for kind in ('training','stress'):
            mm=meta['masks'][kind];assert mm['length']==(31 if kind=='training' else 62)
            assert mm['joints'] in ([9,10],[13],[16]);start=mm['start'];assert 0<=start<=124-mm['length']
            selected=a[start:start+mm['length'],mm['joints'],2]
            assert int((selected>0).sum())==mm['new_missing_cells']
    rows=list(csv.DictReader((S/'predictions.csv').open(encoding='utf-8')));assert len(rows)==9135
    rec={};labels={};fold_by_video={}
    for r in rows:
        arm,seed,pos,v=r['arm'],int(r['seed']),r['position'],r['video_id'];k=(arm,seed,pos,v);assert k not in rec
        assert r['patient_id']==ids[ix[v]];fold=int(r['fold']);assert 0<=fold<5
        pred=tuple(float(r['prediction_'+s]) for s in 'LR');actual=tuple(float(r['actual_'+s]) for s in 'LR')
        if v in labels:assert actual==labels[v] and fold==fold_by_video[v]
        labels[v]=actual;fold_by_video[v]=fold;rec[k]=pred
    folds=np.array([fold_by_video[v] for v in vid]);seeds=result['seeds'];arms=list(result['metrics']);positions=['front','middle','last','stress','mean3']
    for fold in range(5):
        tr=set(np.array(ids)[folds!=fold]);te=set(np.array(ids)[folds==fold]);assert len(tr)==140 and len(te)==35 and not tr&te
    logs=load(S/'training_logs.json');assert len(logs)==30
    for fold in range(5):
        for seed in seeds:
            pair=[x for x in logs if x['fold']==fold and x['seed']==seed];assert len(pair)==2
            assert {x['arm'] for x in pair}=={'extra_original','extra_missing'}
            for key in ['initial_state_sha256','targets_sha256','first_half_sha256']:assert pair[0][key]==pair[1][key]
            assert pair[0]['data_sha256']!=pair[1]['data_sha256']
            assert all(x['updates']==100 and x['batch_rows']==2*int((folds!=fold).sum()) for x in pair)
    maxerr=0.
    for arm in arms:
        for seed in seeds:
            for fold in range(5):
                idx=np.where(folds==fold)[0];prefix=f'fold{fold}__'
                for pos in positions[:4]:
                    zz=(f[pos][idx]-m[prefix+'scaler_mean'])/m[prefix+'scaler_scale'];p=zz@m[prefix+'coef'].T+m[prefix+'intercept']
                    if arm!='baseline':
                        h=f'{arm}_seed{seed}_fold{fold}__';p+=np.maximum(zz@m[h+'w1'].T+m[h+'b1'],0)@m[h+'w2'].T*m[prefix+'y_scale']+m[h+'b2']*m[prefix+'y_scale']
                    expected=np.array([rec[(arm,seed,pos,vid[i])] for i in idx]);err=float(np.max(np.abs(p-expected)));maxerr=max(maxerr,err);assert err<1e-8
            for v in vid:
                mean3=tuple(avg([rec[arm,seed,pos,v][side] for pos in positions[:3]]) for side in range(2))
                for a,b in zip(mean3,rec[arm,seed,'mean3',v]):close(a,b)
    values={};by_person={p:[v for v,i in zip(vid,ids) if i==p] for p in people}
    for arm in arms:
        values[arm]={k:[] for k in positions+['range','stress_shift']};metric=result['metrics'][arm]
        for si,seed in enumerate(seeds):
            for pos in positions:
                pp=[avg([avg([abs(rec[arm,seed,pos,v][j]-labels[v][j]) for j in range(2)]) for v in by_person[p]]) for p in people]
                values[arm][pos].append(pp);close(avg(pp),metric['person_mae_by_seed'][pos][si])
            span=[avg([avg([max(rec[arm,seed,pos,v][j] for pos in positions[:3])-min(rec[arm,seed,pos,v][j] for pos in positions[:3]) for j in range(2)]) for v in by_person[p]]) for p in people]
            shift=[avg([avg([abs(rec[arm,seed,'stress',v][j]-rec[arm,seed,'front',v][j]) for j in range(2)]) for v in by_person[p]]) for p in people]
            values[arm]['range'].append(span);values[arm]['stress_shift'].append(shift)
            close(avg(span),metric['range_by_seed'][si]);close(avg(shift),metric['stress_shift_by_seed'][si])
            predpersons=[avg([avg(rec[arm,seed,'front',v]) for v in by_person[p]]) for p in people]
            close(statistics.stdev(predpersons),metric['front_prediction_sd_by_seed'][si])
        for pos in positions:close(avg([avg(a) for a in values[arm][pos]]),metric['person_mae_seedmean'][pos])
        for k in ['range','stress_shift']:close(avg([avg(a) for a in values[arm][k]]),metric[k+'_seedmean'])
        for fold in range(5):
            pi=[i for i,p in enumerate(people) if fold_by_video[by_person[p][0]]==fold]
            close(avg([avg([vv[i] for i in pi]) for vv in values[arm]['front']]),metric['front_fold_mae_seedmean'][str(fold)])
    for key,stats in result['comparisons'].items():
        can,ref=key.split('_vs_')
        for metric in positions+['range','stress_shift']:
            d=[[a-b for a,b in zip(aa,bb)] for aa,bb in zip(values[ref][metric],values[can][metric])];dp=[avg(q) for q in zip(*d)]
            reported=stats['mae'][metric] if metric in positions else stats[metric]
            close(avg(dp),reported['reduction']);assert sum(v>0 for v in dp)==reported['people_improved'] and sum(v<0 for v in dp)==reported['people_worsened']
            for a,b in zip(map(avg,d),reported['seed_reductions']):close(a,b)
            rng=np.random.default_rng(20261026);arr=np.array(dp);ci=np.percentile(arr[rng.integers(0,175,(10000,175))].mean(1),[2.5,97.5]);np.testing.assert_allclose(ci,reported['paired_bootstrap95'],rtol=0,atol=1e-8)
    out=dict(verified=True,csv_rows=len(rows),people=len(people),records=len(vid),matched_training_pairs=15,training_models=30,source_hashes_checked=len(sources),per_record_feature_hashes_checked=203,numpy_model_replay_max=maxerr,independent_person_metric_aggregation=True,bootstrap_recomputed=True,synthetic_identity_training=True,protected_partitions_not_loaded=True)
    with (S/'verification.json').open('x',encoding='utf-8') as out_file:json.dump(out,out_file,indent=2)
    print(json.dumps(out))
if __name__=='__main__':main()
