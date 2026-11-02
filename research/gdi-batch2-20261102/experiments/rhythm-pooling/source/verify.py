import csv,json,statistics,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,t=1e-9):assert abs(a-b)<=t,(a,b)
def main():
    a=load(S/'result.json');rows=list(csv.DictReader((S/'predictions.csv').open(encoding='utf-8')))
    assert len(rows)==3248 and len({r['video_id'] for r in rows})==203 and len({r['patient_id'] for r in rows})==175
    assert len({(r['method'],r['position'],r['video_id']) for r in rows})==len(rows)
    pids=sorted({r['patient_id'] for r in rows});errs=defaultdict(list);videos=defaultdict(dict);personfold={}
    for r in rows:
        p=[float(r['prediction_L']),float(r['prediction_R'])];y=[float(r['actual_L']),float(r['actual_R'])]
        errs[(r['method'],r['position'],r['patient_id'])].append(sum(abs(u-v) for u,v in zip(p,y))/2)
        videos[(r['method'],r['video_id'])][r['position']]=(r['patient_id'],p)
        assert r['patient_id'] not in personfold or personfold[r['patient_id']]==r['fold'];personfold[r['patient_id']]=r['fold']
    range_values={}
    for m,metrics in a['metrics'].items():
        for pos,value in metrics['person_mae'].items():close(statistics.mean(statistics.mean(errs[(m,pos,p)]) for p in pids),value)
        for fold,values in metrics['fold_mae'].items():
            ids=[p for p in pids if personfold[p]==fold];assert len(ids)==35
            for pos,value in values.items():close(statistics.mean(statistics.mean(errs[(m,pos,p)]) for p in ids),value)
        ranges=defaultdict(list);means=defaultdict(list)
        for (method,v),pp in videos.items():
            if method!=m:continue
            p=pp['front'][0];seq=[pp[k][1] for k in ('front','middle','last')]
            ranges[p].append(sum(max(q[j] for q in seq)-min(q[j] for q in seq) for j in (0,1))/2)
            means[p].append(statistics.mean(pp['mean3'][1]))
            for j in (0,1):close(sum(q[j] for q in seq)/3,pp['mean3'][1][j])
        range_values[m]=[statistics.mean(ranges[p]) for p in pids]
        close(statistics.mean(range_values[m]),metrics['range_mean']);close(statistics.median(range_values[m]),metrics['range_median'])
        close(statistics.stdev(statistics.mean(means[p]) for p in pids),metrics['prediction_sd_mean3'])
    for name,c in a['comparisons'].items():
        can,ref=name.split('_vs_')
        for pos,value in c['mae'].items():
            d=[statistics.mean(errs[(ref,pos,p)])-statistics.mean(errs[(can,pos,p)]) for p in pids]
            close(statistics.mean(d),value['reduction']);assert sum(q>0 for q in d)==value['people_improved']
        close(statistics.mean(x-y for x,y in zip(range_values[ref],range_values[can])),c['range']['reduction'])
    for rel,d in load(S/'sources.json')['sha256'].items():assert sha(R/rel)==d
    for rel,d in load(S/'analysis_sources.json').items():assert sha(R/rel)==d
    with np.load(S/'features.npz',allow_pickle=False) as f,np.load(S/'fold_models.npz',allow_pickle=False) as stored:
        mod={k:stored[k] for k in stored.files}
        vids={str(v):i for i,v in enumerate(f['video_ids'])};diag=load(S/'diagnostics.json');no_missing=0
        for i in range(203):
            record=S/f'records/row_{i:04d}.npz';proof=load(record.with_suffix('.json'));assert sha(record)==proof['sha256']
            with np.load(record,allow_pickle=False) as rr:
                for pos in ('front','middle','last'):
                    for key in ('valid_pool','rhythm'):np.testing.assert_array_equal(rr[pos+'_'+key],f[pos+'_'+key][i])
                    if diag[i][pos]['missing_cells']==0:
                        np.testing.assert_allclose(f[pos+'_valid_pool'][i],f[pos+'_baseline'][i],rtol=0,atol=1e-6);no_missing+=1
        for r in rows:
            if r['position']=='mean3':continue
            i=vids[r['video_id']];m=r['method'];pos=r['position'];prefix=m+'_fold'+r['fold']+'__'
            x=np.r_[f[pos+'_baseline'][i],f[pos+'_rhythm'][i]] if m=='combined' else f[pos+'_'+m][i]
            for j,side in enumerate(('L','R')):
                p=float(mod[prefix+'intercept'][j])+sum((float(v)-float(mod[prefix+'scaler_mean'][k]))/float(mod[prefix+'scaler_scale'][k])*float(mod[prefix+'coef'][j,k]) for k,v in enumerate(x))
                close(p,float(r['prediction_'+side]),1e-8)
    v=dict(verified=True,people=175,records=203,prediction_rows=3248,feature_windows=609,no_missing_pool_controls=no_missing,person_metrics_and_fold_metrics_independently_recomputed=True,scalar_saved_model_replay=True,source_hashes_unchanged=True,bootstrap_independently_recomputed=False,raw_feature_formula_checks='synthetic invariances and known signals, not clinical ground truth')
    with (S/'verification.json').open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
    print(json.dumps(v))
if __name__=='__main__':main()

