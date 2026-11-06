"""Post-hoc, added after seeing results: same-age Stanford high scorers vs unseen normal sets.
Also recomputes the primary group means by a separate code path."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'tmp/ljmu-analysis-python-deps'))
import numpy as np
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105');SRC=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
R=json.loads((HERE/'results.json').read_text(encoding='utf-8'));meta=json.loads((HERE/'preparation.json').read_text(encoding='utf-8'))
ens=json.loads((ROOT/'studies/20261104-ljmu100-mixed/bundle/ensemble.json').read_text(encoding='utf-8-sig'))
C=np.load(BIG/'pred-CNN.npz');S=np.load(BIG/'pred-STT.npz')
def simple(n,arm):
    # Left-right mean of fusion equals the weighted CNN/STT bag mean; the bilateral term only moves the difference.
    return np.mean([ens[f]['cnn_weight']*C[f'{n}__{arm}__{f}__bag'].mean(1)+(1-ens[f]['cnn_weight'])*S[f'{n}__{arm}__{f}__bag'].mean(1) for f in range(5)],axis=0)
lp=np.array(meta['live_people']);nums=np.array(meta['self_numbers']);norm=np.isin(nums,meta['self_normal']);abn=np.isin(nums,meta['self_abnormal'])
check={}
for arm in ['baseline','mixed']:
    l=simple('live',arm);v=simple('self',arm)
    check[arm]=dict(live=float(np.mean([l[lp==p].mean() for p in sorted(set(lp.tolist()))])),self_normal=float(v[norm].mean()),self_abnormal=float(v[abn].mean()))
    assert abs(check[arm]['live']-R['live'][arm]['person_mean'])<1e-9
    assert abs(check[arm]['self_normal']-R['self'][arm]['normal_mean'])<1e-9
    assert abs(check[arm]['self_abnormal']-R['self'][arm]['abnormal_mean'])<1e-9
Z=np.load(SRC/'comparison-predictions.npz');d=np.load(SRC/'data.npz',allow_pickle=False)
s=Z['source']=='Stanford';Y=Z['y'][s].mean(1);ids=Z['person_id'][s]
m=np.load(ROOT/'gdi-research-harness/studies/20261031-ljmu-normal-cnn/diagnostics/development_age_metadata.npz',allow_pickle=False)
age_by={(a,b):c for a,b,c in zip(m['person_id'],m['video_id'],m['age'])}
ages=np.array([age_by[(str(p),v.split(':')[1])] for p,v in zip(d['source_person_id'][s],d['video_id'][s])])
live_age=np.array(meta['live_ages'])
def summary(mask,label):
    u=np.array(sorted(set(ids[mask].tolist())));out=dict(label=label,records=int(mask.sum()),people=len(u),true_mean=float(np.mean([Y[mask&(ids==p)].mean() for p in u])))
    for arm in ['baseline','mixed']:
        p=Z[arm+'__fusion'][s].mean(1);v=np.array([p[mask&(ids==q)].mean() for q in u]);r=np.random.default_rng(20261105)
        b=v[r.integers(0,len(v),(10000,len(v)))].mean(1);out[arm]=dict(mean=float(v.mean()),ci95=[float(np.quantile(b,.025)),float(np.quantile(b,.975))])
    return out
a=(ages>=live_age.min())&(ages<live_age.max()+1)
rows=[summary(a&(Y>=90),'Stanford age matched to LIVE, record true GDI >= 90'),
      summary(a&(Y>=95),'Stanford age matched to LIVE, record true GDI >= 95'),
      summary((ages>=18)&(Y>=90),'Stanford adults, record true GDI >= 90')]
out=dict(status='post_hoc_after_results',independent_primary_recalculation=check,live_age_range=[float(live_age.min()),float(live_age.max())],stanford_groups=rows)
(HERE/'posthoc-gap.json').write_text(json.dumps(out,indent=1,ensure_ascii=False),encoding='utf-8')
print(json.dumps(out,indent=1,ensure_ascii=False))
