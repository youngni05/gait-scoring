from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[2]/'outputs/gdi-ridge-experiment-20261016/deps'))
import pandas as pd
import numpy as np
d=pd.read_csv(HERE/'comparison-predictions.csv',dtype={'condition':str});overlap=json.loads((HERE/'overlap-summary.json').read_text());old=set(overlap['file_fingerprint_matches'].values())|set(overlap['clinical4times_height_weight_matches'].values());rng=np.random.default_rng(20261030)
def diff_ci(values):
    means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
    return {'mean_seconds':float(values.mean()),'conditional_bootstrap95':np.quantile(means,[.025,.975]).tolist()}
results={'scope':'postfit descriptive paired person bootstrap; conditional on storedOOF predictions, excludes retraining/modelselection variability','duration_comparisons':{},'baseline_comparisons':{},'subgroups':{}}
models=[x for x in ['Ridge','RF','CNN','TCN'] if x in d]
for model in models:
    a=d[d.condition=='3'].set_index('id');b=d[d.condition=='6'].set_index('id').loc[a.index]
    results['duration_comparisons'][model+'_3minus6']=diff_ci((abs(a[model]-a.target)-abs(b[model]-b.target)).to_numpy())
for cond,g in d.groupby('condition'):
    results['baseline_comparisons'][cond]={m:diff_ci((abs(g[m]-g.target)-abs(g.Median-g.target)).to_numpy()) for m in models}
    results['subgroups'][cond]={}
    for name,mask in [('known_or_probable_old',g.id.isin(old)),('no_overlap_identified',~g.id.isin(old))]:
        z=g[mask];results['subgroups'][cond][name]={'people':len(z),**{m:float(abs(z[m]-z.target).mean()) for m in ['Median']+models}}
(HERE/'uncertainty-and-overlap.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results['duration_comparisons'],indent=2))
