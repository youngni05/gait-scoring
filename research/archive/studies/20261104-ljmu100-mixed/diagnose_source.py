"""Can pose features distinguish sources? Diagnostic classifier, not a GDI model."""
import json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files}
raw=d['x'];w=np.concatenate([raw.mean(axis=2),raw.std(axis=2),np.ptp(raw,axis=2),np.abs(np.diff(raw,axis=2)).mean(axis=2)],axis=2).mean(axis=1)
x=np.zeros((len(d['y']),48));np.add.at(x,d['window_record'],w);x/=np.bincount(d['window_record'])[:,None]
with np.load(ROOT/'gdi-research-harness/studies/20261031-ljmu-normal-cnn/diagnostics/development_age_metadata.npz') as z:
    age=np.full(len(x),np.nan);age[d['source']=='Stanford']=z['age']
age[d['source']=='LJMU']=[r['age'] for r in json.loads((HERE/'ljmu-membership.json').read_text())]
y=(d['source']=='LJMU').astype(int);ids=d['person_id'];fold=d['fold'];out=[]
for name,use in [('all_sources',np.ones(len(x),bool)),('adults_only',age>=18)]:
    probabilities=np.full(len(x),np.nan);scores=[]
    for f in range(5):
        tr=use&(fold!=f);te=use&(fold==f)
        assert not set(ids[tr])&set(ids[te])
        # Fit with equal person weights, then balance the two sources.
        weights=np.array([1/np.sum(ids[tr]==p) for p in ids[tr]])
        for source in [0,1]:weights[y[tr]==source]/=weights[y[tr]==source].sum()
        weights*=tr.sum()/2
        m=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=20261105))
        m.fit(x[tr],y[tr],logisticregression__sample_weight=weights)
        probabilities[te]=m.predict_proba(x[te])[:,1]
    people=np.unique(ids[use]);p=np.array([probabilities[use&(ids==q)].mean() for q in people]);truth=np.array([y[ids==q][0] for q in people])
    assert np.isfinite(p).all()
    out.append(dict(group=name,people=len(people),Stanford_people=int((truth==0).sum()),LJMU_people=int(truth.sum()),
        person_auc=float(roc_auc_score(truth,p)),person_balanced_accuracy=float(balanced_accuracy_score(truth,p>=.5))))
result=dict(task='Predict dataset source from 48 pooled motion features. No age, GDI, person ID or image background used as inputs.',
    design='Five fixed person-disjoint folds; train-only standardization; fixed regularized logistic regression; equal person and source fit weights.',
    limitation='Dataset discrimination does not prove the GDI model uses a shortcut or identify age/camera/disease effects separately.',results=out)
(HERE/'source-diagnostics.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
