"""Walking-only published video features -> clinician TUG, person-held-out pilot."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
HERE=Path(__file__).resolve().parent
SRC=HERE.parent/'20261030-older44-sppb-timeseries/candidate-audit'
def save(n,d): (HERE/n).write_text(json.dumps(d,indent=2),encoding='utf8')
expected={'participant':'e38c5f802c74650326b93eee828a6e53','features':'310587c52c605b5d986e9f5d2a41ed9c','key':'58ac2b67f6a44d0d4bbec99fbae1585b'}
for n,h in expected.items(): assert hashlib.md5((SRC/f'opencap129-{n}.csv').read_bytes()).hexdigest()==h,n
feat=pd.read_csv(SRC/'opencap129-features.csv'); labels=pd.read_csv(SRC/'opencap129-participant.csv')
cols=[c for c in feat if c.startswith('10mwt_')]; assert len(cols)==8
merged=feat.merge(labels[['ID','day','tft_tug']],on=['ID','day'],validate='one_to_one')
eligible=merged.replace([np.inf,-np.inf],np.nan).dropna(subset=cols+['tft_tug'])
eligible=eligible[eligible.tft_tug>0].sort_values(['ID','day']).drop_duplicates('ID',keep='first').reset_index(drop=True)
X=eligible[cols].to_numpy(); y=eligible.tft_tug.to_numpy(); ids=eligible.ID.to_numpy(); assert len(set(ids))==len(ids)>12
save('protocol.json',{'source':'https://zenodo.org/records/13788592','source_md5':expected,'features':cols,'target':'clinician tft_tug seconds','visits':'first eligible visit per person, numeric day order, complete8walkingfeatures and positive finiteTUG','outer':'person KFold5 shuffle seed20261030','inner':'training-person KFold3 shuffle seed20261031','Ridge_alphas':[1,100],'RF_leaf':[2,5],'RF_trees':128,'seed':20261030,'scope':'published RGB-derived features from recorded part of10mwalk task; not3second input, rawRGB reinference, elderly-only, or independent external validation','merged_visits':len(merged),'eligible_people':len(ids),'excluded_people':sorted(set(labels.ID)-set(ids))})
eligible[['ID','day','tft_tug']+cols].to_csv(HERE/'inputs.csv',index=False)
def model(name,value):
    return make_pipeline(StandardScaler(),Ridge(alpha=value)) if name=='Ridge' else RandomForestRegressor(n_estimators=128,min_samples_leaf=value,random_state=20261030,n_jobs=2)
pred={m:np.full(len(y),np.nan) for m in ['Mean','Median','Ridge','RF']}; folds=[]
for f,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=20261030).split(ids)):
    inner=list(KFold(3,shuffle=True,random_state=20261031).split(tr)); entry={'fold':f,'train':ids[tr].tolist(),'test':ids[te].tolist(),'inner':[{'train':ids[tr[a]].tolist(),'test':ids[tr[b]].tolist()} for a,b in inner],'selection':{}}
    assert not set(ids[tr])&set(ids[te])
    pred['Mean'][te]=np.mean(y[tr]);pred['Median'][te]=np.median(y[tr])
    for name,values in [('Ridge',[1,100]),('RF',[2,5])]:
        scores=[]
        for value in values:
            errs=[]
            for a,b in inner:
                m=model(name,value).fit(X[tr[a]],y[tr[a]])
                errs.extend(abs(m.predict(X[tr[b]])-y[tr[b]]))
            scores.append(float(np.mean(errs)))
        best=values[int(np.argmin(scores))];m=model(name,best).fit(X[tr],y[tr]);pred[name][te]=m.predict(X[te]);joblib.dump(m,HERE/f'{name}-{f}.joblib');entry['selection'][name]={'value':best,'inner_MAE':scores}
    folds.append(entry)
out=eligible[['ID','day','tft_tug']].copy()
for m,v in pred.items(): assert np.isfinite(v).all();out[m]=v
out.to_csv(HERE/'predictions.csv',index=False);save('splits.json',folds)
summary={'people':len(ids),'TUG_range_seconds':[float(y.min()),float(y.max())],'metrics':{m:{'MAE_seconds':float(np.mean(abs(v-y))),'RMSE_seconds':float(np.sqrt(np.mean((v-y)**2)))} for m,v in pred.items()}}
save('summary.json',summary);print(json.dumps(summary,indent=2))
