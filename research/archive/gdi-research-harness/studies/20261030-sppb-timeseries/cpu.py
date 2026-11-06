"""Same person folds for baselines, frame-count control, Ridge and RF."""
from common import *
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib
from threadpoolctl import threadpool_limits

def fit_predict(name, setting, x, y, ids, test):
    scale = StandardScaler().fit(x, sample_weight=weights(ids))
    if name in ['Ridge','FramesRidge']:
        model = Ridge(alpha=setting)
    else:
        model = RandomForestRegressor(n_estimators=128,min_samples_leaf=setting,random_state=20261030,n_jobs=2,max_features=1.)
    w = weights(ids); w /= w.mean()
    model.fit(scale.transform(x), y, sample_weight=w)
    return model.predict(scale.transform(test)), (scale,model)

def run():
    d = load(); x,y,ids,folds = (d[k] for k in ['x','y','ids','folds'])
    f = features(x,d['frames'])
    pred = pd.DataFrame({'sample':range(len(y)), 'id':ids, 'fold':folds,'target':y})
    selections=[]; splits=[]
    for fold in range(5):
        tr = np.flatnonzero(folds!=fold); te = np.flatnonzero(folds==fold)
        people = np.unique(ids[tr]); inner=[]
        for a,b in KFold(3,shuffle=True,random_state=20261031).split(people):
            it = tr[np.isin(ids[tr],people[a])]; iv=tr[np.isin(ids[tr],people[b])]
            assert not(set(ids[it]) & set(ids[iv])) and not(set(ids[it]) & set(ids[te]))
            inner.append({'train':it.tolist(),'validation':iv.tolist()})
        splits.append({'fold':fold,'train':tr.tolist(),'test':te.tolist(),'inner':inner})
        pred.loc[te,'Mean'] = np.average(y[tr],weights=weights(ids[tr]))
        pred.loc[te,'Median'] = median(y[tr],ids[tr])
        for name,grid in [('FramesRidge',[1.,100.]),('Ridge',[1.,100.]),('RF',[2,5])]:
            z = np.log1p(d['frames'])[:,None] if name=='FramesRidge' else f
            scores=[]
            for setting in grid:
                inner_pred=np.full(len(y),np.nan)
                for item in inner:
                    it=np.array(item['train']);iv=np.array(item['validation'])
                    inner_pred[iv]=fit_predict(name,setting,z[it],y[it],ids[it],z[iv])[0]
                scores.append(mae(y[tr],inner_pred[tr],ids[tr]))
            best=grid[int(np.argmin(scores))]
            pp,model=fit_predict(name,best,z[tr],y[tr],ids[tr],z[te])
            pred.loc[te,name]=pp
            out=HERE/'cpu-models';out.mkdir(exist_ok=True)
            joblib.dump(model,out/f'{name}-fold{fold}.joblib')
            selections.append({'fold':fold,'model':name,'setting':best,'grid':grid,'inner_person_MAE':scores})
            print('CPU fold',fold,name,'selected',best,flush=True)
    save('splits.json',splits)
    pred.to_csv(HERE/'cpu-predictions.csv',index=False)
    save('cpu-selection.json',selections)
    save('cpu-summary.json',{k:metrics(y,pred[k].to_numpy(),ids) for k in ['Mean','Median','FramesRidge','Ridge','RF']})
    print((HERE/'cpu-summary.json').read_text())

if __name__=='__main__':
    with threadpool_limits(limits=2):
        run()
