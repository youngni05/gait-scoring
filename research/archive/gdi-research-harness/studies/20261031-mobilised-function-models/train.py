"""Fixed, paired CPU comparison. Calibration people never fit base estimators."""
from common import *
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier

def inputs(g,view):
    if view=='pattern':return g[SPEED+PATTERN].to_numpy(float)
    x=g[SPEED].to_numpy(float)
    if view=='missingness_control':x=np.column_stack([x,g[PATTERN].isna().to_numpy(float)])
    return x

def estimator(name,seed):
    if name=='Ridge':model=Ridge(alpha=10)
    elif name=='Logistic':model=LogisticRegression(C=1,max_iter=2000,random_state=seed)
    elif name=='RFreg':model=RandomForestRegressor(n_estimators=200,min_samples_leaf=10,max_features=1.0,n_jobs=2,random_state=seed)
    else:model=RandomForestClassifier(n_estimators=200,min_samples_leaf=10,max_features=1.0,n_jobs=2,random_state=seed)
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),model)

def raw_score(model,name,x):
    if name in ['Ridge','RFreg']:return model.predict(x)
    p=np.clip(model.predict_proba(x)[:,1],1e-6,1-1e-6)
    return np.log(p/(1-p))

def main():
    started=time.time();parts=pd.read_csv(OUT/'person-folds.csv');preds=[];cals=[];ledger=[]
    save('execution.json',{'source_hashes':{str(p):sha(p) for p in [HERE/'common.py',HERE/'prepare.py',HERE/'train.py',OUT/'protocol.json',OUT/'STS-cohort.csv',OUT/'TUG-cohort.csv',OUT/'person-folds.csv']},'status':'running','device':'CPU','models_saved':False})
    for target in ['STS','TUG']:
        g=pd.read_csv(OUT/f'{target}-cohort.csv').sort_values('participantid').reset_index(drop=True)
        for rep in [0,1,2]:
            assignments=g.participantid.map(parts[parts['repeat']==rep].set_index('participantid').fold).to_numpy()
            assert np.isfinite(assignments).all()
            for fold in range(5):
                train=(assignments!=fold)&(assignments!=(fold+1)%5);cal=assignments==(fold+1)%5;test=assignments==fold
                seed=1000+int(parts[parts['repeat']==rep].seed.iloc[0])*31+fold
                assert train.sum()+cal.sum()+test.sum()==len(g)
                for view in ['speed','pattern','missingness_control']:
                    x=inputs(g,view)
                    for name in ['Ridge','Logistic','RFreg','RFclass']:
                        reg=name in ['Ridge','RFreg'];y=g.target.to_numpy() if reg else g.slow.to_numpy()
                        model=estimator(name,seed);model.fit(x[train],y[train])
                        rawcal=raw_score(model,name,x[cal]);rawtest=raw_score(model,name,x[test])
                        calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,max_iter=2000))
                        calibrator.fit(rawcal[:,None],g.slow.to_numpy()[cal])
                        cp=calibrator.predict_proba(rawcal[:,None])[:,1];tp=calibrator.predict_proba(rawtest[:,None])[:,1]
                        negative=cp[g.slow.to_numpy()[cal]==0]
                        threshold=float(np.nextafter(np.quantile(negative,.9,method='higher'),np.inf))
                        row={'outcome':target,'repeat':rep,'fold':fold,'view':view,'model':name,'seed':seed,'threshold':threshold,
                             'train_n':int(train.sum()),'calibration_n':int(cal.sum()),'test_n':int(test.sum()),'train_slow':int(g.loc[train,'slow'].sum()),'calibration_slow':int(g.loc[cal,'slow'].sum()),
                             'calibration_fpr':float((negative>=threshold).mean()),'calibration_coef':float(calibrator[-1].coef_[0,0]),'train_median':float(g.loc[train,'target'].median()),'train_prevalence':float(g.loc[train,'slow'].mean())}
                        ledger.append(row)
                        for mask,raw,p,role,destination in [(test,rawtest,tp,'test',preds),(cal,rawcal,cp,'calibration',cals)]:
                            frame=g.loc[mask,['participantid','visit.number','cohort','target','slow']].copy()
                            for k in ['outcome','repeat','fold','view','model','seed','threshold','train_median','train_prevalence']:frame[k]=row[k]
                            frame['raw_score']=raw;frame['prediction_seconds']=raw if reg else np.nan
                            frame['probability']=p;frame['decision']=p>=threshold;frame['decision_p50']=p>=.5
                            frame['decision_time20']=(raw>=20) if reg else False
                            frame['role']=role;destination.append(frame)
                pd.concat(preds,ignore_index=True).to_csv(OUT/'predictions.csv.gz',index=False)
                pd.concat(cals,ignore_index=True).to_csv(OUT/'calibration-predictions.csv.gz',index=False)
                pd.DataFrame(ledger).to_csv(OUT/'fit-ledger.csv',index=False)
                print(f'{target} repeat={rep} fold={fold} fits={len(ledger)} seconds={time.time()-started:.1f}',flush=True)
    execution=json.loads((OUT/'execution.json').read_text(encoding='utf8'))
    execution.update(status='complete',base_model_fits=len(ledger),seconds=time.time()-started,prediction_rows=sum(len(p) for p in preds))
    save('execution.json',execution)

if __name__=='__main__':main()
