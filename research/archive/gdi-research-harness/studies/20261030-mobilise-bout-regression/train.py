"""Nested person CV. Draw 0 trains; draws 1/2 only assess sampling robustness."""
from common import *
import itertools,time,joblib,sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

GRIDS={'Ridge':[{'alpha':a} for a in [.1,10.,1000.]],'SVR':[{'C':c} for c in [1.,10.,100.]],'RF':[{'min_samples_leaf':n} for n in [5,20,50]]}
def make_model(name,params,target):
    if name=='Ridge':model=Ridge(**params)
    elif name=='SVR':model=SVR(kernel='rbf',gamma='scale',epsilon=.1 if target=='sppb' else .5,cache_size=256,**params)
    else:model=RandomForestRegressor(n_estimators=128,max_features=1.,random_state=20261030,n_jobs=4,**params)
    return Pipeline([('scale',StandardScaler()),('model',model)])
def fit(model,x,y,ids):
    w=weights(ids)
    model.fit(x,y,scale__sample_weight=w,model__sample_weight=w)
    return model
def run():
    started=time.time()
    data=pd.read_csv(HERE/'features.csv')
    protocol={'features':FEATURES,'duration_intervals_seconds':{'short':'[4,10)','long':'[30,60)'},'k':[1,3,5],'aggregation':'per-feature median of nested sampled bouts','eligibility':'at least 5 valid bouts of each band on the same day, valid target','sampling':'one uniformly hash-selected eligible day per person-visit, five hash-ranked bouts per band; independent of clinical scores','draws':{'0':'all fitting, tuning, and primary OOF evaluation','1':'additional held-out input draw; no retraining, no prediction averaging','2':'additional held-out input draw; no retraining, no prediction averaging'},'outer':'fixed cohort-stratified person 5-fold CV, all visits grouped','inner':'cohort-stratified person 3-fold CV within each outer training set','weights':'each person has equal total training weight; each person has equal metric weight','baseline':'outer-training person-weighted median label','target_columns':{'sppb':'total_sppb_score','tug':'tugtime'},'hyperparameter_grids':GRIDS,'fixed_parameters':{'RF':{'n_estimators':128,'max_features':1.,'random_state':20261030},'SVR':{'kernel':'rbf','gamma':'scale','epsilon':{'sppb':.1,'tug':.5}}},'clipping_or_rounding':False,'features_sha256':sha(HERE/'features.csv'),'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'interpretation':'exploratory sensor-feature prediction; no video or independent external validation'}
    save('protocol.json',protocol)
    (HERE/'models').mkdir(exist_ok=True);(HERE/'fold-results').mkdir(exist_ok=True)
    for target,band,k in itertools.product(['sppb','tug'],['short','long'],[1,3,5]):
        label=protocol['target_columns'][target]
        cell=data[data[target+'_ok']&data.band.eq(band)&data.k.eq(k)].sort_values(['draw']+KEY)
        primary=cell[cell.draw.eq(0)].reset_index(drop=True)
        for fold in range(5):
            tag=f'{target}-{band}-{k}-f{fold}'
            result_path=HERE/'fold-results'/f'{tag}.csv'
            if result_path.exists():
                print('Resume completed',tag,flush=True);continue
            train=primary[primary.fold.ne(fold)].reset_index(drop=True)
            test=cell[cell.fold.eq(fold)].copy()
            assert not set(train.participantid)&set(test.participantid)
            people=train[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
            inner=[]
            for a,b in StratifiedKFold(3,shuffle=True,random_state=20261100+fold).split(people,people.cohort):
                tr=train.participantid.isin(people.iloc[a].participantid).to_numpy()
                va=train.participantid.isin(people.iloc[b].participantid).to_numpy()
                inner.append((tr,va))
            out=test[KEY+['draw','fold','cohort',label]].rename(columns={label:'actual'}).copy()
            out['target']=target;out['band']=band;out['k']=k
            out['baseline']=weighted_median(train[label].to_numpy(),weights(train.participantid.to_numpy()))
            tuning={}
            for name,grid in GRIDS.items():
                trials=[]
                for params in grid:
                    pred=np.full(len(train),np.nan)
                    for tr,va in inner:
                        m=fit(make_model(name,params,target),train.loc[tr,FEATURES].to_numpy(),train.loc[tr,label].to_numpy(),train.loc[tr,'participantid'].to_numpy())
                        pred[va]=m.predict(train.loc[va,FEATURES].to_numpy())
                    assert np.isfinite(pred).all()
                    trials.append({'params':params,'inner_person_mae':person_mae(train[label].to_numpy(),pred,train.participantid.to_numpy())})
                best=min(trials,key=lambda x:x['inner_person_mae'])
                m=fit(make_model(name,best['params'],target),train[FEATURES].to_numpy(),train[label].to_numpy(),train.participantid.to_numpy())
                out[name]=m.predict(test[FEATURES].to_numpy())
                joblib.dump(m,HERE/'models'/f'{tag}-{name}.joblib',compress=3)
                tuning[name]={'selected':best,'trials':trials}
            save(f'fold-results/{tag}.json',{'target':target,'band':band,'k':k,'fold':fold,'train_people':int(train.participantid.nunique()),'test_people':int(test.participantid.nunique()),'train_visits':len(train),'tuning':tuning})
            out.to_csv(result_path,index=False)
            print(f'{tag} finished; elapsed {time.time()-started:.0f}s',flush=True)
    predictions=pd.concat([pd.read_csv(p) for p in sorted((HERE/'fold-results').glob('*.csv'))],ignore_index=True)
    predictions.to_csv(HERE/'predictions.csv',index=False)
    save('training-completion.json',{'fold_cells':60,'models':180,'elapsed_seconds':time.time()-started,'predictions_sha256':sha(HERE/'predictions.csv')})
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
