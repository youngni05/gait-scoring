"""Rebuild inputs and metrics, audit person separation, and replay saved models."""
from common import *
import joblib,csv,math
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

def run():
    f=pd.read_csv(HERE/'features.csv');s=pd.read_csv(HERE/'selected-bouts.csv');p=pd.read_csv(HERE/'predictions.csv')
    protocol=json.loads((HERE/'protocol.json').read_text())
    prep=json.loads((HERE/'preparation.json').read_text())
    source=json.loads((HERE/'source-verification.json').read_text())
    assert source['passed']
    assert sha(HERE/'features.csv')==protocol['features_sha256']==prep['features_sha256']
    assert sha(HERE/'selected-bouts.csv')==prep['selected_bouts_sha256']
    for name,digest in prep['source_sha256'].items():assert sha(SOURCE/name)==digest
    assert len(p.groupby(['target','band','k','fold','draw']))==180
    assert not p.duplicated(['target','band','k','draw']+KEY).any()
    assert p.groupby('participantid').fold.nunique().eq(1).all()
    days=pd.read_csv(HERE/'chosen-days.csv')
    selected_day=s.merge(days[KEY+['draw','wbday']],on=KEY+['draw'],validate='many_to_one',suffixes=('','_expected'))
    assert selected_day.wbday.eq(selected_day.wbday_expected).all()
    assert s.groupby(KEY+['draw','band'])['rank'].apply(lambda x:sorted(x)==[1,2,3,4,5]).all()
    assert s[s.band.eq('short')].duration.between(4,10,inclusive='left').all()
    assert s[s.band.eq('long')].duration.between(30,60,inclusive='left').all()
    for k in [1,3,5]:
        agg=s[s['rank'].le(k)].groupby(KEY+['draw','band'])[FEATURES].median().sort_index()
        actual=f[f.k.eq(k)].set_index(KEY+['draw','band'])[FEATURES].sort_index()
        assert agg.index.equals(actual.index)
        np.testing.assert_allclose(agg,actual,rtol=1e-12,atol=1e-12)
    raw=pd.read_csv(SOURCE/'All_cohorts_dataset.csv',usecols=['participantid','visit.number','total_sppb_score','tugtime']).rename(columns={'visit.number':'visit_number'})
    for target,column in [('sppb','total_sppb_score'),('tug','tugtime')]:
        d=p[p.target.eq(target)].merge(raw[KEY+[column]],on=KEY,validate='many_to_one')
        np.testing.assert_allclose(d.actual,d[column],rtol=0,atol=1e-12)
        sets=[set(map(tuple,g[KEY].to_numpy())) for _,g in p[p.target.eq(target)].groupby(['band','k','draw'])]
        assert all(x==sets[0] for x in sets)
        n=2174 if target=='sppb' else 1939
        assert p[p.target.eq(target)].participantid.nunique()==n
    replayed=0;inner_splits=0;closed_form=0
    for (target,band,k,fold),d in p.groupby(['target','band','k','fold']):
        label=protocol['target_columns'][target]
        base=f[f[target+'_ok']&f.band.eq(band)&f.k.eq(k)&f.draw.eq(0)]
        train=base[base.fold.ne(fold)]
        assert not set(train.participantid)&set(d.participantid)
        person=train[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
        for a,b in StratifiedKFold(3,shuffle=True,random_state=20261100+int(fold)).split(person,person.cohort):
            assert not set(person.iloc[a].participantid)&set(person.iloc[b].participantid)
            inner_splits+=1
        # Independently calculate the weighted empirical median and scaler statistics.
        counts=train.groupby('participantid').participantid.transform('size').to_numpy()
        w=1/counts;w=w/w.mean()
        y=train[label].to_numpy();order=np.argsort(y);cumulative=0.;median=None
        for i in order:
            cumulative+=w[i]
            if cumulative>=w.sum()/2:median=float(y[i]);break
        np.testing.assert_allclose(d.baseline,median,atol=1e-12,rtol=0)
        test=d.merge(f[KEY+['draw','band','k']+FEATURES],on=KEY+['draw','band','k'],validate='one_to_one')
        ledger=json.loads((HERE/'fold-results'/f'{target}-{band}-{k}-f{fold}.json').read_text())
        for name in ['Ridge','SVR','RF']:
            chosen=ledger['tuning'][name]
            assert chosen['selected']['inner_person_mae']==min(x['inner_person_mae'] for x in chosen['trials'])
            model=joblib.load(HERE/'models'/f'{target}-{band}-{k}-f{fold}-{name}.joblib')
            assert model.n_features_in_==4
            np.testing.assert_allclose(model.named_steps['scale'].mean_,np.average(train[FEATURES],axis=0,weights=w),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(model.predict(test[FEATURES].to_numpy()),test[name],rtol=1e-10,atol=1e-10)
            replayed+=1
            if name=='Ridge' and fold==0:
                x=model.named_steps['scale'].transform(train[FEATURES].to_numpy())
                mean_x=np.average(x,axis=0,weights=w);mean_y=np.average(y,weights=w)
                xc=x-mean_x;yc=y-mean_y
                alpha=chosen['selected']['params']['alpha']
                coef=np.linalg.solve(xc.T@(w[:,None]*xc)+alpha*np.eye(4),xc.T@(w*yc))
                predicted=(model.named_steps['scale'].transform(test[FEATURES].to_numpy())-mean_x)@coef+mean_y
                np.testing.assert_allclose(predicted,test.Ridge,rtol=1e-9,atol=1e-9)
                closed_form+=1
    # Metrics are recomputed with stdlib loops, not the training metric function.
    errors=defaultdict(lambda:defaultdict(list))
    with (HERE/'predictions.csv').open(newline='',encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            for model in ['baseline','Ridge','SVR','RF']:
                key=(row['target'],row['band'],row['k'],row['draw'],model)
                errors[key][row['participantid']].append(abs(float(row['actual'])-float(row[model])))
    checked=0
    with (HERE/'metrics.csv').open(newline='',encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            key=(row['target'],row['band'],row['k'],row['draw'],row['model'])
            per=[math.fsum(x)/len(x) for x in errors[key].values()]
            value=math.fsum(per)/len(per)
            assert math.isclose(value,float(row['person_mae']),rel_tol=1e-12,abs_tol=1e-12)
            assert len(per)==int(row['people']);checked+=1
    assert checked==144 and replayed==180 and closed_form==12
    result={'passed':True,'models_replayed':replayed,'inner_person_splits_checked':inner_splits,'independent_closed_form_ridge_fits':closed_form,'independently_recomputed_metric_rows':checked,'same_people_visits_all_six_conditions_and_three_draws':True,'all_visits_of_each_person_have_same_fold':True,'scaler_statistics_match_outer_training_only':True,'sampled_bout_feature_medians_match':True,'source_clinical_labels_match':True,'source_hashes_match':True,'selected_bouts_independently_checked_against_original':source['unique_selected_bouts_verified'],'artifacts_sha256':{name:sha(HERE/name) for name in ['protocol.json','predictions.csv','metrics.csv','paired-comparisons.json','subgroup-metrics.csv','summary.json']},'scripts_sha256':{x.name:sha(x) for x in HERE.glob('*.py')}}
    save('verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
