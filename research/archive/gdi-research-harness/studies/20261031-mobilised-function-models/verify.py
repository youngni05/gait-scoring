"""Independent ledgers, metric arithmetic, and deterministic fit replay."""
from common import *
from train import inputs,estimator,raw_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    ex=json.loads((OUT/'execution.json').read_text(encoding='utf8'));assert ex['status']=='complete' and ex['base_model_fits']==360
    for path,digest in ex['source_hashes'].items():assert sha(path)==digest,path
    p=pd.read_csv(OUT/'predictions.csv.gz',float_precision='round_trip')
    c=pd.read_csv(OUT/'calibration-predictions.csv.gz',float_precision='round_trip')
    led=pd.read_csv(OUT/'fit-ledger.csv',float_precision='round_trip');parts=pd.read_csv(OUT/'person-folds.csv')
    metric=pd.read_csv(OUT/'metrics-by-repeat.csv');checks=0;replayed=0
    assert len(led)==360 and len(p)==ex['prediction_rows']
    assert p.probability.between(0,1).all() and np.isfinite(p.raw_score).all()
    datasets={t:pd.read_csv(OUT/f'{t}-cohort.csv') for t in ['STS','TUG']}
    for target,data in datasets.items():
        assert data.participantid.nunique()==len(data)
        assert (data.slow==(data.target>=20)).all()
        if target=='STS':assert data.sts_status.eq('valid_standard_time').all()
        form,label=('sppb','5crtresult') if target=='STS' else ('tug','tugtime')
        original=pd.read_csv(SRC/f'extracted/Clinical Data/T1-T5-study-instances-{form}.csv').rename(columns={'visit-number':'visit.number'})
        linked=data.merge(original[['participantid','visit.number',label]],on=['participantid','visit.number'],validate='one_to_one')
        assert len(linked)==len(data)
        np.testing.assert_allclose(linked.target,linked[label])
    for key,g in p.groupby(['outcome','repeat','fold','view','model']):
        outcome,rep,fold,view,name=key
        data=datasets[outcome].sort_values('participantid').reset_index(drop=True)
        mapping=parts[parts['repeat']==rep].set_index('participantid').fold
        a=data.participantid.map(mapping);test=set(data.loc[a==fold,'participantid']);cal=set(data.loc[a==(fold+1)%5,'participantid']);train=set(data.participantid)-test-cal
        assert test.isdisjoint(cal) and test.isdisjoint(train) and cal.isdisjoint(train)
        assert set(g.participantid)==test and len(g)==len(test)
        cg=c[(c.outcome==outcome)&(c['repeat']==rep)&(c.fold==fold)&(c.view==view)&(c.model==name)]
        assert set(cg.participantid)==cal and len(cg)==len(cal)
        labels=data.set_index('participantid').target
        np.testing.assert_allclose(g.target,g.participantid.map(labels));np.testing.assert_allclose(cg.target,cg.participantid.map(labels))
        negatives=cg.loc[cg.slow==0,'probability'].to_numpy()
        threshold=np.nextafter(np.quantile(negatives,.9,method='higher'),np.inf)
        assert np.allclose(g.threshold,threshold,rtol=0,atol=1e-14)
        assert ((g.probability>=g.threshold)==g.decision).all()
        assert (negatives>=threshold).mean()<=.100001
        assert np.allclose(g.train_prevalence,data[data.participantid.isin(train)].slow.mean())
        checks+=1
        replay=(rep==0 and fold==0 and ((outcome=='STS' and ((view=='speed' and name=='RFreg') or (view=='pattern' and name=='RFclass'))) or (outcome=='TUG' and view=='pattern' and name in ['Ridge','Logistic'])))
        if replay:
            tr=data.participantid.isin(train).to_numpy();ca=data.participantid.isin(cal).to_numpy();te=data.participantid.isin(test).to_numpy()
            x=inputs(data,view);reg=name in ['Ridge','RFreg'];y=data.target if reg else data.slow
            model=estimator(name,int(g.seed.iloc[0]));model.fit(x[tr],y.to_numpy()[tr])
            sc=raw_score(model,name,x[ca]);st=raw_score(model,name,x[te])
            calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1e6,max_iter=2000));calibrator.fit(sc[:,None],data.slow.to_numpy()[ca])
            probability=calibrator.predict_proba(st[:,None])[:,1]
            gs=g.sort_values('participantid')
            np.testing.assert_allclose(st,gs.raw_score,rtol=0,atol=1e-9)
            np.testing.assert_allclose(probability,gs.probability,rtol=0,atol=1e-9);replayed+=1
    # Arithmetic from person-level predictions, independent of analysis.metrics.
    errors=[]
    for key,g in p.groupby(['outcome','model','view','repeat']):
        outcome,name,view,rep=key
        assert len(g)==g.participantid.nunique()==len(datasets[outcome])
        m=metric[(metric.outcome==outcome)&(metric.model==name)&(metric.view==view)&(metric['repeat']==rep)&(metric.cohort=='all')].iloc[0]
        y=g.slow.to_numpy(bool);pr=g.probability.to_numpy();d=g.decision.to_numpy(bool)
        expected={'brier':float(((pr-y)**2).mean()),'fn':int((y&~d).sum()),'fp':int((~y&d).sum()),'miss_rate':float((~d[y]).mean()),'fpr':float(d[~y].mean()),'mean_probability':float(pr.mean()),'actual_slow_rate':float(y.mean()),'calibration_bias':float(pr.mean()-y.mean())}
        pos=pr[y];neg=pr[~y];expected['auroc']=float(((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])).mean())
        if name in ['Ridge','RFreg']:expected['mae']=float(abs(g.prediction_seconds-g.target).mean())
        for field,value in expected.items():errors.append(abs(float(m[field])-value));assert abs(float(m[field])-value)<1e-12
    save('verification.json',{'passed':True,'base_fits':360,'disjoint_fold_cells_checked':checks,'deterministically_retrained_cells':replayed,'metric_max_difference':max(errors),'predictions_sha256':sha(OUT/'predictions.csv.gz'),'original_clinical_form_linkage_checked':True,'probability_bounds_checked':True,'source_hashes_unchanged':True,'no_evaluation_labels_used_in_calibration_or_threshold':True,'limits':'Source-label truth not independently adjudicated from recordings. Weekly DMO only. Bootstrap is conditional; no new-site or single-camera validation.'})
    print((OUT/'verification.json').read_text(encoding='utf8'))

if __name__=='__main__':main()
