from common import *
import joblib

def main():
    audit=json.loads((HERE/'preparation.json').read_text())
    assert sha(HERE/'protocol.json')==audit['protocol_sha256']
    assert sha(HERE/'common.py')==audit['common_sha256']
    for path,digest in audit['sources'].items():assert sha(path)==digest,path
    labels=pd.read_csv(prior.LABELS).set_index(['subid','visit']);assert labels.index.is_unique
    labelhash=json.loads((PRIOR/'preparation.json').read_text())['source_hashes'][str(prior.LABELS)]
    assert sha(prior.LABELS)==labelhash
    data={};plan={};checks=0
    for name,meta in audit['cohorts'].items():
        d=load_data(name);data[name]=d;old=dict(np.load(PRIOR/name/'input.npz'))
        assert sha(HERE/name/'input.npz')==meta['input_sha256']
        assert sha(HERE/name/'splits.json')==meta['splits_sha256']
        for key in ['ids','y','x','speed']:np.testing.assert_array_equal(d[key],old[key])
        assert len(d['ids'])==len(set(d['ids']))==meta['people'] and 'sub-283' not in d['ids']
        assert d['x'].shape[1:]==(19,181)
        man=pd.read_csv(HERE/name/'manifest.csv');assert list(d['ids'])==man.id.tolist()
        for i,r in enumerate(man.itertuples()):
            col='tft_5xsts' if name.startswith('STS') else 'tft_tug'
            assert labels.loc[(r.id,r.visit),col]==d['y'][i]
            o,rob,q,s,_=extract(d['x'][i]);np.testing.assert_allclose(o,old['pattern'][i,3:],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(d['quality'][i],np.r_[d['speed'][i],q],rtol=0,atol=0)
            np.testing.assert_allclose(d['pattern'][i],np.r_[d['quality'][i],o],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_allclose(d['robust'][i],np.r_[d['quality'][i],rob],rtol=0,atol=0,equal_nan=True)
            assert d['stable'][i]==s;checks+=1
        ss=json.loads((HERE/name/'splits.json').read_text());plan[name]=ss
        for rep in ss:
            tested=[]
            for split in rep['splits']:
                tr,te=set(split['train']),set(split['test']);assert tr.isdisjoint(te) and tr|te==set(range(len(d['ids'])))
                tested+=split['test'];validated=[]
                for inn in split['inner']:
                    a,b=set(inn['train']),set(inn['validation']);assert a.isdisjoint(b) and a|b==tr and not (a|b)&te
                    validated+=inn['validation']
                assert sorted(validated)==sorted(tr)
            assert sorted(tested)==list(range(len(d['ids'])))
    # Same person's windows are identical across outcome tasks.
    a,b=data['STS_common6'],data['TUG_common6'];lut={pid:i for i,pid in enumerate(b['ids'])}
    for i,pid in enumerate(a['ids']):np.testing.assert_array_equal(a['x'][i],b['x'][lut[pid]])
    t=np.arange(181)/30;x=np.zeros((19,181));x[5]=30+20*np.cos(2*np.pi*t/1.2);x[11]=30+20*np.cos(2*np.pi*t/1.2+np.pi)
    for w,p in VARIANTS:
        v,_=patterns(x,w,p);assert abs(v[0]-1.2)<1e-10
    o,r,q,s,_=extract(np.zeros((19,181)));assert not s and np.isnan(o[0]) and np.isnan(r[0]) and np.isfinite(q).all()
    damaged=x.copy();damaged[5,90]+=100;assert extract(damaged)[4]['bad_fraction']>0
    result={'source_hashes_checked':len(audit['sources'])+1,'windows_checked':checks,'original_features_exactly_reproduced':True,
            'people_and_splits_unchanged':True,'shared_quality_columns_identical':True,'synthetic_checks':True,'passed':True}
    save('input-verification.json',result)
    if '--inputs-only' in sys.argv:print(json.dumps(result),flush=True);return
    count=0;inner=0;maxdiff=0.;baseline_diff=0.;fit_diagnostics={}
    for name in data:
        d=data[name];pred=pd.read_csv(HERE/name/'predictions.csv');selections=json.loads((HERE/name/'selections.json').read_text())
        completion=json.loads((HERE/name/'completion.json').read_text())
        assert len(selections)==completion['final_models']==400 and completion['inner_models']==2700
        assert sha(HERE/name/'predictions.csv')==completion['predictions_sha256']
        assert sha(HERE/'run.py')==completion['run_code_sha256']
        for s in selections:
            split=plan[name][s['repeat']]['splits'][s['fold']];tr=np.array(split['train']);te=np.array(split['test']);x=d[s['view']]
            scores=np.mean(abs(np.array(s['inner_predictions'])-d['y'][tr]),axis=1)
            np.testing.assert_allclose(scores,s['scores'],rtol=0,atol=1e-12)
            assert s['selected']==s['grid'][int(np.argmin(scores))];inner+=len(scores)*3
            path=HERE/s['model_path'];assert sha(path)==s['model_sha256'];m=joblib.load(path)
            expected=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x[tr].T])
            np.testing.assert_allclose(m[0].statistics_,expected,rtol=0,atol=1e-12)
            transformed=m[0].transform(x[tr]).astype(np.float64)
            np.testing.assert_allclose(m[1].mean_,transformed.mean(0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(m[1].var_,transformed.var(0),rtol=1e-12,atol=1e-12)
            p=pred[(pred.repeat==s['repeat'])&(pred.fold==s['fold'])&(pred.view==s['view'])&(pred.model==s['model'])&(pred.seed==s['seed'])].set_index('id')
            assert len(p)==len(te);pp=m.predict(x[te]);diff=np.max(abs(pp-p.loc[d['ids'][te],'prediction'].to_numpy()))
            assert diff<1e-8;maxdiff=max(maxdiff,float(diff));count+=1
            # Descriptive fit diagnostic only: never used for settings or as validation.
            train_prediction=m.predict(x[tr])
            for k,pv in zip(tr,train_prediction):
                key=(name,s['model'],s['view'],str(d['ids'][k]))
                rec=fit_diagnostics.setdefault(key,{'dataset':name,'model':s['model'],'view':s['view'],'id':str(d['ids'][k]),
                    'target':float(d['y'][k]),'stable':bool(d['stable'][k]),'count':0,'ae_sum':0.,'under_sum':0.})
                err=float(pv-d['y'][k]);rec['count']+=1;rec['ae_sum']+=abs(err);rec['under_sum']+=max(-err,0)
        assert len(pred)==len(d['ids'])*5*17
        for key,g in pred.groupby(['repeat','model','view','seed']):assert len(g)==len(d['ids']) and g.id.nunique()==len(d['ids'])
        old=pd.read_csv(PRIOR/name/'predictions.csv');keys=['repeat','fold','view','model','seed','id']
        a=pred[pred.view.isin(['speed','none'])].set_index(keys).sort_index()
        b=old[old.view.isin(['speed','none'])].set_index(keys).sort_index();assert a.index.equals(b.index)
        baseline_diff=max(baseline_diff,float(np.max(abs(a.prediction-b.prediction))));assert baseline_diff<1e-8
        print('REPLAY',name,count,flush=True)
    assert count==800 and inner==5400
    fitrows=[]
    for r in fit_diagnostics.values():
        assert r['count']==(60 if r['model']=='RF' else 20)
        r['training_mae']=r.pop('ae_sum')/r['count'];r['training_under']=r.pop('under_sum')/r['count'];fitrows.append(r)
    pd.DataFrame(fitrows).to_csv(HERE/'training-fit-diagnostic.csv',index=False)
    result.update(final_models_replayed=count,inner_fit_selections_checked=inner,max_prediction_delta=maxdiff,prior_speed_baseline_max_delta=baseline_diff)
    save('verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
