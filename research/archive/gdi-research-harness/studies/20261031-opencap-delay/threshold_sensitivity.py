"""Complete the predeclared threshold grid after the seated-window correction.
Post-primary-result sensitivity; no winning threshold is promoted to primary.
"""
from study import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
def main():
    out=HERE/'threshold-sensitivity';out.mkdir(exist_ok=True);assert not (out/'summary.json').exists()
    save('threshold-sensitivity/protocol.json',{'timing':'after primary results; complete both originally declared alternative thresholds on posture-screened windows, report both; not a new primary or independent confirmation',
      'thresholds':[.05,.15],'source_protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':sha(HERE/'threshold_sensitivity.py'),
      'all_other_conditions':'same176, same3s+D, same outer/inner splits and Ridge grid','comparisons':['CleanDelay05 vs CleanSpeed','CleanDelay15 vs CleanSpeed']})
    source=HERE/'common176';d=dict(np.load(source/'input.npz'));rows=json.loads((source/'manifest.json').read_text());views={}
    for name,th in [('CleanDelay05',.05),('CleanDelay15',.15)]:
        ff=[]
        for i,m in enumerate(rows):
            a=read(TURN/'raw'/m['turn_file']);_,delay,_=features(a,m['clean_windows'],m['axis'],th);ff.append(np.r_[d['CleanSpeed'][i],delay])
        views[name]=np.stack(ff)
    np.savez_compressed(out/'input.npz',ids=d['ids'],y=d['y'],**views)
    predictions=[];selections=[];inner=[];maxdelta=0.;y=d['y'];models=0;innerfits=0
    with threadpool_limits(limits=1):
        for rep in json.loads((source/'splits.json').read_text()):
            for f in rep['splits']:
                tr=np.array(f['train']);te=np.array(f['test'])
                for name,x in views.items():
                    scores=[]
                    for alpha in ALPHAS:
                        pred=np.full(len(y),np.nan)
                        for v in f['inner']:
                            it=np.array(v['train']);iv=np.array(v['validation']);pp=fit(x,y,it,iv,alpha)[0]
                            sc=StandardScaler().fit(x[it]);mm=Ridge(alpha=alpha,solver='svd').fit(sc.transform(x[it]),y[it]);qq=np.maximum(mm.predict(sc.transform(x[iv])),0)
                            assert np.allclose(pp,qq,atol=1e-10,rtol=0);maxdelta=max(maxdelta,float(abs(pp-qq).max()));innerfits+=1;pred[iv]=pp
                        score=float(abs(pred[tr]-y[tr]).mean());scores.append(score)
                        inner.append({'repeat':rep['repeat'],'fold':f['fold'],'view':name,'alpha':alpha,'MAE':score})
                    alpha=ALPHAS[int(np.argmin(scores))];pp,raw,pair=fit(x,y,tr,te,alpha)
                    sc=StandardScaler().fit(x[tr]);mm=Ridge(alpha=alpha,solver='svd').fit(sc.transform(x[tr]),y[tr]);qq=np.maximum(mm.predict(sc.transform(x[te])),0)
                    assert np.allclose(pp,qq,atol=1e-10,rtol=0);maxdelta=max(maxdelta,float(abs(pp-qq).max()));models+=1
                    file=out/f'r{rep["repeat"]}-f{f["fold"]}-{name}.joblib';joblib.dump(pair,file,compress=3)
                    selections.append({'repeat':rep['repeat'],'fold':f['fold'],'view':name,'alpha':alpha,'sha256':sha(file),'file':file.name})
                    predictions.extend({'repeat':rep['repeat'],'fold':f['fold'],'view':name,'id':str(d['ids'][k]),'sample':int(k),'target':float(y[k]),'prediction':float(p),'raw_prediction':float(q)} for k,p,q in zip(te,pp,raw))
    p=pd.DataFrame(predictions);p.to_csv(out/'predictions.csv',index=False);save('threshold-sensitivity/inner.json',inner);save('threshold-sensitivity/selections.json',selections)
    base=pd.read_csv(source/'predictions.csv');base=base[base.view.isin(['SameTUG','CleanSpeed'])];p=pd.concat([p,base],ignore_index=True);p['scenario']='common176';p['family']='Ridge'
    assert not p.duplicated(['repeat','view','id']).any() and (p.groupby(['view','id']).size()==5).all()
    spec=importlib.util.spec_from_file_location('old_assess',HERE.parent/'20261031-opencap-leave283/assess.py');assess=importlib.util.module_from_spec(spec);spec.loader.exec_module(assess)
    part,persons,runs,metrics=assess.aggregate(p);persons.to_csv(out/'person-results.csv',index=False)
    contrasts=[assess.contrast(persons,runs,('common176','Ridge',name),('common176','Ridge',base),band) for name in views for base in ['CleanSpeed','SameTUG'] for band in ['all','ge20']]
    save('threshold-sensitivity/summary.json',{'metrics':metrics,'contrasts':contrasts})
    assert models==50 and innerfits==600
    save('threshold-sensitivity/verification.json',{'passed':True,'models_independently_refit':models,'inner_fits_independently_refit':innerfits,'max_difference':maxdelta,'same_splits_and_coverage':True,'sub283_absent':not any(p.id=='sub-283')})
    print(pd.DataFrame(metrics)[['view','band','MAE','underprediction']].to_string(index=False),flush=True)
    print(json.dumps([{'view':c['a'][2],'base':c['b'][2],'band':c['band'],'under_delta':c['underprediction']['difference'],'ci':c['underprediction']['conditional_bootstrap95'],'partitions':c['underprediction']['better_partitions']} for c in contrasts if c['band']=='ge20']),flush=True)
if __name__=='__main__':main()
