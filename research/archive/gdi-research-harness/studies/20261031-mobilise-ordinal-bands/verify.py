"""Independent split, stored-model replay, calibration and metric audits."""
import study as s
from study import *
from collections import Counter
from sklearn.base import clone

TOL=1e-10
def equal(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);np.testing.assert_allclose(a,b,rtol=0,atol=TOL)
    return float(np.max(abs(a-b))) if a.size else 0.
def pw(ids):
    c=Counter(ids);w=np.array([1/c[i] for i in ids]);return w/w.mean()
def wm(v,ids):return float(np.average(v,weights=pw(ids)))
def check_scale(m,x,ids):
    mean=np.average(x,axis=0,weights=pw(ids));var=np.average((x-mean)**2,axis=0,weights=pw(ids))
    equal(m['scale'].mean_,mean);equal(m['scale'].var_,var)
def check_threshold(p,y,ids,rules):
    # Direct weighted rates at saved cut and immediately adjacent candidate.
    candidates=np.r_[0.,np.unique(p),1.00000001];pos=y==1;neg=~pos
    def rate(cut,mask):return wm((p[mask]>=cut).astype(float),ids[mask])
    assert rules['p50']==.5
    for policy in ['fpr10','sens80']:
        cut=rules[policy];idx=np.flatnonzero(candidates==cut);assert len(idx)==1;i=int(idx[0])
        if policy=='fpr10':
            assert rate(cut,neg)<=.1+1e-12
            if i>0:assert rate(candidates[i-1],neg)>.1-1e-12
        else:
            assert rate(cut,pos)>=.8-1e-12
            if i+1<len(candidates):assert rate(candidates[i+1],pos)<.8+1e-12

def audit_file(path):
    folder=path.parent;tag=path.stem;ledger=json.loads(path.read_text());target,band,k,fold=ledger['task'][:4]
    seed=ledger['task'][4] if len(ledger['task'])==5 else -1
    tr,te,inner,ref=cell_data(target,band,k,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    ids=tr.participantid.to_numpy();y=classes(tr[col].to_numpy(),target);x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy()
    if 'reference_split_sha256' in ledger:assert ledger['reference_split_sha256']==sha(SETS/'folds'/f'{target}-{band}-{k}-f{fold}.json')
    assert sha(folder/f'{tag}.csv.gz')==ledger['predictions_sha256'];assert sha(folder/f'{tag}-inner.npz')==ledger['inner_sha256']
    a=np.load(folder/f'{tag}-inner.npz');d=pd.read_csv(folder/f'{tag}.csv.gz')
    np.testing.assert_array_equal(a['ids'],ids);np.testing.assert_array_equal(a['visits'],tr.visit_number.to_numpy(dtype='U'))
    np.testing.assert_array_equal(a['inner'],inner);np.testing.assert_array_equal(a['y'],y)
    np.testing.assert_array_equal(d[KEY].to_numpy(),te[KEY].to_numpy());equal(d.actual,te[col])
    np.testing.assert_array_equal(d['class'],classes(te[col].to_numpy(),target))
    assert (d.seed==seed).all() and d.fold.eq(fold).all()
    for i in range(3):assert not set(ids[inner==i])&set(ids[inner!=i])
    maximum=0.;refits=0
    for name,info in ledger['models'].items():
        assert sha(info['path'])==info['sha256'];m=joblib.load(info['path'])
        if name in ['RF','OrdinalRF']:
            check_scale(m,x,ids);assert m['model'].random_state==seed+fold*10+9
            assert m['model'].n_estimators==128 and m['model'].min_samples_leaf==20
            raw=probabilities(m,xt,name)
        else:
            z=a[name+'_scores'] if name=='RFreg5' else np.load(PREV/'regression'/f'{tag}-inner.npz')[name+'_oof_prediction']
            check_scale(m,z[:,None],ids)
            if name=='RFreg5':
                assert sha(info['regressor'])==info['regressor_sha256'];rm=joblib.load(info['regressor']);check_scale(rm,x,ids)
                assert sha(ledger['original_model'])==ledger['original_sha256']
                original=joblib.load(ledger['original_model']);pa=original['model'].get_params();pb=rm['model'].get_params()
                assert all(pa[c]==pb[c] for c in pa if c not in ['random_state','n_jobs'])
                assert rm['model'].random_state==seed+fold*10+9
                zt=rm.predict(xt);maximum=max(maximum,equal(zt,d[name+'_score']))
            else:
                for f,h in ledger['input_hashes'].items():assert sha(f)==h
                zt=pd.read_csv(PREV/'regression'/f'{tag}.csv')[name+'_prediction'].to_numpy()
            np.testing.assert_array_equal(d[name+'_point'],classes(zt,target));raw=m.predict_proba(zt[:,None])
        for v,p in [('raw',raw),('cal',temper(raw,info['temperature']))]:
            saved=d[[f'{name}_{v}_p{j}' for j in range(4)]].to_numpy();maximum=max(maximum,equal(saved,p))
            assert saved.min()>=0;equal(saved.sum(1),np.ones(len(saved)))
            for policy,cut in info['thresholds'][v].items():np.testing.assert_array_equal(d[f'{name}_{v}_{policy}'],p[:,lowclass(target)]>=cut)
        np.testing.assert_array_equal(raw.argmax(1),temper(raw,info['temperature']).argmax(1))
        oof=a[name+'_oof'];cross=a[name+'_crosscal'];assert oof.min()>=0;equal(oof.sum(1),np.ones(len(oof)))
        # Same-input deterministic optimization replay; independent formula checks likelihood equivalence.
        equal(temperature(oof,y,ids),info['temperature'])
        for i in range(3):
            train=inner!=i;valid=~train;t=temperature(oof[train],y[train],ids[train]);equal(t,info['cross_temperatures'][i])
            maximum=max(maximum,equal(cross[valid],temper(oof[valid],t)))
        for v,ip in [('raw',oof),('cal',cross)]:check_threshold(ip[:,lowclass(target)],(y==lowclass(target)).astype(int),ids,info['thresholds'][v])
        # Preselected cells, without looking at outcome, replay actual inner fitting and final fit.
        selected=band=='long' and k==5 and fold==0 and seed==SEEDS[0]
        if selected:
            if name in ['RF','OrdinalRF','RFreg5']:
                re_oof=np.zeros_like(oof) if name!='RFreg5' else np.zeros(len(tr))
                for i in range(4):
                    train=inner!=i if i<3 else np.ones(len(tr),dtype=bool);valid=~train
                    si=seed+fold*10+(i if i<3 else 9)
                    if name=='RFreg5':
                        mm=clone(original);mm.set_params(model__random_state=si,model__n_jobs=1);w=weights(ids[train])
                        mm.fit(x[train],tr[col].to_numpy()[train],scale__sample_weight=w,model__sample_weight=w)
                        result=mm.predict(x[valid] if i<3 else xt)
                    else:
                        mm=fit(make(name,si),x[train],y[train],ids[train],name);result=probabilities(mm,x[valid] if i<3 else xt,name)
                    if i<3:re_oof[valid]=result
                    else:maximum=max(maximum,equal(result,zt if name=='RFreg5' else raw))
                    refits+=1
                maximum=max(maximum,equal(re_oof,z if name=='RFreg5' else oof))
                if name=='RFreg5':
                    for i in range(3):
                        train=inner!=i;valid=~train;mm=mapper(z[train],y[train],ids[train]);maximum=max(maximum,equal(mm.predict_proba(z[valid,None]),oof[valid]));refits+=1
    return {'tag':tag,'max_replay_error':maximum,'refits':refits,'models':len(ledger['models'])}

def audit_inputs():
    p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(HERE/'study.py')
    for f,h in p['source_hashes'].items():assert sha(f)==h
    assert json.loads((HERE/'regression-five-protocol.json').read_text())['code_sha256']==sha(HERE/'regression_five.py')
    bouts=pd.read_csv(BASE/'selected-bouts.csv');bouts=bouts[bouts.draw.eq(0)]
    counts={}
    for target,npeople,nvisits in [('sppb',2174,7356),('tug',1939,5688)]:
        reference=None
        for band,k in itertools.product(['short','long'],[1,3,5]):
            d=s.DATA[s.DATA[target+'_ok']&s.DATA.band.eq(band)&s.DATA.k.eq(k)].sort_values(KEY)
            assert d.participantid.nunique()==npeople and len(d)==nvisits
            assert d.groupby('participantid').fold.nunique().max()==1
            keys=d[KEY].to_numpy()
            if reference is None:reference=keys
            else:np.testing.assert_array_equal(reference,keys)
            b=bouts[bouts.band.eq(band)&bouts['rank'].le(k)];lo,hi=(4,10) if band=='short' else (30,60)
            assert b.duration.ge(lo).all() and b.duration.lt(hi).all()
            assert b.groupby(KEY).size().eq(k).all()
            med=b.groupby(KEY)[FEATURES].median().reset_index();joined=d.merge(med,on=KEY,suffixes=('','_replay'),validate='one_to_one')
            assert len(joined)==len(d);equal(joined[FEATURES],joined[[c+'_replay' for c in FEATURES]])
        counts[target]={'people':npeople,'visits':nvisits}
    return counts

def audit_metrics():
    from analyze import load
    d=load();m=pd.read_csv(HERE/'metrics-by-seed.csv');ops=pd.read_csv(HERE/'low-function-by-seed.csv');cs=pd.read_csv(HERE/'class-probabilities-by-seed.csv')
    assert len(m)==840 and len(ops)==1800 and len(d)==30*(7356+5688)
    maxerr=0.
    for (target,band,k,seed),g in d.groupby(['target','band','k','seed']):
        ids=g.participantid.to_numpy();y=g['class'].to_numpy();low=lowclass(target);pos=y==low;neg=~pos
        assert len(g)==(7356 if target=='sppb' else 5688) and len(set(ids))==(2174 if target=='sppb' else 1939)
        sub=m[m.target.eq(target)&m.band.eq(band)&m.k.eq(k)&m.seed.eq(seed)].set_index('method')
        for method,row in sub.iterrows():
            if method.endswith('_point'):pred=g[method].to_numpy()
            else:
                p=g[[f'{method}_p{j}' for j in range(4)]].to_numpy();pred=p.argmax(1)
                brier=np.array([sum((p[i,j]-int(y[i]==j))**2 for j in range(4)) for i in range(len(y))])
                rps=sum((p[:,:j+1].sum(1)-(y<=j))**2 for j in range(3))/3
                maxerr=max(maxerr,equal([wm(brier,ids),wm(rps,ids),wm(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)),ids)],row[['brier','rps','log_loss']]))
                cls=cs[cs.target.eq(target)&cs.band.eq(band)&cs.k.eq(k)&cs.seed.eq(seed)&cs.method.eq(method)].set_index('class')
                w=pw(ids)
                for j in range(4):
                    ece=0.
                    for b in range(10):
                        ix=(p[:,j]>=b/10)&((p[:,j]<(b+1)/10) if b<9 else (p[:,j]<=1+1e-12))
                        if ix.any():ece+=abs(np.sum(w[ix]*(p[ix,j]-(y[ix]==j))))/w.sum()
                    maxerr=max(maxerr,equal([wm((p[:,j]-(y==j))**2,ids),ece],cls.loc[j,['brier','ece10']]))
            err=abs(pred-y)
            values=[wm(pred==y,ids),wm(err,ids),wm(err==3,ids),wm(pred[pos]!=low,ids[pos]),wm(err[pos]==3,ids[pos]),wm(pred[neg]==low,ids[neg])]
            maxerr=max(maxerr,equal(values,row[['accuracy','band_mae','opposite_extreme','low_miss','low_to_best','argmax_fpr']]))
            maxerr=max(maxerr,equal([np.mean([wm(pred[y==j]==j,ids[y==j]) for j in range(4)]),wm(err>=2,ids),wm(err[pos]>=2,ids[pos])],row[['balanced_recall','two_or_more_bands','low_two_or_more']]))
            if method not in ['Prior','SVRreg_point','RFreg_point','RFreg5_point']:
                rows=ops[ops.target.eq(target)&ops.band.eq(band)&ops.k.eq(k)&ops.seed.eq(seed)&ops.method.eq(method)]
                for _,r in rows.iterrows():
                    dec=g[f'{method}_{r.policy}'].to_numpy();maxerr=max(maxerr,equal([wm(dec[pos]==0,ids[pos]),wm(dec[neg]==1,ids[neg])],r[['miss','fpr']]))
    return {'max_independent_metric_error':maxerr,'metric_rows':len(m),'policy_rows':len(ops)}

def run():
    start=time.time();s.init();counts=audit_inputs();paths=[]
    for folder,n in [('folds',300),('regression',60),('regression-five',300)]:
        pp=sorted((STORE/folder).glob('*.json'));assert len(pp)==n;paths+=pp
    results=[]
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=4) as pool:
        for n,f in enumerate(as_completed([pool.submit(audit_file,p) for p in paths]),1):
            results.append(f.result())
            if n%50==0:print('replay',n,'/',len(paths),flush=True)
    metrics=audit_metrics()
    result={'status':'passed','people':counts,'ledgers':len(results),'final_model_or_mapper_replays':sum(r['models'] for r in results),
            'selected_refits':sum(r['refits'] for r in results),'max_prediction_replay_error':max(r['max_replay_error'] for r in results),
            **metrics,'absolute_tolerance':TOL,'seconds':time.time()-start,'verification_sha256':sha(__file__),
            'checks':['source and model SHA256','identical participants visits and selected-bout median features across six conditions','person-disjoint outer and inner folds',
                      'training-only weighted scaling','four class boundaries and label alignment','all stored final predictions replayed','temperature and cross-calibration replay',
                      'internal threshold feasibility and adjacent-cut extremality','preselected inner/final forest refits','independent probability class and operating metrics'],
            'limits':'selected inner fits replayed, not every inner forest; paired bootstrap is conditional on fixed fitted models and reused development folds'}
    save(HERE/'verification.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
