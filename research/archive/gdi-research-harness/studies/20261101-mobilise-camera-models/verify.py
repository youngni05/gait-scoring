from study import *

def mean(values,ids):return float(pd.DataFrame({'id':ids,'value':values}).dropna().groupby('id').value.mean().mean())

def check_scale(m,x,ids,leaf):
    center=np.average(x,axis=0,weights=weights(ids));var=np.average((x-center)**2,axis=0,weights=weights(ids))
    np.testing.assert_allclose(m['scale'].mean_,center,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(m['scale'].var_,var,rtol=1e-10,atol=1e-12)
    assert m['model'].n_estimators==128 and m['model'].min_samples_leaf==leaf and m['model'].max_features==1.

def ds_scale(bundle,x,ids):
    flat=x.reshape(-1,x.shape[-1]);w=weights(ids);w=w/w.sum();fw=np.repeat(w/x.shape[1],x.shape[1]);center=np.average(flat,axis=0,weights=fw);std=np.maximum(np.sqrt(np.average((flat-center)**2,axis=0,weights=fw)),1e-8)
    np.testing.assert_allclose(bundle['mean'],center,rtol=0,atol=1e-12);np.testing.assert_allclose(bundle['std'],std,rtol=0,atol=1e-12)

def threshold(p,y,ids,cut,policy):
    neg=y==0;pos=~neg
    if policy=='fpr10':
        assert mean((p[neg]>=cut).astype(float),ids[neg])<=.1+1e-12
        candidates=np.r_[0.,np.unique(p),1.00000001];ix=np.flatnonzero(candidates==cut);assert len(ix)==1
        if ix[0]>0:assert mean((p[neg]>=candidates[ix[0]-1]).astype(float),ids[neg])>.1-1e-12
    elif policy=='sens80':assert mean((p[pos]>=cut).astype(float),ids[pos])>=.8-1e-12
    else:assert cut==.5

def choose(candidates,y,ids,values,large):
    acc=[mean((z.argmax(1)==y).astype(float),ids) for z in candidates];nll=[mean(-np.log(np.maximum(z[np.arange(len(y)),y],1e-12)),ids) for z in candidates]
    index=sorted(range(len(values)),key=lambda i:(-round(acc[i],12),round(nll[i],12),-values[i] if large else values[i]))[0]
    return index,np.array(acc),np.array(nll)

def check(task,refit=False):
    target,fold,seed=task;tag=f'{target}-f{fold}-s{seed}';folder=STORE/'folds';l=json.loads((folder/f'{tag}.json').read_text())
    assert l['code_sha256']==sha(HERE/'study.py') and l['predictions_sha256']==sha(folder/f'{tag}.csv.gz') and l['inner_sha256']==sha(folder/f'{tag}-inner.npz')
    for q,digest in l['source_files'].items():assert sha(q)==digest
    d=pd.read_csv(folder/f'{tag}.csv.gz');ins=np.load(folder/f'{tag}-inner.npz');tr,te,inner,split=old.cell_data(target,'long',5,fold);x,xt=inputs(tr),inputs(te);ids=tr.participantid.to_numpy();col='total_sppb_score' if target=='sppb' else 'tugtime';y=old.classes(tr[col].to_numpy(),target)
    assert set(tr.participantid).isdisjoint(te.participantid)
    for key,value in {'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}.items():np.testing.assert_array_equal(ins[key],value)
    np.testing.assert_array_equal(d[KEY],te[KEY]);np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d['class'],old.classes(te[col].to_numpy(),target))
    count=0;refits=0;error=0.
    for kind in ['Speed','Cadence']:
        for name in [kind+'FixedRF',kind+'AccRF',kind+'DS']:
            info=l['models'][name];assert sha(info['path'])==info['sha256'];isds=name.endswith('DS');dtype='ds' if isds else 'rf';xx=x[kind][dtype];test=xt[kind][dtype]
            m=dict(np.load(info['path'])) if isds else joblib.load(info['path'])
            if isds:ds_scale(m,xx,ids);raw=p.predict_ds(m,test);np.testing.assert_allclose(raw,p.predict_ds(m,test[:,::-1]),rtol=0,atol=1e-12)
            else:check_scale(m,xx,ids,info.get('selected_leaf',20));raw=old.probabilities(m,test,'RF')
            count+=1;np.testing.assert_allclose(raw,d[[f'{name}_raw_p{j}' for j in range(4)]],rtol=0,atol=1e-12);oof=ins[name+'_oof']
            if name.endswith('AccRF'):
                candidates=ins[kind+'_rf_candidates'];best,acc,nll=choose(candidates,y,ids,LEAVES,True);assert info['selected_leaf']==LEAVES[best]
                np.testing.assert_allclose(acc,info['candidate_accuracy'],rtol=0,atol=1e-12);np.testing.assert_allclose(nll,info['candidate_nll'],rtol=0,atol=1e-12);np.testing.assert_array_equal(oof,candidates[best])
                for c,ci in enumerate(l['rf_candidates'][kind]):
                    for i,mi in enumerate(ci['inner_models']):
                        a=inner!=i;b=~a;assert sha(mi['path'])==mi['sha256'];model=joblib.load(mi['path']);check_scale(model,xx[a],ids[a],LEAVES[c]);prob=old.probabilities(model,xx[b],'RF');err=float(np.max(abs(prob-candidates[c,b])));assert err<1e-12;error=max(error,err);count+=1
            else:
                if isds:
                    candidates=ins[kind+'_ds_candidates'];best,acc,nll=choose(candidates,y,ids,EPOCHS,False);assert info['epochs']==EPOCHS[best]
                    np.testing.assert_allclose(acc,info['candidate_accuracy'],rtol=0,atol=1e-12);np.testing.assert_allclose(nll,info['candidate_nll'],rtol=0,atol=1e-12);np.testing.assert_array_equal(oof,candidates[best])
                for i,mi in enumerate(info['inner_models']):
                    a=inner!=i;b=~a;assert sha(mi['path'])==mi['sha256'];model=dict(np.load(mi['path'])) if isds else joblib.load(mi['path'])
                    if isds:ds_scale(model,xx[a],ids[a]);prob=p.predict_ds(model,xx[b])
                    else:check_scale(model,xx[a],ids[a],20);prob=old.probabilities(model,xx[b],'RF')
                    err=float(np.max(abs(prob-oof[b])));assert err<1e-12;error=max(error,err);count+=1
            temperature=old.temperature(oof,y,ids);assert abs(temperature-info['temperature'])<1e-12;cal=old.temper(raw,temperature);np.testing.assert_allclose(cal,d[[f'{name}_cal_p{j}' for j in range(4)]],rtol=0,atol=1e-12)
            cross=ins[name+'_crosscal']
            for i in range(3):
                a=inner!=i;b=~a;t=old.temperature(oof[a],y[a],ids[a]);assert abs(t-info['cross_temperatures'][i])<1e-12;np.testing.assert_allclose(cross[b],old.temper(oof[b],t),rtol=0,atol=1e-12)
            for variant,ip,op in [('raw',oof,raw),('cal',cross,cal)]:
                for policy,cut in info['thresholds'][variant].items():threshold(ip[:,old.lowclass(target)],(y==old.lowclass(target)).astype(int),ids,cut,policy);np.testing.assert_array_equal(op[:,old.lowclass(target)]>=cut,d[f'{name}_{variant}_{policy}'].astype(bool))
            if refit and not name.endswith('FixedRF'):
                for i in [0,9]:
                    a=inner!=i if i==0 else np.ones(len(tr),bool);test=xx[inner==0] if i==0 else xt[kind][dtype];expected=oof[inner==0] if i==0 else raw
                    # Match the actual training path: the final fit uses the full array,
                    # not a boolean copy that changes its floating-point reduction layout.
                    train_x=xx[a] if i==0 else xx;train_y=y[a] if i==0 else y;train_ids=ids[a] if i==0 else ids
                    if isds:
                        if i==0:
                            _,_,prob=ds_fit(train_x,train_y,train_ids,seed+fold*10+i,600,test);np.testing.assert_array_equal(np.stack(prob),ins[kind+'_ds_candidates'][:,inner==0]);pred=prob[EPOCHS.index(info['epochs'])]
                        else:bundle,_,_=ds_fit(train_x,train_y,train_ids,seed+fold*10+i,info['epochs']);pred=p.predict_ds(bundle,test)
                    else:
                        model=old.make('RF',seed+fold*10+i);model['model'].set_params(min_samples_leaf=info['selected_leaf']);model=old.fit(model,train_x,train_y,train_ids,'RF');pred=old.probabilities(model,test,'RF')
                    try:np.testing.assert_array_equal(pred,expected)
                    except AssertionError:
                        print('REFIT_MISMATCH',task,name,'stage',i,'max_error',float(np.max(abs(pred-expected))),flush=True)
                        if isds and i==9:
                            print('PARAMETER_DIFFERENCES',[(k,float(np.max(abs(bundle[k]-m[k])))) for k in m if not np.array_equal(bundle[k],m[k])],flush=True)
                            print('SAVED_FORWARD_REPEAT',float(np.max(abs(p.predict_ds(m,test)-expected))),flush=True)
                        raise
                    refits+=1
    return {'models':count,'refits':refits,'error':error}

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv');assert len(m)==130 and len(mt)==60;worst=0.
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        d=pd.concat([pd.read_csv(STORE/'folds'/f'{target}-f{f}-s{seed}.csv.gz') for f in range(5)]).sort_values(['fold']+KEY).reset_index(drop=True)
        assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939) and len(d)==(7356 if target=='sppb' else 5688)
        y=d['class'].to_numpy();ids=d.participantid.to_numpy();truth=np.eye(4)[y];low=y==old.lowclass(target)
        for _,r in m[m.target.eq(target)&m.seed.eq(seed)].iterrows():
            name=r.method;p=d[[f'{name}_p{j}' for j in range(4)]].to_numpy();pred=p.argmax(1);assert p.min()>=0 and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=0,atol=1e-12)
            vectors={'accuracy':(pred==y).astype(float),'brier':((p-truth)**2).sum(1),'rps':((p.cumsum(1)[:,:3]-truth.cumsum(1)[:,:3])**2).mean(1),'log_loss':-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)),'extreme_error':(abs(pred-y)==3).astype(float),'low_miss':np.where(low,pred!=old.lowclass(target),np.nan),'fpr':np.where(~low,pred==old.lowclass(target),np.nan)}
            recalls=[]
            for j in range(4):
                recall=mean((pred[y==j]==j).astype(float),ids[y==j]);assert abs(recall-r[f'recall_{j}'])<1e-10;recalls.append(recall)
            assert abs(np.mean(recalls)-r.balanced_recall)<1e-10
            if name!='Prior':warn=d[f'{name}_fpr10'].to_numpy();vectors.update(fpr10_miss=np.where(low,warn==0,np.nan),fpr10_fpr=np.where(~low,warn==1,np.nan))
            for key,value in vectors.items():err=abs(mean(value,ids)-r[key]);assert err<1e-10;worst=max(worst,err)
        pairs=pd.read_csv(STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy();assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all() and (ids[a]!=ids[b]).all()
        np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b]);np.testing.assert_array_equal(d.fold.to_numpy()[a],d.fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
        for name in METHODS:
            risk=d[f'{name}_cal_p{old.lowclass(target)}'].to_numpy();diff=risk[a]-risk[b];row=mt[mt.target.eq(target)&mt.seed.eq(seed)&mt.method.eq(name+'_cal')].iloc[0]
            assert abs(mean((diff>0).astype(float)+.5*(diff==0),ids[a])-row.pair_order_accuracy)<1e-10
    return worst


def run(mode):
    initialize();start=time.time();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for q,digest in protocol['sources'].items():assert sha(q)==digest
    results=[]
    for i,t in enumerate(tasks(mode),1):
        while not (STORE/'folds'/f'{t[0]}-f{t[1]}-s{t[2]}.json').exists():time.sleep(1)
        results.append(check(t,t[1]==0 and t[2] in [SEEDS[0],SEEDS[-1]]))
        if i%10==0:print('REPLAY',i,flush=True)
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error=verify_metrics()
    else:metric_error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(results),'model_replays':sum(r['models'] for r in results),'refits':sum(r['refits'] for r in results),'max_prediction_error':max(r['error'] for r in results),'independent_metric_error':metric_error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start});print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
