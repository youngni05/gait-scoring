"""Replay stored models, internal selections and independent person-level metrics."""
from run import *
from collections import Counter
TOL=1e-10
def eq(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);np.testing.assert_allclose(a,b,atol=TOL,rtol=0)
    return float(np.max(abs(a-b))) if a.size else 0.
def w(ids):
    c=Counter(ids);v=np.array([1/c[i] for i in ids]);return v/v.mean()
def scale(m,x,ids):
    mu=np.average(x,axis=0,weights=w(ids));eq(mu,m['scale'].mean_);eq(np.average((x-mu)**2,axis=0,weights=w(ids)),m['scale'].var_)
def selection_check(meta):
    rows=meta['rules'];se=meta['selection']
    def key(a):
        m=a['metrics'];return (-m['balanced_accuracy'],m['extreme'],m['miss'],m['fpr'],a['candidate'],RULES.index(a['rule']))
    def allowed(a,b):return all(a['metrics'][j]-b['metrics'][j]<=1e-12 for j in ['extreme','miss','fpr'])
    mode=next(a for a in rows if a['candidate']==0 and a['rule']=='mode')
    fixed=sorted([a for a in rows if a['candidate']==0 and allowed(a,mode)],key=key)[0]
    adj=sorted([a for a in rows if len(meta['candidates'][a['candidate']]['cuts'])==3 and allowed(a,fixed)],key=key)[0]
    mi=len(meta['candidates'])-1;mb=next(a for a in rows if a['candidate']==mi and a['rule']=='mode')
    merged=sorted([a for a in rows if a['candidate']==mi and allowed(a,mb)],key=key)[0]
    for scheme,chosen in [('fixed4',fixed),('adjusted4',adj),('merged3',merged)]:
        assert all(se[scheme][key]==chosen[key] for key in ['candidate','rule','metrics'])
    assert adj['metrics']['balanced_accuracy']>=fixed['metrics']['balanced_accuracy']-1e-12
def audit(path):
    tag=path.stem;meta=json.loads(path.read_text());target,band,n,fold,seed=meta['task']
    assert meta['split_sha256']==sha(old.SETS/'folds'/f'{target}-{band}-{n}-f{fold}.json')
    table=STORE/'folds'/f'{tag}.csv.gz';arr=STORE/'folds'/f'{tag}-inner.npz';assert sha(table)==meta['predictions_sha256'] and sha(arr)==meta['arrays_sha256']
    d=pd.read_csv(table);aa=np.load(arr);tr,te,inner,ref=old.cell_data(target,band,n,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    ids=tr.participantid.to_numpy();real=tr[col].to_numpy();x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();kc=len(FINE[target])+1
    np.testing.assert_array_equal(aa['ids'],ids);np.testing.assert_array_equal(aa['visits'],tr.visit_number.to_numpy(dtype='U'));np.testing.assert_array_equal(aa['inner'],inner);eq(aa['real'],real)
    assert not set(ids)&set(te.participantid)
    for i in range(3):assert not set(ids[inner==i])&set(ids[inner!=i])
    assert len(d)==9*len(te) and not d.duplicated(KEY+['family','scheme']).any();maxerr=0.;refits=0;modelcount=0
    for family,info in meta['families'].items():
        if family=='RFreg':
            for p,h in info['source_hashes'].items():assert sha(p)==h
            z=aa['reg_scores'];source=np.load(old.STORE/'regression-five'/f'{tag}-inner.npz');eq(z,source['RFreg5_scores'])
            baseline=pd.read_csv(old.STORE/'regression-five'/f'{tag}.csv.gz');zt=baseline.RFreg5_score.to_numpy()
        else:
            assert sha(info['path'])==info['sha256'];model=joblib.load(info['path']);scale(model,x,ids)
            assert model['model'].random_state==seed+fold*10+9 and model['model'].n_estimators==128 and model['model'].min_samples_leaf==20
            ft=fine_prob(model,xt,family,kc);maxerr=max(maxerr,eq(ft,aa[family+'_fine_test']));fo=aa[family+'_fine_oof'];modelcount+=1
            if band=='long' and n==5 and fold==0 and seed==SEEDS[0]:
                fy=labels(real,FINE[target])
                for i in range(4):
                    a=inner!=i if i<3 else np.ones(len(tr),dtype=bool);b=~a
                    mm=fine_fit(family,seed+fold*10+(i if i<3 else 9),x[a],fy[a],ids[a],kc)
                    actual=fine_prob(mm,x[b] if i<3 else xt,family,kc);maxerr=max(maxerr,eq(actual,fo[b] if i<3 else ft));refits+=1
        candidates=[]
        for ci,c in enumerate(info['candidates']):
            cut=c['cuts'];k=len(cut)+1;y=labels(real,cut);assert cut[0]==ANCHORS[target][0] and cut[-1]==ANCHORS[target][-1]
            if family=='RFreg':
                assert sha(c['path'])==c['sha256'];mm=joblib.load(c['path']);scale(mm,z[:,None],ids);raw=mm.predict_proba(zt[:,None]);modelcount+=1
                oo=aa[f'{family}_c{ci}_oof']
                if band=='long' and n==5 and fold==0 and seed==SEEDS[0]:
                    for i in range(3):
                        a=inner!=i;b=~a;rr=map_fit(z[a],y[a],ids[a],k);maxerr=max(maxerr,eq(rr.predict_proba(z[b,None]),oo[b]));refits+=1
            else:
                raw=coarsen(ft,target,cut);oo=coarsen(fo,target,cut);maxerr=max(maxerr,eq(oo,aa[f'{family}_c{ci}_oof']))
            assert oo.min()>=0;eq(oo.sum(1),np.ones(len(oo)));eq(raw.sum(1),np.ones(len(raw)))
            cross=aa[f'{family}_c{ci}_cross'];eq(old.temperature(oo,y,ids),c['temperature'])
            for i in range(3):
                a=inner!=i;b=~a;tt=old.temperature(oo[a],y[a],ids[a]);eq(tt,c['cross_temperatures'][i]);maxerr=max(maxerr,eq(cross[b],old.temper(oo[b],tt)))
            support=[[int(len(set(ids[(inner!=i)&(y==j)]))) for j in range(k)] for i in range(3)]
            shares=[float(np.average(y==j,weights=w(ids))) for j in range(k)];assert support==c['supports'];eq(shares,c['shares'])
            assert c['eligible']==(min(map(min,support))>=40 and min(shares)>=.03)
            candidates.append(dict(c,y=y,cross=cross))
            for scheme,sel in info['selection'].items():
                if sel['candidate']!=ci:continue
                g=d[d.family.eq(family)&d.scheme.eq(scheme)].reset_index(drop=True);np.testing.assert_array_equal(g[KEY],te[KEY]);eq(g.actual,te[col])
                np.testing.assert_array_equal(g['class'],labels(te[col].to_numpy(),cut));cal=old.temper(raw,c['temperature'])
                maxerr=max(maxerr,eq(g[[f'p{j}' for j in range(k)]],cal));assert g.classes.eq(k).all() and g.rule.eq(sel['rule']).all()
                if family=='RFreg' and scheme=='fixed4':maxerr=max(maxerr,eq(cal,baseline[[f'RFreg5_cal_p{j}' for j in range(4)]]))
                for rule,rr in [('mode','mode'),('selected',sel['rule'])]:np.testing.assert_array_equal(g[rule],decide(cal,target,rr))
                prior=np.bincount(y,weights=w(ids),minlength=k);prior/=prior.sum();eq(g[[f'prior{j}' for j in range(k)]],np.tile(prior,(len(g),1)))
                cuts=old.thresholds(cross[:,low(target,k)],(y==low(target,k)).astype(int),ids);assert cuts==sel['thresholds']
                for policy,threshold in cuts.items():np.testing.assert_array_equal(g[policy],cal[:,low(target,k)]>=threshold)
                # Low and best ground truth anchors stay identical across all partitions.
                truth=te[col].to_numpy();assert np.array_equal(g['class'].to_numpy()==low(target,k),(truth<=3) if target=='sppb' else (truth>20))
        selections,rows=choose(candidates,target,ids)
        # Selected row objects also carry the independently checked thresholds in the ledger.
        assert rows==[{key:r[key] for key in ['candidate','rule','metrics']} for r in info['rules']]
        for scheme in selections:assert all(selections[scheme][key]==info['selection'][scheme][key] for key in ['candidate','rule','metrics'])
        selection_check(info)
    return {'tag':tag,'max_error':maxerr,'refits':refits,'models':modelcount}

def mean_by_person(values,ids):return float(pd.Series(values,index=ids).groupby(level=0).mean().mean())
def audit_metrics(limit=None):
    import importlib.util
    spec=importlib.util.spec_from_file_location('adaptive_analysis',HERE/'analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);cells=module.cells
    m=pd.read_csv(HERE/'metrics-by-seed.csv');p=pd.read_csv(HERE/'probabilities-by-seed.csv');o=pd.read_csv(HERE/'warnings-by-seed.csv')
    assert len(m)==1620 and len(p)==540 and len(o)==1620;maxerr=0.;checked=0
    for index,(info,d) in enumerate(cells()):
        if limit is not None and index>=limit:break
        for (family,scheme),g in d.groupby(['family','scheme']):
            ids=g.participantid.to_numpy();y=g['class'].to_numpy();k=int(g.classes.iloc[0]);target=info['target'];lc=low(target,k);pos=y==lc;neg=~pos
            assert len(ids)==(7356 if target=='sppb' else 5688) and len(set(ids))==(2174 if target=='sppb' else 1939)
            def rows(df):
                ix=df.family.eq(family)&df.scheme.eq(scheme)
                for key,val in info.items():ix&=df[key].eq(val)
                return df[ix]
            for _,row in rows(m).iterrows():
                pr=g[row.decision].to_numpy();recalls=[mean_by_person(pr[y==j]==j,ids[y==j]) for j in range(k)]
                actual=[mean_by_person(pr==y,ids),np.mean(recalls),mean_by_person(abs(pr-y)==k-1,ids),mean_by_person(pr[pos]!=lc,ids[pos]),mean_by_person(pr[neg]==lc,ids[neg]),mean_by_person(abs(pr[pos]-y[pos])==k-1,ids[pos])]
                maxerr=max(maxerr,eq(actual,row[['accuracy','balanced_accuracy','extreme','miss','fpr','low_to_best']]));maxerr=max(maxerr,eq(recalls,row[[f'recall{j}' for j in range(k)]]));checked+=1
                widths=np.array([np.diff(json.loads(c)) for c in g.cuts]);mid=(pr>0)&(pr<k-1);tmid=(y>0)&(y<k-1)
                if mid.any():maxerr=max(maxerr,eq(mean_by_person(widths[np.flatnonzero(mid),pr[mid]-1],ids[mid]),row.prediction_middle_width))
                else:assert pd.isna(row.prediction_middle_width)
                maxerr=max(maxerr,eq(mean_by_person(widths[np.flatnonzero(tmid),y[tmid]-1],ids[tmid]),row.true_middle_width))
            probs=g[[f'p{j}' for j in range(k)]].to_numpy();prior=g[[f'prior{j}' for j in range(k)]].to_numpy();truth=np.eye(k)[y];pr=rows(p).iloc[0]
            actual=[mean_by_person(((probs-truth)**2).sum(1),ids),mean_by_person(((prior-truth)**2).sum(1),ids),mean_by_person((probs[:,lc]-pos)**2,ids)]
            maxerr=max(maxerr,eq(actual,pr[['brier','prior_brier','low_brier']]))
            rps=sum((probs[:,:j+1].sum(1)-(y<=j))**2 for j in range(k-1))/(k-1)
            maxerr=max(maxerr,eq([1-actual[0]/actual[1],mean_by_person(rps,ids)],pr[['brier_skill','rps']]))
            for _,row in rows(o).iterrows():
                dec=g[row.policy].to_numpy();maxerr=max(maxerr,eq([mean_by_person(dec[pos]==0,ids[pos]),mean_by_person(dec[neg]==1,ids[neg]),mean_by_person(dec==1,ids)],row[['miss','fpr','warning_rate']]))
    return {'independent_metric_max_error':maxerr,'metric_rows':checked}
def run():
    start=time.time();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'run.py')
    assert HERE.resolve().drive.upper()=='C:' and STORE.resolve().drive.upper()=='D:'
    for path,h in protocol['sources'].items():assert sha(path)==h
    analysis=json.loads((HERE/'analysis-completion.json').read_text());assert analysis['sha256']==sha(HERE/'analyze.py')
    for path,h in analysis['output_sha256'].items():assert sha(path)==h
    old.init();paths=sorted((STORE/'folds').glob('*.json'));assert len(paths)==300;results=[]
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(audit,p) for p in paths]),1):
            results.append(f.result())
            if i%25==0:print('MODEL_REPLAY',i,'/300',flush=True)
    save(HERE/'model-replay.json',{'results':results,'verify_sha256':sha(__file__),'ledger_hashes':{str(p):sha(p) for p in paths}})
    print('METRIC_REPLAY_START',flush=True);met=audit_metrics()
    result={'status':'passed','fold_seed_ledgers':300,'final_model_replays':sum(r['models'] for r in results),'selected_refits':sum(r['refits'] for r in results),
            'max_prediction_error':max(r['max_error'] for r in results),**met,'absolute_tolerance':TOL,'seconds':time.time()-start,'verify_sha256':sha(__file__),
            'checks':['source and model SHA256','person-disjoint outer and inner folds and same visits','training-only scaling','all final predictions replayed',
                      'cross-calibration and threshold replay','candidate support and eligibility','all inner decision metrics and constrained optimum replay','fixed low and best endpoints','independent person-level metrics'],
            'limits':'only preselected inner models refitted; repeated development folds; boundary selection evaluated only on outer held-out people, not a fresh external cohort'}
    save(HERE/'verification.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
