from study import *
def equal_avg(values,ids):
    values=np.asarray(values,dtype=float);ok=np.isfinite(values);ids=np.asarray(ids)[ok];values=values[ok];u,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    return float(np.mean(np.bincount(inv,weights=values)/count))
def run(mode):
    initialize();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for q,h in protocol['sources'].items():assert sha(q)==h
    error=0.;replays=0
    for target,fold,seed in tasks(mode):
        tag=f'{target}-f{fold}-s{seed}';q=json.loads((STORE/'folds'/f'{tag}.json').read_text());assert q['code_sha256']==sha(HERE/'study.py') and q['predictions_sha256']==sha(STORE/'folds'/f'{tag}.npz') and q['keys_sha256']==sha(STORE/'folds'/f'{tag}-keys.csv.gz')
        for path,h in q['sources'].items():assert sha(path)==h
        tr,te,_,_=old.cell_data(target,'long',5,fold);d=pd.read_csv(STORE/'folds'/f'{tag}-keys.csv.gz');np.testing.assert_array_equal(d[KEY],te[KEY]);assert set(tr.participantid).isdisjoint(te.participantid);col='total_sppb_score' if target=='sppb' else 'tugtime';np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d['class'],old.classes(te[col].to_numpy(),target))
        z=np.load(STORE/'folds'/f'{tag}.npz');bag=p.bag_array(te,5);original=pd.read_csv(ref.STORE/'folds'/f'{tag}.csv.gz')
        for j,name in enumerate(METHODS):
            info=q['models'][name];assert sha(info['path'])==info['sha256'];model=dict(np.load(info['path'])) if name.endswith('DS') else joblib.load(info['path'])
            for i,scenario in enumerate(SCENARIOS):
                v=bag.copy()
                if scenario=='bias_m10':v[:,:,2]=bag[:,:,2]-.1
                elif scenario=='bias_p10':v[:,:,2]=bag[:,:,2]+.1
                elif scenario=='scale_m10':v[:,:,2]=bag[:,:,2]*.9
                elif scenario=='scale_p10':v[:,:,2]=bag[:,:,2]*1.1
                elif scenario.startswith('jitter'):
                    for row,(sid,visit) in enumerate(te[KEY].itertuples(index=False,name=None)):
                        key=int(hashlib.sha256(('20261101|'+str(sid)+'|'+str(visit)).encode()).hexdigest()[:16],16);v[row,:,2]=bag[row,:,2]+np.random.Generator(np.random.PCG64(key)).standard_normal(5)*(.05 if scenario=='jitter05' else .1)
                assert int((v[:,:,2]<0).sum())==q['clipped'][scenario];v[:,:,2]=np.maximum(0.,v[:,:,2]);visible=v[:,:,[2,0]] if name.startswith('Cadence') else v[:,:,[2]]
                if name.endswith('DS'):x=visible
                else:
                    x=np.array([row[np.lexsort((row[:,1],row[:,0]))] if row.shape[1]==2 else row[np.argsort(row[:,0],kind='stable')] for row in visible]).reshape(len(te),-1)
                raw=p.predict_ds(model,x) if name.endswith('DS') else old.probabilities(model,x,'RF');pred=old.temper(raw,info['temperature']);err=float(np.max(abs(pred-z['probabilities'][i,j])));assert err<1e-12;error=max(error,err);np.testing.assert_array_equal(z['warnings'][i,j],pred[:,old.lowclass(target)]>=info['thresholds']['cal']['fpr10']);replays+=1
                if scenario=='clean':np.testing.assert_allclose(pred,original[[f'{name}_cal_p{c}' for c in range(4)]],rtol=0,atol=1e-12)
    if mode=='main':
        m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv');assert len(m)==280 and len(mt)==280
        for target,seed in itertools.product(['sppb','tug'],SEEDS):
            frames=[];zs=[];ws=[]
            for f in range(5):
                tag=f'{target}-f{f}-s{seed}';frames.append(pd.read_csv(STORE/'folds'/f'{tag}-keys.csv.gz'));z=np.load(STORE/'folds'/f'{tag}.npz');zs.append(z['probabilities']);ws.append(z['warnings'])
            d=pd.concat(frames,ignore_index=True);order=d.sort_values(['fold']+KEY).index.to_numpy();d=d.iloc[order].reset_index(drop=True);assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939);assert len(d)==(7356 if target=='sppb' else 5688);prob=np.concatenate(zs,2)[:,:,order];warn=np.concatenate(ws,2)[:,:,order];y=d['class'].to_numpy();ids=d.participantid.to_numpy();low=y==old.lowclass(target);truth=np.eye(4)[y];pairs=pd.read_csv(ref.STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy()
            assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all() and (ids[a]!=ids[b]).all();np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b]);np.testing.assert_array_equal(d.fold.to_numpy()[a],d.fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
            for i,scenario in enumerate(SCENARIOS):
                for j,name in enumerate(METHODS):
                    r=m[m.target.eq(target)&m.seed.eq(seed)&m.scenario.eq(scenario)&m.method.eq(name)].iloc[0];pr=prob[i,j];pred=pr.argmax(1);w=warn[i,j];recalls=[equal_avg((pred[y==c]==c).astype(float),ids[y==c]) for c in range(4)]
                    values={'accuracy':equal_avg(pred==y,ids),'balanced_recall':float(np.mean(recalls)),'brier':equal_avg(((pr-truth)**2).sum(1),ids),'extreme_error':equal_avg(abs(pred-y)==3,ids),'low_miss':equal_avg(np.where(low,pred!=old.lowclass(target),np.nan),ids),'fpr10_miss':equal_avg(np.where(low,~w,np.nan),ids),'fpr10_fpr':equal_avg(np.where(~low,w,np.nan),ids),**{f'recall_{c}':recalls[c] for c in range(4)}}
                    for key,value in values.items():assert abs(value-r[key])<1e-12
                    delta=pr[a,old.lowclass(target)]-pr[b,old.lowclass(target)];r=mt[mt.target.eq(target)&mt.seed.eq(seed)&mt.scenario.eq(scenario)&mt.method.eq(name)].iloc[0];assert abs(equal_avg((delta>0).astype(float)+.5*(delta==0),ids[a])-r.pair_order_accuracy)<1e-12
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(tasks(mode)),model_scenario_replays=replays,new_fits=0,max_prediction_error=error,tolerance=1e-12,verify_sha256=sha(__file__),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,replays,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
