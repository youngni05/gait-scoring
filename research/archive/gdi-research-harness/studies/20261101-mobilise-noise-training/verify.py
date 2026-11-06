from study import *
def equal_avg(values,ids):
    values=np.asarray(values,dtype=float);ok=np.isfinite(values);ids=np.asarray(ids)[ok];values=values[ok];u,inv,count=np.unique(ids,return_inverse=True,return_counts=True)
    return float(np.mean(np.bincount(inv,weights=values)/count))
def scale_check(m,x,ids):
    w=weights(ids);w=w/w.sum();flat=x.reshape(-1,x.shape[-1]);fw=np.repeat(w/5,5);mean=np.average(flat,axis=0,weights=fw);std=np.maximum(np.sqrt(np.average((flat-mean)**2,axis=0,weights=fw)),1e-8)
    np.testing.assert_allclose(m['mean'],mean,rtol=0,atol=1e-12);np.testing.assert_allclose(m['std'],std,rtol=0,atol=1e-12)
def check_threshold(prob,y,ids,cut):
    neg=~y;assert equal_avg((prob[neg]>=cut).astype(float),ids[neg])<=.1+1e-12
    values=np.r_[0.,np.unique(prob),1.00000001];ix=np.flatnonzero(values==cut);assert len(ix)==1
    if ix[0]:assert equal_avg((prob[neg]>=values[ix[0]-1]).astype(float),ids[neg])>.1-1e-12
def run(mode):
    initialize();protocol=json.loads((HERE/'protocol.json').read_text());assert protocol['code_sha256']==sha(HERE/'study.py')
    for z,h in protocol['sources'].items():assert sha(z)==h
    replays=0;refits=0;error=0.
    for target,fold,seed in tasks(mode):
        tag=f'{target}-f{fold}-s{seed}';folder=STORE/'folds';q=json.loads((folder/f'{tag}.json').read_text());assert q['code_sha256']==sha(HERE/'study.py') and q['predictions_sha256']==sha(folder/f'{tag}.npz') and q['keys_sha256']==sha(folder/f'{tag}-keys.csv.gz')
        for z,h in q['sources'].items():assert sha(z)==h
        tr,te,inner,_=old.cell_data(target,'long',5,fold);assert set(tr.participantid).isdisjoint(te.participantid);d=pd.read_csv(folder/f'{tag}-keys.csv.gz');np.testing.assert_array_equal(d[KEY],te[KEY]);col='total_sppb_score' if target=='sppb' else 'tugtime';np.testing.assert_allclose(d.actual,te[col],rtol=0,atol=1e-12);np.testing.assert_array_equal(d['class'],old.classes(te[col].to_numpy(),target));z=np.load(folder/f'{tag}.npz');ids=tr.participantid.to_numpy();y=old.classes(tr[col].to_numpy(),target)
        for key,value in dict(ids=ids,visits=tr.visit_number.to_numpy(dtype='U'),y=y,inner=inner).items():np.testing.assert_array_equal(z[key],value)
        source=np.load(stress.STORE/'folds'/f'{tag}.npz')
        for j,name in enumerate(METHODS):
            info=q['models'][name];assert sha(info['path'])==info['sha256'];model=dict(np.load(info['path']));x=stress.model_input(p.bag_array(tr,5),name);scale_check(model,x,ids)
            if name.endswith('AugDS'):
                kind='Cadence' if name.startswith('Cadence') else 'Speed';assert info['epochs']==q['models'][kind+'DS']['epochs'];assert sha(info['history_path'])==info['history_sha256'];history=json.loads(Path(info['history_path']).read_text());assert history[-1]['epoch']==info['epochs'] and all(np.isfinite([r['clean_loss'],r['noisy_loss']]).all() for r in history)
                oof=z[name+'_oof'];cross=z[name+'_crosscal']
                for i,mi in enumerate(info['inner_models']):
                    a=inner!=i;b=~a;assert sha(mi['path'])==mi['sha256'] and sha(mi['history_path'])==mi['history_sha256'];m=dict(np.load(mi['path']));scale_check(m,x[a],ids[a]);pred=p.predict_ds(m,x[b]);np.testing.assert_allclose(pred,oof[b],rtol=0,atol=1e-12);replays+=1
                    t=old.temperature(oof[a],y[a],ids[a]);assert abs(t-info['cross_temperatures'][i])<1e-12;np.testing.assert_allclose(cross[b],old.temper(oof[b],t),rtol=0,atol=1e-12)
                assert abs(old.temperature(oof,y,ids)-info['temperature'])<1e-12;check_threshold(cross[:,old.lowclass(target)],y==old.lowclass(target),ids,info['thresholds']['cal']['fpr10'])
                if fold==0 and seed in [SEEDS[0],SEEDS[-1]]:
                    for i in [0,9]:
                        a=inner!=0;b=inner==0;xx=x[a] if i==0 else x;yy=y[a] if i==0 else y;ii=ids[a] if i==0 else ids;test=x[b] if i==0 else stress.model_input(stress.changed_bag(te,'clean')[0],name);expected=oof[b] if i==0 else p.predict_ds(model,test);m,h=fit_aug(xx,yy,ii,seed+fold*10+i,info['epochs']);np.testing.assert_array_equal(p.predict_ds(m,test),expected);refits+=1
            for i,scenario in enumerate(SCENARIOS):
                bag,_=stress.changed_bag(te,scenario);test=stress.model_input(bag,name);raw=p.predict_ds(model,test);pred=old.temper(raw,info['temperature']);err=float(np.max(abs(pred-z['probabilities'][i,j])));assert err<1e-12;error=max(error,err);np.testing.assert_array_equal(z['warnings'][i,j],pred[:,old.lowclass(target)]>=info['thresholds']['cal']['fpr10']);replays+=1
                if not name.endswith('AugDS'):
                    ix=stress.METHODS.index(name);np.testing.assert_array_equal(z['probabilities'][i,j],source['probabilities'][i,ix]);np.testing.assert_array_equal(z['warnings'][i,j],source['warnings'][i,ix])
    if mode=='main':metric_check()
    save(HERE/f'{mode}-verification.json',dict(passed=True,tasks=len(tasks(mode)),model_subset_replays=replays,refits=refits,max_prediction_error=error,tolerance=1e-12,verify_sha256=sha(__file__),analysis_sha256=sha(HERE/'analyze.py')));print('VERIFIED',mode,replays,refits,flush=True)

def metric_check():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');mt=pd.read_csv(HERE/'matched-by-seed.csv');assert len(m)==280 and len(mt)==280
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        frames=[];zs=[];ws=[]
        for f in range(5):
            tag=f'{target}-f{f}-s{seed}';frames.append(pd.read_csv(STORE/'folds'/f'{tag}-keys.csv.gz'));z=np.load(STORE/'folds'/f'{tag}.npz');zs.append(z['probabilities']);ws.append(z['warnings'])
        d=pd.concat(frames,ignore_index=True);order=d.sort_values(['fold']+KEY).index.to_numpy();d=d.iloc[order].reset_index(drop=True);assert not d.duplicated(KEY).any() and d.participantid.nunique()==(2174 if target=='sppb' else 1939);assert len(d)==(7356 if target=='sppb' else 5688);prob=np.concatenate(zs,2)[:,:,order];warn=np.concatenate(ws,2)[:,:,order];y=d['class'].to_numpy();ids=d.participantid.to_numpy();low=y==old.lowclass(target);truth=np.eye(4)[y];pairs=pd.read_csv(stress.ref.STORE/f'matches-{target}.csv.gz');a=pairs.positive_index.to_numpy();b=pairs.negative_index.to_numpy()
        assert (y[a]==old.lowclass(target)).all() and (y[b]!=old.lowclass(target)).all() and (ids[a]!=ids[b]).all();np.testing.assert_array_equal(d.cohort.to_numpy()[a],d.cohort.to_numpy()[b]);np.testing.assert_array_equal(d.fold.to_numpy()[a],d.fold.to_numpy()[b]);assert (abs(d.speed.to_numpy()[a]-d.speed.to_numpy()[b])<=.05+1e-12).all()
        for i,scenario in enumerate(SCENARIOS):
            for j,name in enumerate(METHODS):
                r=m[m.target.eq(target)&m.seed.eq(seed)&m.scenario.eq(scenario)&m.method.eq(name)].iloc[0];pr=prob[i,j];pred=pr.argmax(1);w=warn[i,j];recalls=[equal_avg((pred[y==c]==c).astype(float),ids[y==c]) for c in range(4)]
                values={'accuracy':equal_avg(pred==y,ids),'balanced_recall':float(np.mean(recalls)),'brier':equal_avg(((pr-truth)**2).sum(1),ids),'extreme_error':equal_avg(abs(pred-y)==3,ids),'low_miss':equal_avg(np.where(low,pred!=old.lowclass(target),np.nan),ids),'fpr10_miss':equal_avg(np.where(low,~w,np.nan),ids),'fpr10_fpr':equal_avg(np.where(~low,w,np.nan),ids),**{f'recall_{c}':recalls[c] for c in range(4)}}
                for key,value in values.items():assert abs(value-r[key])<1e-12
                delta=pr[a,old.lowclass(target)]-pr[b,old.lowclass(target)];r=mt[mt.target.eq(target)&mt.seed.eq(seed)&mt.scenario.eq(scenario)&mt.method.eq(name)].iloc[0];assert abs(equal_avg((delta>0).astype(float)+.5*(delta==0),ids[a])-r.pair_order_accuracy)<1e-12

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
