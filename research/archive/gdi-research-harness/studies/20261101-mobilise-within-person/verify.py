from study import *

def check(task):
    target,cohort,seed=task;tag=f'{target}-{cohort}-s{seed}';l=json.loads((STORE/f'{tag}.json').read_text())
    assert l['code_sha256']==sha(HERE/'study.py') and l['predictions_sha256']==sha(STORE/f'{tag}.csv.gz')
    assert sha(l['source_predictions'])==l['source_predictions_sha256'] and sha(l['source_ledger'])==l['source_ledger_sha256']
    source=pd.read_csv(l['source_predictions']);ledger=json.loads(Path(l['source_ledger']).read_text());d=pd.read_csv(STORE/f'{tag}.csv.gz')
    tr,te,_,_=ref.cell(target,cohort);col='total_sppb_score' if target=='sppb' else 'tugtime'
    assert set(tr.participantid).isdisjoint(te.participantid) and tr.cohort.ne(cohort).all() and te.cohort.eq(cohort).all()
    np.testing.assert_array_equal(source[KEY],te[KEY]);np.testing.assert_allclose(source.actual,te[col],rtol=0,atol=1e-12)
    expected=[]
    for person,group in source.groupby('participantid'):
        ordered=sorted(group.visit_number.tolist())
        expected.extend((int(person),a,b) for a,b in itertools.combinations(ordered,2))
    assert sorted(zip(d.participantid,d.visit_number_a,d.visit_number_b))==sorted(expected)
    assert not d.duplicated(['participantid','visit_number_a','visit_number_b']).any() and len(d)==l['pairs'] and d.participantid.nunique()==l['people']
    lookup=source.set_index(KEY);a=[(pid,visit) for pid,visit in zip(d.participantid,d.visit_number_a)];b=[(pid,visit) for pid,visit in zip(d.participantid,d.visit_number_b)]
    for side,index in [('a',a),('b',b)]:
        sd=lookup.loc[index]
        for name in ['actual','class','speed']:np.testing.assert_allclose(d[name+'_'+side],sd[name],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(d['cohort_'+side],sd.cohort)
    error=0.
    for name,kind,src in [('CoreRF','Core','LocoCoreRF'),('TurnRF','Turn','LocoTurnRF')]:
        info=ledger['models'][src];assert sha(info['path'])==info['sha256'];m=joblib.load(info['path']);x=ref.inputs(tr)[kind];ids=tr.participantid.to_numpy();center=np.average(x,axis=0,weights=weights(ids));var=np.average((x-center)**2,axis=0,weights=weights(ids))
        np.testing.assert_allclose(m['scale'].mean_,center,rtol=1e-10,atol=1e-12);np.testing.assert_allclose(m['scale'].var_,var,rtol=1e-10,atol=1e-12)
        raw=old.probabilities(m,ref.inputs(te)[kind],'RF');cal=old.temper(raw,info['temperature']);stored=source[[f'{src}_cal_p{j}' for j in range(4)]].to_numpy();err=float(np.max(abs(cal-stored)));assert err<1e-12;error=max(error,err)
        for side,index in [('a',a),('b',b)]:
            pos=source.set_index(KEY).index.get_indexer(index);assert (pos>=0).all()
            expected_class=cal@np.arange(4);risk=cal[:,old.lowclass(target)];warn=risk>=info['thresholds']['cal']['fpr10']
            for field,value in [('expected',expected_class),('risk',risk),('warn',warn.astype(int)),('class',cal.argmax(1))]:np.testing.assert_allclose(d[f'{name}_{field}_{side}'],value[pos],rtol=0,atol=1e-12)
    return error

def verify_metrics():
    m=pd.read_csv(HERE/'metrics-by-seed.csv');cm=pd.read_csv(HERE/'cohort-metrics-by-seed.csv');assert len(m)==40 and len(cm)==160;worst=0.
    for target,seed in itertools.product(['sppb','tug'],SEEDS):
        data=pd.concat([pd.read_csv(STORE/f'{target}-{c}-s{seed}.csv.gz') for c in COHORTS],ignore_index=True)
        assert data.participantid.nunique()==(1834 if target=='sppb' else 1653) and len(data)==(11055 if target=='sppb' else 6721)
        for cohort in [None]+COHORTS:
            d=data if cohort is None else data[data.held_cohort.eq(cohort)];scores=m[m.target.eq(target)&m.seed.eq(seed)] if cohort is None else cm[cm.target.eq(target)&cm.seed.eq(seed)&cm.cohort.eq(cohort)]
            truth=d.class_b.to_numpy()-d.class_a.to_numpy();low=old.lowclass(target);la=d.class_a.eq(low).to_numpy();lb=d.class_b.eq(low).to_numpy();changed=truth!=0;switch=la!=lb
            for _,r in scores.iterrows():
                name=r.method
                if name=='Speed':delta=(d.speed_b-d.speed_a).to_numpy()*(1 if target=='sppb' else -1)
                elif name=='NoChange':delta=np.zeros(len(d))
                else:delta=d[f'{name}_expected_b'].to_numpy()-d[f'{name}_expected_a'].to_numpy()
                direction=np.where(delta==0,.5,(np.sign(delta)==np.sign(truth)).astype(float))
                arrays={'direction':np.where(changed,direction,np.nan),'large_change_direction':np.where(abs(truth)>=2,direction,np.nan)}
                if name!='Speed':
                    wa=np.zeros(len(d),int) if name=='NoChange' else d[f'{name}_warn_a'].to_numpy();wb=np.zeros(len(d),int) if name=='NoChange' else d[f'{name}_warn_b'].to_numpy()
                    ca=np.zeros(len(d),int) if name=='NoChange' else d[f'{name}_class_a'].to_numpy();cb=np.zeros(len(d),int) if name=='NoChange' else d[f'{name}_class_b'].to_numpy()
                    arrays.update(class_delta_mae=abs(delta-truth),false_class_switch=np.where(~changed,ca!=cb,np.nan),correct_low_switch=np.where(switch,(wa==la)&(wb==lb),np.nan),false_low_switch=np.where(~switch,wa!=wb,np.nan))
                for key,value in arrays.items():
                    value=pd.DataFrame({'person':d.participantid.to_numpy(),'value':value}).dropna().groupby('person').value.mean().mean()
                    if np.isnan(value):assert pd.isna(r[key]);continue
                    error=abs(value-r[key]);assert error<1e-10;worst=max(worst,error)
    return worst

def run(mode):
    ref.initialize();start=time.time();p=json.loads((HERE/'protocol.json').read_text());assert p['code_sha256']==sha(HERE/'study.py')
    for f,digest in p['sources'].items():assert sha(f)==digest
    errors=[]
    for t in tasks(mode):
        while not (STORE/f'{t[0]}-{t[1]}-s{t[2]}.json').exists():time.sleep(1)
        errors.append(check(t))
    if mode=='main':
        while not (HERE/'summary.json').exists():time.sleep(1)
        metric_error=verify_metrics()
    else:metric_error=None
    save(HERE/f'{mode}-verification.json',{'passed':True,'tasks':len(errors),'model_replays':2*len(errors),'refits':0,'max_prediction_error':max(errors),'independent_metric_error':metric_error,'verify_sha256':sha(__file__),'analysis_sha256':sha(HERE/'analyze.py'),'seconds':time.time()-start});print('VERIFIED',mode,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pilot','main']);run(ap.parse_args().mode)
