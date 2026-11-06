"""Repeat original selected RF regression settings under five seeds."""
import study as s
from study import *
from sklearn.base import clone

def work(task):
    target,band,k,fold,seed=task;tag=f'{target}-{band}-{k}-f{fold}-s{seed}';folder=STORE/'regression-five';done=folder/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,ref=cell_data(target,band,k,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();ids=tr.participantid.to_numpy();real=tr[col].to_numpy();y=classes(real,target)
    original=BASE/'models'/f'{target}-{band}-{k}-f{fold}-RF.joblib';template=joblib.load(original)
    z=np.zeros(len(tr));name='RFreg5'
    for i in range(3):
        a=inner!=i;b=~a;m=clone(template);m.set_params(model__random_state=seed+fold*10+i,model__n_jobs=1);w=weights(ids[a])
        m.fit(x[a],real[a],scale__sample_weight=w,model__sample_weight=w);z[b]=m.predict(x[b])
    m=clone(template);m.set_params(model__random_state=seed+fold*10+9,model__n_jobs=1);w=weights(ids)
    m.fit(x,real,scale__sample_weight=w,model__sample_weight=w);zt=m.predict(xt)
    mp=STORE/'models'/f'{tag}-{name}.joblib';joblib.dump(m,mp,compress=3)
    oof=np.zeros((len(tr),4))
    for i in range(3):
        a=inner!=i;b=~a;cal=mapper(z[a],y[a],ids[a]);oof[b]=cal.predict_proba(z[b,None])
    cal=mapper(z,y,ids);raw=cal.predict_proba(zt[:,None]);cp=STORE/'models'/f'{tag}-{name}-mapper.joblib';joblib.dump(cal,cp,compress=3)
    out=outframe(te,target,band,k,fold,seed);out[name+'_point']=classes(zt,target);out[name+'_score']=zt
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'inner':inner,'y':y,name+'_scores':z}
    info=finish_prob(name,oof,raw,y,ids,inner,out,target,arrays)
    info.update(path=str(cp),sha256=sha(cp),regressor=str(mp),regressor_sha256=sha(mp))
    out.to_csv(folder/f'{tag}.csv.gz',index=False);np.savez_compressed(folder/f'{tag}-inner.npz',**arrays)
    ledger={'task':list(task),'models':{name:info},'original_model':str(original),'original_sha256':sha(original),
            'selected_leaf':int(template['model'].min_samples_leaf),'seconds':time.time()-start,
            'predictions_sha256':sha(folder/f'{tag}.csv.gz'),'inner_sha256':sha(folder/f'{tag}-inner.npz')}
    save(done,ledger);return tag,round(ledger['seconds'],1)

def run():
    # This amendment precedes examination of new held-out results and only fixes repetition parity.
    amendment={'purpose':'original RF regression had only one seed; repeat its previously selected settings under all five seeds, preserving original baseline as historical control',
               'seeds':SEEDS,'settings':'clone original fold-selected RF hyperparameters; change random_state only and n_jobs=1',
               'inner_final_seeds':'same seed+fold*10+i / +9 as classifiers','probabilities':'same cross-fitted scalar-score mapping and temperature as historical regression control',
               'SVR':'deterministic existing predictions reused, not counted as five independent fits','code_sha256':sha(__file__)}
    ap=HERE/'regression-five-protocol.json'
    if ap.exists():assert json.loads(ap.read_text())==amendment
    save(ap,amendment);(STORE/'regression-five').mkdir(exist_ok=True);s.init();start=time.time()
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=3) as pool:
        for n,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,sec=f.result()
            if n%10==0:print(n,'/300',tag,sec,'elapsed',round(time.time()-start),flush=True)
    save(HERE/'regression-five-completion.json',{'tasks':300,'seconds':time.time()-start,'code_sha256':sha(__file__)})
    print('REGRESSION_FIVE_DONE',flush=True)

if __name__=='__main__':run()
