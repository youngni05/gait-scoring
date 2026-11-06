"""Same operating policies for saved regression models: separates ranking from cutoff effects."""
import study as s
from study import *
from sklearn.base import clone
from concurrent.futures import ThreadPoolExecutor

def initialize():
    s.initialize()

def work(task):
    target,band,k,fold=task;tag=f'{target}-{band}-{k}-f{fold}'
    dest=HERE/'regression';done=dest/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();col='total_sppb_score' if target=='sppb' else 'tugtime'
    d=s.DATA;cell=d[d[target+'_ok']&d.band.eq(band)&d.k.eq(k)].sort_values(KEY)
    tr=cell[cell.fold.ne(fold)].reset_index(drop=True);te=cell[cell.fold.eq(fold)].reset_index(drop=True)
    original=json.loads((SETS/'folds'/f'{tag}.json').read_text())
    x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();ids=tr.participantid.to_numpy()
    y=label(tr[col].to_numpy(),target);real=tr[col].to_numpy();inner=np.full(len(tr),-1)
    for i,sp in enumerate(original['inner_people']):inner[tr.participantid.isin(sp['valid'])]=i
    assert (inner>=0).all()
    out=te[KEY+['fold','cohort']].copy();out['actual']=te[col];out['positive']=label(te[col].to_numpy(),target)
    out['target']=target;out['band']=band;out['k']=k
    arrays={'participantid':ids,'visit_number':tr.visit_number.to_numpy(dtype='U'),'y':y,'inner':inner}
    ledger={'target':target,'band':band,'k':k,'fold':fold,'models':{},'inner_people':original['inner_people']}
    for name in ['SVR','RF']:
        path=BASE/'models'/f'{tag}-{name}.joblib';model=joblib.load(path)
        if name=='RF':model['model'].n_jobs=1
        oof=np.full(len(tr),np.nan)
        for i in range(3):
            a=inner!=i;b=~a;m=clone(model);w=weights(ids[a])
            m.fit(x[a],real[a],scale__sample_weight=w,model__sample_weight=w)
            if name=='SVR':assert m['model'].fit_status_==0
            oof[b]=m.predict(x[b])
        direction=-1. if target=='sppb' else 1.
        z=oof*direction;cp=np.full(len(y),np.nan)
        for i in range(3):
            a=inner!=i;b=~a;c=calibration(z[a],y[a],ids[a]);cp[b]=c.predict_proba(z[b,None])[:,1]
        c=calibration(z,y,ids);pred=model.predict(xt);p=c.predict_proba((pred*direction)[:,None])[:,1]
        prefix=name+'reg';out[prefix+'_prediction']=pred;out[prefix+'_cal']=p
        arrays[prefix+'_oof_prediction']=oof;arrays[prefix+'_crosscal']=cp
        rules={}
        for policy,t in thresholds(cp,y,ids).items():
            cuts={}
            for reject in [0.,.1,.2]:
                cut=-1. if reject==0 else quantile(margin(cp,t),weights(ids),reject)
                out[f'{prefix}_cal_{policy}_r{int(reject*100)}']=np.where(margin(p,t)<cut,-1,(p>=t).astype(int))
                cuts[str(reject)]=cut
            rules[policy]={'threshold':t,'reject_margin':cuts}
        cpfile=MODELS/f'{tag}-{prefix}-calibrator.joblib';joblib.dump(c,cpfile)
        ledger['models'][prefix]={'original_model':str(path),'original_sha256':sha(path),'calibrator':str(cpfile),'calibrator_sha256':sha(cpfile),'rules':rules}
    out.to_csv(dest/f'{tag}.csv',index=False);np.savez_compressed(dest/f'{tag}-inner.npz',**arrays)
    ledger['seconds']=time.time()-start;ledger['predictions_sha256']=sha(dest/f'{tag}.csv')
    save(done,ledger);return tag,round(ledger['seconds'],1)

def run():
    dest=HERE/'regression';dest.mkdir(exist_ok=True)
    save(dest/'protocol.json',{'code_sha256':sha(__file__),'models':'reuse saved original SVR/RF outer models exactly; do not refit final models',
          'calibration_data':'refit selected regression hyperparameters on exact inner train folds; held-out inner predictions feed sigmoid calibration',
          'limit':'hyperparameters were originally selected on these inner folds; only outer held-out comparisons are performance estimates',
          'seeds':'original regression was a single fixed seed; do not count it as five independent runs',
          'policies':'same p50/sens80/fpr10 and ambiguity 0/10/20 percent rejection as direct classifiers'})
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5)));start=time.time()
    initialize()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            tag,secs=f.result()
            if n%5==0:print(n,'/60',tag,secs,'elapsed',round(time.time()-start),flush=True)
    pd.concat([pd.read_csv(p) for p in sorted(dest.glob('*.csv'))],ignore_index=True).to_csv(dest/'predictions.csv.gz',index=False)
    save(dest/'completion.json',{'jobs':60,'inner_regression_fits':360,'calibrator_fits':480,'seconds':time.time()-start})
    print('REGRESSION_CONTROL_DONE',flush=True)

if __name__=='__main__':run()
