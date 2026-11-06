"""Train-only band/rule selection with fixed low and best-function anchors."""
from pathlib import Path
import sys,json,itertools,time,argparse
HERE=Path(__file__).resolve().parent;PREVIOUS=HERE.parent/'20261031-mobilise-ordinal-bands'
sys.path.insert(0,str(PREVIOUS))
import study as old
from study import np,pd,joblib,sha,save,weights,Pipeline,StandardScaler,LogisticRegression,ThreadPoolExecutor,as_completed,threadpool_limits
STORE=Path('D:/gait-research-data/D_P/mobilise-adaptive-bands-20261031')
SEEDS=old.SEEDS;KEY=old.KEY;FEATURES=old.FEATURES
MIDDLES={'sppb':[6.5,4.5,5.5,7.5,8.5],'tug':[15.,11.,12.,13.,14.,16.,17.,18.,19.]}
ANCHORS={'sppb':[3.5,9.5],'tug':[10.,20.]}
FINE={t:sorted(ANCHORS[t]+MIDDLES[t]) for t in MIDDLES}
FAMILIES=['RFreg','RF','OrdinalRF']
RULES=['mode','absolute','squared']+[f'mix-{l}-{a}-{b}' for l,a,b in itertools.product([.5,2.],[0.,1.],[0.,1.])]

def labels(y,cuts):return np.digitize(y,cuts,right=True)
def low(target,k):return 0 if target=='sppb' else k-1
def fine_fit(name,seed,x,y,ids,k):
    m=old.make(name,seed);w=weights(ids);yy=(y[:,None]>np.arange(k-1)).astype(float) if name=='OrdinalRF' else y
    m.fit(x,yy,scale__sample_weight=w,model__sample_weight=w);return m
def fine_prob(m,x,name,k):
    if name=='RF':
        p=np.zeros((len(x),k));p[:,m.classes_.astype(int)]=m.predict_proba(x)
    else:
        q=m.predict(x);assert (np.diff(q,axis=1)<=1e-12).all()
        p=np.diff(np.column_stack([np.zeros(len(x)),1-q,np.ones(len(x))]),axis=1)
    assert p.min()>=-1e-12;p=np.maximum(p,0);p/=p.sum(1,keepdims=True);return p
def coarsen(p,target,cuts):
    indices=[FINE[target].index(c)+1 for c in cuts];return np.column_stack([a.sum(1) for a in np.split(p,indices,axis=1)])
def map_fit(z,y,ids,k):
    m=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-8))]);w=weights(ids)
    m.fit(z[:,None],y,scale__sample_weight=w,model__sample_weight=w)
    assert np.array_equal(m.classes_,np.arange(k)) and (m['model'].n_iter_<2000).all();return m
def cost(k,target,rule):
    d=abs(np.arange(k)[:,None]-np.arange(k)[None,:])/(k-1);wrong=(d>0).astype(float)
    if rule=='mode':return wrong
    if rule=='absolute':return d
    if rule=='squared':return d*d
    _,lam,a,b=rule.split('-');c=wrong+float(lam)*d*d;lc=low(target,k)
    c[lc,:]+=float(a)*(np.arange(k)!=lc);c[:,lc]+=float(b)*(np.arange(k)!=lc);return c
def decide(p,target,rule):return (p@cost(p.shape[1],target,rule)).argmin(1)
def stats(y,pred,ids,k,target):
    lc=low(target,k);pos=y==lc;neg=~pos
    def mean(v,ii):return float(np.average(v,weights=weights(ii)))
    recalls=[mean(pred[y==j]==j,ids[y==j]) for j in range(k)]
    return {'accuracy':mean(pred==y,ids),'balanced_accuracy':float(np.mean(recalls)),
            'extreme':mean(abs(pred-y)==k-1,ids),'miss':mean(pred[pos]!=lc,ids[pos]),'fpr':mean(pred[neg]==lc,ids[neg]),
            'low_to_best':mean(abs(pred[pos]-y[pos])==k-1,ids[pos]),'recalls':recalls}
def guards(a,b):return all(a[c]<=b[c]+1e-12 for c in ['extreme','miss','fpr'])
def rank(r):
    v=r['metrics'];return (-v['balanced_accuracy'],v['extreme'],v['miss'],v['fpr'],r['candidate'],RULES.index(r['rule']))
def choose(candidates,target,ids):
    rows=[]
    for ci,c in enumerate(candidates):
        if not c['eligible']:continue
        for rule in RULES:rows.append({'candidate':ci,'rule':rule,'metrics':stats(c['y'],decide(c['cross'],target,rule),ids,len(c['cuts'])+1,target)})
    base=next(r for r in rows if r['candidate']==0 and r['rule']=='mode')
    fixed=min([r for r in rows if r['candidate']==0 and guards(r['metrics'],base['metrics'])],key=rank)
    adjusted=min([r for r in rows if len(candidates[r['candidate']]['cuts'])==3 and guards(r['metrics'],fixed['metrics'])],key=rank)
    mi=len(candidates)-1;mb=next(r for r in rows if r['candidate']==mi and r['rule']=='mode')
    merged=min([r for r in rows if r['candidate']==mi and guards(r['metrics'],mb['metrics'])],key=rank)
    return {'fixed4':fixed,'adjusted4':adjusted,'merged3':merged},rows
def candidate(raw,oof,target,cuts,real,ids,inner):
    y=labels(real,cuts);k=len(cuts)+1;cross=np.zeros_like(oof);ts=[]
    for i in range(3):
        a=inner!=i;b=~a;t=old.temperature(oof[a],y[a],ids[a]);cross[b]=old.temper(oof[b],t);ts.append(t)
    t=old.temperature(oof,y,ids)
    supports=[[int(len(np.unique(ids[(inner!=i)&(y==j)]))) for j in range(k)] for i in range(3)]
    shares=[float(np.average(y==j,weights=weights(ids))) for j in range(k)]
    eligible=min(map(min,supports))>=40 and min(shares)>=.03
    return {'cuts':cuts,'y':y,'raw':raw,'oof':oof,'cross':cross,'cal':old.temper(raw,t),'temperature':t,'cross_temperatures':ts,
            'supports':supports,'shares':shares,'eligible':bool(eligible)}

def work(task):
    target,band,n,fold,seed=task;tag=f'{target}-{band}-{n}-f{fold}-s{seed}';done=STORE/'folds'/f'{tag}.json'
    if done.exists():return tag,'cached'
    start=time.time();tr,te,inner,ref=old.cell_data(target,band,n,fold);col='total_sppb_score' if target=='sppb' else 'tugtime'
    x=tr[FEATURES].to_numpy();xt=te[FEATURES].to_numpy();real=tr[col].to_numpy();ids=tr.participantid.to_numpy();yt=te[col].to_numpy()
    finey=labels(real,FINE[target]);kc=len(FINE[target])+1
    cuts=[sorted(ANCHORS[target]+[c]) for c in MIDDLES[target]]+[ANCHORS[target]]
    arrays={'ids':ids,'visits':tr.visit_number.to_numpy(dtype='U'),'real':real,'inner':inner}
    ledger={'task':list(task),'split_sha256':sha(old.SETS/'folds'/f'{target}-{band}-{n}-f{fold}.json'),'families':{}}
    frames=[]
    for family in FAMILIES:
        meta={};cand=[]
        if family=='RFreg':
            source=old.STORE/'regression-five'/f'{tag}-inner.npz';table=old.STORE/'regression-five'/f'{tag}.csv.gz';previous=json.loads((old.STORE/'regression-five'/f'{tag}.json').read_text())
            a=np.load(source);np.testing.assert_array_equal(a['ids'],ids);np.testing.assert_array_equal(a['inner'],inner);z=a['RFreg5_scores'];ztable=pd.read_csv(table)
            np.testing.assert_array_equal(ztable[KEY].to_numpy(),te[KEY].to_numpy());zt=ztable.RFreg5_score.to_numpy();arrays['reg_scores']=z
            meta['source_hashes']={str(p):sha(p) for p in [source,table,Path(previous['models']['RFreg5']['regressor'])]}
        else:
            fo=np.zeros((len(tr),kc))
            for i in range(3):
                a=inner!=i;b=~a;m=fine_fit(family,seed+fold*10+i,x[a],finey[a],ids[a],kc);fo[b]=fine_prob(m,x[b],family,kc)
            m=fine_fit(family,seed+fold*10+9,x,finey,ids,kc);fr=fine_prob(m,xt,family,kc)
            path=STORE/'models'/f'{tag}-{family}.joblib';joblib.dump(m,path,compress=3);meta.update(path=str(path),sha256=sha(path));arrays[family+'_fine_oof']=fo;arrays[family+'_fine_test']=fr
        for ci,cut in enumerate(cuts):
            yy=labels(real,cut);k=len(cut)+1
            if family=='RFreg':
                oo=np.zeros((len(tr),k))
                for i in range(3):
                    a=inner!=i;b=~a;m=map_fit(z[a],yy[a],ids[a],k);oo[b]=m.predict_proba(z[b,None])
                m=map_fit(z,yy,ids,k);rr=m.predict_proba(zt[:,None]);path=STORE/'models'/f'{tag}-RFreg-c{ci}.joblib';joblib.dump(m,path,compress=3)
            else:oo=coarsen(fo,target,cut);rr=coarsen(fr,target,cut)
            c=candidate(rr,oo,target,cut,real,ids,inner)
            if ci in [0,len(cuts)-1]:assert c['eligible'],(task,ci,c['supports'],c['shares'])
            if family=='RFreg':c.update(path=str(path),sha256=sha(path))
            cand.append(c);arrays[f'{family}_c{ci}_oof']=oo;arrays[f'{family}_c{ci}_cross']=c['cross']
        selected,rules=choose(cand,target,ids);meta['selection']=selected;meta['rules']=rules;meta['candidates']=[]
        for c in cand:meta['candidates'].append({key:value for key,value in c.items() if key not in ['y','raw','oof','cross','cal']})
        for scheme,sel in selected.items():
            ci=sel['candidate'];c=cand[ci];k=len(c['cuts'])+1;y=labels(yt,c['cuts']);trainy=c['y'];prior=np.bincount(trainy,weights=weights(ids),minlength=k);prior/=prior.sum()
            thresholds=old.thresholds(c['cross'][:,low(target,k)],(trainy==low(target,k)).astype(int),ids)
            meta['selection'][scheme]['thresholds']=thresholds
            out=te[KEY+['cohort']].copy();out['actual']=yt;out['class']=y;out['target']=target;out['band']=band;out['n']=n;out['fold']=fold;out['seed']=seed
            out['family']=family;out['scheme']=scheme;out['candidate']=ci;out['cuts']=json.dumps(c['cuts']);out['classes']=k;out['rule']=sel['rule']
            out['mode']=decide(c['cal'],target,'mode');out['selected']=decide(c['cal'],target,sel['rule']);out['prior_class']=prior.argmax()
            for j in range(4):out[f'p{j}']=c['cal'][:,j] if j<k else 0.;out[f'prior{j}']=prior[j] if j<k else 0.
            for policy,cut in thresholds.items():out[policy]=(c['cal'][:,low(target,k)]>=cut).astype(int)
            frames.append(out)
        ledger['families'][family]=meta
    table=STORE/'folds'/f'{tag}.csv.gz';arr=STORE/'folds'/f'{tag}-inner.npz';pd.concat(frames,ignore_index=True).to_csv(table,index=False);np.savez_compressed(arr,**arrays)
    ledger.update(predictions_sha256=sha(table),arrays_sha256=sha(arr),seconds=time.time()-start);save(done,ledger);return tag,round(time.time()-start,1)

def prepare():
    p={'code_sha256':sha(__file__),'sources':{str(p):sha(p) for p in [PREVIOUS/'study.py',PREVIOUS/'verification.json',old.BASE/'features.csv',old.BASE/'selected-bouts.csv']},
       'seeds':SEEDS,'fine_cuts':FINE,'middle_candidates_ordered':MIDDLES,'anchors':ANCHORS,
       'people':'same SPPB2174/TUG1939 people, all visits and same outer5/inner3 person IDs, draw0 [4,10)/[30,60) x1/3/5 median four features',
       'models':'RFreg reuses previous five-seed continuous predictions; RF/OrdinalRF train common fine distribution with same128trees leaf20, then sum probabilities for each partition; distinct from previous four-class-trained forests',
       'calibration':'candidate-specific train-only OOF multinomial mapping for RFreg; cross-fit temperature on inner OOF probabilities for all, final T uses outer train only',
       'support':'each class at least40 distinct people in each inner training fold and3percent outer training person-weighted mass; no test-based exclusion',
       'rules':RULES,'selection':'fixed4: maximize balanced recall with extreme/miss/FPR no worse than inner mode; adjusted4: maximize balanced recall jointly over eligible boundaries/rules with same guards versus fixed4 selected; merged3: same guards versus own mode; deterministic tie-break; fallback always available',
       'primary':'long5 fixed before this run; all six conditions; argmax and chosen-rule reported separately; train-prior baseline for each partition',
       'comparison':'extreme endpoints unchanged for all partitions; low miss and FPR fixed target; raw accuracy/Brier across K not claimed directly comparable; expected width of true/predicted middle bands and boundary frequencies reported',
       'limits':'inner reuse for selection not unbiased performance; outer development folds reused across research, not fresh confirmatory validation; no video or clinical grade validation',
       'storage':{'small':str(HERE),'large':str(STORE)}}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==p
    save(HERE/'protocol.json',p);print('PROTOCOL_READY',flush=True)
def run(mode,workers):
    assert json.loads((HERE/'protocol.json').read_text())['code_sha256']==sha(__file__)
    for f in ['models','folds']:(STORE/f).mkdir(parents=True,exist_ok=True)
    old.init();tasks=list(itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5),SEEDS))
    if mode=='pilot':tasks=[('sppb','long',5,0,SEEDS[0]),('tug','long',5,0,SEEDS[0])]
    start=time.time()
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(work,t) for t in tasks]),1):
            tag,sec=f.result()
            if mode=='pilot' or i%10==0:print(i,'/',len(tasks),tag,sec,'elapsed',round(time.time()-start),flush=True)
    save(HERE/f'{mode}-completion.json',{'tasks':len(tasks),'seconds':time.time()-start,'code_sha256':sha(__file__)});print(mode.upper()+'_DONE',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['prepare','pilot','main']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    if args.mode=='prepare':prepare()
    else:run(args.mode,args.workers)
