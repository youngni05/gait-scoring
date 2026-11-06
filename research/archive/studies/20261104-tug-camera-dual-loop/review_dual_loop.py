"""Independent replay of all fits, source/person boundaries and held-out metrics."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np,joblib
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
threadpool_limits(2)
H=Path('D:/Projects/D_P/studies/20261104-tug-camera-dual-loop');B=Path('D:/gait-research-data/D_P/tug-camera-dual-loop-20261104');OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');OB=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');M=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');MB=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(4*1024**2),b''):h.update(c)
    return h.hexdigest()
def save(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def close(a,b):assert np.allclose(a,b,atol=1e-6,rtol=1e-6),float(np.max(abs(np.asarray(a)-b)))
def lin(m,X):return ((X-m['mean'])/m['std'])@m['coef']+m['intercept']
def predict(m,X,speed=None):
    filled=np.where(np.isfinite(X),X,m['median']);Z=(filled-m['mean'])/m['std'];net=m['model'];name=net.__class__.__name__
    if name=='Ridge':p=Z@net.coef_+net.intercept_
    elif name=='SVR':
        sv=net.support_vectors_;dist=np.maximum((Z*Z).sum(1)[:,None]+(sv*sv).sum(1)[None,:]-2*Z@sv.T,0);p=np.exp(-net._gamma*dist)@net.dual_coef_[0]+net.intercept_[0]
    else:p=net.predict(Z)
    t=m['config']['transform'];return p if t=='direct' else p/speed if t=='product' else p+3/speed
def prepcheck(m,X):
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in X.T]);assert np.array_equal(med,m['median']);F=np.where(np.isfinite(X),X,med);mu=np.mean(F,axis=0);sd=np.std(F,axis=0);sd[sd<1e-8]=1;assert np.array_equal(mu,m['mean']) and np.array_equal(sd,m['std'])
def metrics(y,p):
    e=abs(y-p);slow=y>=20;bands={}
    for label,ok in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',slow)]:
        bands[label]=dict(n=int(ok.sum()),mae=float(e[ok].mean()),bias=float((p-y)[ok].mean()),max_error=float(e[ok].max())) if ok.any() else dict(n=0)
    return dict(n=len(y),mae=float(e.sum()/len(y)),p95=float(np.percentile(e,95)),bias=float(np.mean(p-y)),slow_n=int(slow.sum()),slow_mae=float(e[slow].mean()) if slow.any() else None,bands=bands)
def sensor(proto):
    r=read(H/'sensor-receipt.json');assert r['protocol_sha256']==sha(H/'protocol.json');data=np.load(G/'data.npz');native=np.load(G/'native-motion/data.npz');out={};proof=[];count=0
    for rec in r['records']:
        assert sha(rec['model'])==rec['model_sha256'] and sha(rec['context'])==rec['context_sha256'];src=rec['source'];assert sha(src['record'])==src['record_sha256'] and sha(src['context'])==src['context_sha256'];old=np.load(src['context']);d=np.load(rec['context']);fold=rec['fold'];tr=np.flatnonzero(data['fold']!=fold);te=np.flatnonzero(data['fold']==fold);assert np.array_equal(d['train_ids'],data['ids'][tr]) and np.array_equal(d['test_ids'],data['ids'][te]) and not set(d['train_ids'])&set(d['test_ids']);assert np.array_equal(d['y'],data['y'][tr]);keep=np.ones(34,bool)
        if rec['condition'].startswith('wave_manual11'):keep[list(range(13))+[14,16]+list(range(17,25))]=False
        assert np.array_equal(keep,d['keep']) and np.array_equal(d['A'],old['X'][:,keep]) and np.array_equal(d['T'],old['T'][:,keep]);assert np.array_equal(d['guard'],old['guard_flags']) and np.array_equal(d['fallback'],old['fallback']) and np.array_equal(d['duration'],native['F'][tr,1]);A=d['A'];T=d['T'];y=d['y'];m=joblib.load(rec['model']);delta=None
        if m['type']=='linear':
            assert np.array_equal(m['mean'],A.mean(0));sd=A.std(0);sd[sd<1e-8]=1;assert np.array_equal(m['std'],sd);beta=float(rec['condition'].split('beta')[1]);Z=(A-m['mean'])/sd;err=lin(m,A)-y;g=Z.T@np.clip(err/beta,-1,1)/len(y)+.02*m['coef'];active=m['coef']>1e-7;violation=max(float(abs(g[active]).max()) if active.any() else 0.,float(np.maximum(-g[~active],0).max()) if (~active).any() else 0.,float(abs(np.clip(err/beta,-1,1).mean())));assert violation<1e-4,violation;delta=violation;p=lin(m,T)
        elif m['type']=='simple':prepcheck(m,A);p=predict(m,T)
        else:
            low=m['low'];high=m['high']
            for expert,tail in [(low,1),(high,16)]:
                assert np.array_equal(expert['mean'],A.mean(0));sd=A.std(0);sd[sd<1e-8]=1;assert np.array_equal(expert['std'],sd);Z=(A-expert['mean'])/sd;w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));grad=Z.T@(w*(lin(expert,A)-y))+10*expert['coef'];active=expert['coef']>1e-7;assert (abs(grad[active]).max() if active.any() else 0)<1e-6 and (np.maximum(-grad[~active],0).max() if (~active).any() else 0)<1e-6;assert abs(np.sum(w*(lin(expert,A)-y)))<1e-7
            L=lin(low,A);U=lin(high,A);Lt=lin(low,T);Ut=lin(high,T);j31,j15=m['columns'];expected_cols=[int(np.flatnonzero(np.flatnonzero(keep)==j)[0]) for j in [31,15]];assert m['columns']==expected_cols
            def raw(X,sp,dt):return np.stack([X[:,j31],X[:,j15],np.log1p(np.std(X,axis=1)),np.log(sp),dt],axis=1)
            P0=raw(A,d['train_speed'],d['duration']);Q0=raw(T,d['test_speed'],d['test_duration']);cfg=m['config']
            if cfg['extended']:P0=np.column_stack([P0,U-L,P0[:,0]-P0[:,1],P0[:,2]*P0[:,4]]);Q0=np.column_stack([Q0,Ut-Lt,Q0[:,0]-Q0[:,1],Q0[:,2]*Q0[:,4]])
            assert np.array_equal(P0.mean(0),m['mean']);sd=P0.std(0);sd[sd<1e-8]=1;assert np.array_equal(sd,m['std']);Pz=(P0-m['mean'])/sd;Qz=(Q0-m['mean'])/sd;theta=m['theta'];z=Pz@theta[:-1]+theta[-1];gate=expit(z);res=L+gate*(U-L)-y;beta=cfg['beta'];reg=cfg['reg'];target=(y>=20).astype(float);pw=(len(y)-target.sum())/target.sum();obj=np.mean(np.where(abs(res)<beta,.5*res**2/beta,abs(res)-beta/2))+.1*np.mean((1-target)*np.logaddexp(0,z)+pw*target*np.logaddexp(0,-z))+reg*np.sum(theta[:-1]**2);dz=(np.clip(res/beta,-1,1)*(U-L)*gate*(1-gate)+.1*((1-target)*gate-pw*target*(1-gate)))/len(y);grad=np.r_[Pz.T@dz+2*reg*theta[:-1],dz.sum()];assert abs(obj-m['objective'])<1e-10 and np.allclose(grad,m['gradient'],atol=1e-8,rtol=1e-8);p=Lt+expit(Qz@theta[:-1]+theta[-1])*(Ut-Lt)
        close(p,rec['prediction']);guarded=np.where(d['guard'],d['fallback'],p);close(guarded,rec['guarded']);count+=1
        for policy,pr in [('plain',p),('guarded',guarded)]:key=rec['condition']+'/'+policy;out.setdefault(key,np.full(len(data['y']),np.nan))[te]=pr
        proof.append(dict(condition=rec['condition'],fold=fold,source_and_people=True,replay=True,KKT_violation=delta))
    baseline=np.full(len(data['y']),np.nan);source_receipt=read(OLD/'round12/run-receipt.json')
    for fold in range(5):
        path=OB/'round12'/f'combined-{fold}.json';assert sha(path)==source_receipt['hashes'][str(path)];rec=read(path);base=next(x for x in rec['models'] if x['index']==1);baseline[data['fold']==fold]=base['guarded_prediction']
    assert abs(abs(baseline-data['y']).mean()-.8805398997851134)<1e-12
    return data['ids'],data['y'],out,{'original_0.881':baseline},dict(models=count,checks=proof)

def nested(track,proto):
    r=read(H/(track+'-receipt.json'));assert r['protocol_sha256']==sha(H/'protocol.json');out={};proof=[];count=0;source=None
    for rec in r['records']:
        assert sha(rec['data'])==rec['data_sha256'] and sha(rec['context'])==rec['context_sha256'];d=np.load(rec['data']);ctx=np.load(rec['context']);X=d['X'];y=d['y'];ids=d['ids'];fold=d['fold'];sp=d['speed'];tr=np.flatnonzero(fold!=rec['fold']);te=np.flatnonzero(fold==rec['fold']);assert np.array_equal(ctx['train'],tr) and np.array_equal(ctx['test'],te) and not set(ids[tr])&set(ids[te]);source=d
        if track=='proxy_round1':
            raw=np.load(G/'data.npz');assert np.array_equal(raw['ids'],ids) and np.array_equal(raw['y'],y) and np.array_equal(raw['fold'],fold) and np.array_equal(raw['F'][:,0],sp);names=read('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn/protocol.json')['feature_names'];spec=read(H/'proxy-feature-spec.json')['feature_names'];condition=rec['condition']
            if condition=='availability5':
                core=raw['F'][:,[names.index(n) for n in spec['core3']]];tim=raw['F'][:,[names.index(n) for n in spec['timing7']]];expected=np.column_stack([core,np.isfinite(tim[:,3]),np.isfinite(tim[:,4])])
            else:expected=raw['F'][:,[names.index(n) for n in spec[condition]]]
            assert np.array_equal(X,expected,equal_nan=True)
        else:
            lab={x['person_id']:x['tug'] for x in read(M/'clinical-label-audit.json')['labels']};assert np.array_equal(y,[lab[i] for i in ids]);assert np.array_equal(fold,np.arange(14))
        grid=proto[track]['candidates'];A=np.full(ctx['A'].shape,np.nan);T=np.full(ctx['T'].shape,np.nan);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261104).split(tr,np.digitize(y[tr],[12,20]))) if len(y)>30 else [(np.flatnonzero(np.arange(len(tr))!=j),np.array([j])) for j in range(len(tr))]
        for row in rec['models']:
            assert sha(row['path'])==row['sha256'];m=joblib.load(row['path']);ci=row['candidate'];assert m['config']==grid[ci]
            if row['inner'] is None:it=tr;iv=te
            else:a,b=splits[row['inner']];it=tr[a];iv=tr[b]
            assert row['train']==it.tolist() and row['eval']==iv.tolist() and not set(ids[it])&set(ids[iv]) and not set(it)&set(te);prepcheck(m,X[it]);p=predict(m,X[iv],sp[iv]);close(p,row['prediction']);count+=1
            if row['inner'] is None:T[:,ci]=p
            else:A[b,ci]=p
        close(A,ctx['A']);close(T,ctx['T']);scores=np.mean(abs(A-y[tr,None]),axis=0);close(scores,rec['inner_scores']);selected={'inner_mae_selected':int(np.argmin(scores))}
        if 'inner_balanced_selected' in rec['policies']:
            bs=[]
            for p in A.T:v=metrics(y[tr],p);bs.append(max(v['mae']/.9,v['p95']/3,v['slow_mae']/3))
            selected['inner_balanced_selected']=int(np.argmin(bs))
        for policy,row in rec['policies'].items():
            assert selected[policy]==row['candidate'];p=T[:,row['candidate']];close(p,row['prediction']);key=rec['condition']+'/'+policy;out.setdefault(key,np.full(len(y),np.nan))[te]=p
        proof.append(dict(condition=rec['condition'],fold=rec['fold'],all_train_only_scaling_and_person_selection=True,models_replayed=len(rec['models'])))
    y=source['y'];ids=source['ids'];fold=source['fold'];baseline={}
    if track=='proxy_round1':
        for k in ['inner_mae_selected','inner_balanced_selected']:baseline['matched_speed/'+k]=out['speed/'+k]
    else:
        for label,fun in [('train_mean',np.mean),('train_median',np.median)]:baseline[label]=np.array([fun(y[fold!=f]) for f in fold])
        prev=read(MB/'first4s-heldout.json');baseline['previous_image_xy_ridge']=np.array([next(x['prediction'] for x in prev if x['id']==i and x['condition']=='image_xy/ridge') for i in ids])
    return ids,y,out,baseline,dict(models=count,checks=proof)

def main():
    proto=read(H/'protocol.json');assert proto['code_sha256']==sha(H/'dual_loop.py')
    for p,s in proto['source_hashes'].items():assert sha(p)==s
    result={};checks={};heavy={};rng=np.random.default_rng(20261109)
    for track in ['sensor','proxy_round1','camera_round1']:
        ids,y,outputs,baselines,audit=sensor(proto) if track=='sensor' else nested(track,proto);checks[track]=audit;boot=rng.integers(0,len(y),(5000,len(y)));allp={**{'baseline/'+k:v for k,v in baselines.items()},**outputs};summary={};held={}
        for name,p in allp.items():
            assert np.isfinite(p).all();m=metrics(y,p);e=abs(y-p);m['mae_ci95']=np.quantile(e[boot].mean(1),[.025,.975]).tolist();m['comparisons']={}
            for key,reference in baselines.items():
                diff=e-abs(reference-y);m['comparisons'][key]=dict(mae_difference=float(diff.mean()),ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
            summary[name]=m;held[name]=p.tolist()
        result[track]=summary;heavy[track]=dict(ids=ids.tolist(),y=y.tolist(),predictions=held);print(track,'reviewed models',audit['models'],flush=True)
    save(B/'heldout.json',heavy);save(H/'comparison.json',result);save(H/'verification.json',dict(status='passed',protocol_sha256=sha(H/'protocol.json'),review_code_sha256=sha(__file__),checks=checks,comparison_sha256=sha(H/'comparison.json'),heldout_sha256=sha(B/'heldout.json'),uncertainty='5000pairedpersonbootstrap pertrack. Conditionaldevelopmentuncertainty,notsearch/multiplicityadjusted. Sensor9slow,camera1slow. Allconditionsreported,notindependentexternaltest.'));print('allthree tracks passed',flush=True)
if __name__=='__main__':main()
