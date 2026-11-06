"""Separate sensor prediction, camera-observable proxy, and actual camera loops."""
from pathlib import Path
import os,sys,json,hashlib,time,argparse,warnings
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import joblib
from scipy.optimize import minimize
from scipy.special import expit
from scipy.signal import savgol_filter
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold,KFold
from threadpoolctl import threadpool_limits
threadpool_limits(2)
H=Path('D:/Projects/D_P/studies/20261104-tug-camera-dual-loop')
B=Path('D:/gait-research-data/D_P/tug-camera-dual-loop-20261104')
OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop')
OB=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103')
G=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
P=Path('D:/Projects/D_P/studies/20261104-pose-compatible-tug')
M=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop')
MB=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(4*1024**2),b''):h.update(c)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def metric(y,p):
    e=np.abs(y-p);slow=y>=20
    return dict(n=len(y),mae=float(e.mean()),p95=float(np.quantile(e,.95)),bias=float((p-y).mean()),slow_n=int(slow.sum()),slow_mae=float(e[slow].mean()) if slow.any() else None,slow_bias=float((p-y)[slow].mean()) if slow.any() else None)
def freeze():
    H.mkdir(exist_ok=True);B.mkdir(exist_ok=True)
    protocol=dict(owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',created_unix=time.time(),code_sha256=sha(__file__),authorization='User requested renewed improvement loops on original0.881 and phone-camera-available inputs. Camera only, no worn phone. Local CPU; no new school job or upload.',source_hashes={str(p):sha(p) for p in [G/'data.npz',G/'native-motion/data.npz',OLD/'round11/verification.json',OLD/'round12/verification.json',P/'verification.json',P/'protocol.json',M/'first4s-verification.json',M/'clinical-label-audit.json',MB/'bottom14-inputs.npz']},design='Three separately labelled tracks. Sensor: new fixed meta learning/gating, verified source baseOOF reused, no source CNN retraining. Proxy: fit raw permitted sensor-derived quantities with nested outer5/inner3 model choice. Camera: same14people/first qualified4s as previously reviewed, feature extraction outcome-blind, outerleave-one-person-out/innerLOO, fixed candidate recipes. No outer-based recipe selection within any declared policy. Every declared condition reported, observed best labelled exploratory.',success='Compare against original0.8805398998 with pairedperson95percent uncertainty, p95<=3seconds and slow20plusMAE<=3seconds retained from prior development criteria. Camera/proxy improvement evaluated against matched strong speed or camera baselines. No new clinical usability threshold or phone accuracy claim.',limits='Repeatedly reused development cohorts; post-search CIs conditional not multiplicity/selection-adjusted.9slow/594,1slow/14. Proxy sensor-derived features are not actual camera. Camera frontal/back, released clock only, no synchronized sensor supervision or metric calibration. Short clips cannot guarantee enough cycles for variability. Failures and all candidates retained.',sensor_round1=dict(families=['full34','wave_manual11'],methods=['robust_positive_beta.25','robust_positive_beta1','robust_positive_beta3','rbf_C1','rbf_C10','median_boosting']),sensor_round2=dict(families=['full34','wave_manual11'],gate_variants=[dict(beta=b,reg=r,extended=e) for b,r,e in [(.25,.01,False),(.5,.01,False),(1,.001,False),(1,.1,False),(1,.01,True),(.5,.1,True)]],expert_tail=[1,16],aux=.1),proxy_round1=dict(conditions=['speed','core3','timing7','metric12','availability5'],candidates=[dict(kind=k,C=c,transform=t) for k,c,t in [('svr',1,'direct'),('svr',10,'direct'),('svr',100,'direct'),('svr',10,'product'),('svr',10,'residual'),('ridge',10,'direct'),('ridge',10,'product'),('hist',1,'direct')]],policies=['inner_mae_selected','inner_balanced_selected']),camera_round1=dict(conditions=['mechanics','smooth_mechanics','optical_flow','mechanics_flow','quality_only'],candidates=[dict(kind=k,C=c,transform='direct') for k,c in [('ridge',10),('ridge',100),('ridge',1000),('svr',1),('svr',10)]],policies=['inner_mae_selected']),expected_additional_GB=.25)
    if(H/'protocol.json').exists():
        old=read(H/'protocol.json');protocol['created_unix']=old['created_unix'];assert protocol==old
    else:save(H/'protocol.json',protocol)
    return protocol

def positive(A,y,tail=1,beta=None):
    mu=A.mean(0);sd=A.std(0);sd[sd<1e-8]=1;X=(A-mu)/sd;w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));ym=np.average(y,weights=w)
    if beta is None:
        from scipy.optimize import nnls
        xm=np.average(X,axis=0,weights=w);coef,_=nnls(np.vstack([(X-xm)*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(X.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(X.shape[1])],maxiter=10000);inter=ym-xm@coef
    else:
        def objective(v):
            err=X@v[:-1]+v[-1]-y;ab=np.abs(err);loss=np.where(ab<beta,.5*err**2/beta,ab-.5*beta);grad=np.clip(err/beta,-1,1)*w/len(y)
            return np.mean(w*loss)+.01*(v[:-1]@v[:-1]),np.r_[X.T@grad+.02*v[:-1],grad.sum()]
        r=minimize(objective,np.r_[np.zeros(X.shape[1]),ym],jac=True,bounds=[(0,None)]*X.shape[1]+[(None,None)],method='L-BFGS-B',options=dict(maxiter=3000,ftol=1e-12,gtol=1e-8));assert np.isfinite(r.fun);coef=r.x[:-1];inter=r.x[-1]
    return dict(type='linear',mean=mu,std=sd,coef=coef,intercept=inter)
def line(m,X):return ((X-m['mean'])/m['std'])@m['coef']+m['intercept']
def fit_simple(X,y,cfg,speed=None):
    count=np.isfinite(X).sum(0);med=np.array([np.median(a[np.isfinite(a)]) if n else 0. for a,n in zip(X.T,count)]);filled=np.where(np.isfinite(X),X,med);mu=filled.mean(0);sd=filled.std(0);sd[sd<1e-8]=1;Z=(filled-mu)/sd
    t=cfg.get('transform','direct');target=y if t=='direct' else y*speed if t=='product' else y-3/speed
    if cfg['kind']=='svr':model=SVR(C=cfg['C'],epsilon=.15,gamma=1/X.shape[1])
    elif cfg['kind']=='ridge':model=Ridge(alpha=cfg['C'])
    else:model=HistGradientBoostingRegressor(loss='absolute_error',max_leaf_nodes=7,max_iter=150,learning_rate=.04,l2_regularization=10,min_samples_leaf=20,early_stopping=False,random_state=20261104)
    model.fit(Z,target);return dict(type='simple',model=model,median=med,mean=mu,std=sd,config=cfg)
def simple_predict(m,X,speed=None):
    Z=(np.where(np.isfinite(X),X,m['median'])-m['mean'])/m['std'];p=m['model'].predict(Z);t=m['config'].get('transform','direct')
    return p if t=='direct' else p/speed if t=='product' else p+3/speed
def sensor():
    proto=freeze();d=np.load(G/'data.npz');native=np.load(G/'native-motion/data.npz');receipt=read(OLD/'round11/run-receipt.json');out=[]
    for fold in range(5):
        tr=np.flatnonzero(d['fold']!=fold);te=np.flatnonzero(d['fold']==fold);rpath=OB/'round11'/f'combined-{fold}.json';assert sha(rpath)==receipt['hashes'][str(rpath)];r=read(rpath);assert sha(r['context'])==r['context_sha256'];ctx=np.load(r['context']);assert np.array_equal(ctx['train_ids'],d['ids'][tr]) and np.array_equal(ctx['test_ids'],d['ids'][te]) and np.array_equal(ctx['y'],d['y'][tr]);y=d['y'][tr]
        for family in ['full34','wave_manual11']:
            keep=np.ones(34,bool)
            if family=='wave_manual11':keep[list(range(13))+[14,16]+list(range(17,25))]=False
            A=ctx['X'][:,keep];T=ctx['T'][:,keep];base=B/'sensor'/family/f'fold{fold}';base.mkdir(parents=True,exist_ok=True);context=base/'context.npz';np.savez_compressed(context,A=A,T=T,y=y,train_ids=d['ids'][tr],test_ids=d['ids'][te],train_speed=ctx['train_speed'],test_speed=ctx['test_speed'],duration=native['F'][tr,1],test_duration=native['F'][te,1],guard=ctx['guard_flags'],fallback=ctx['fallback'],keep=keep);source=dict(record=str(rpath),record_sha256=sha(rpath),context=r['context'],context_sha256=r['context_sha256'])
            for method in proto['sensor_round1']['methods']:
                dest=base/(method+'.json')
                if dest.exists():out.append(read(dest));continue
                if method.startswith('robust'):m=positive(A,y,beta=float(method.split('beta')[1]));pr=line(m,T)
                else:
                    cfg=dict(kind='svr' if method.startswith('rbf') else 'hist',C=float(method.split('_C')[1]) if method.startswith('rbf') else 1,transform='direct');m=fit_simple(A,y,cfg);pr=simple_predict(m,T)
                path=base/(method+'.joblib');joblib.dump(m,path);rec=dict(track='sensor_round1',condition=family+'/'+method,fold=fold,model=str(path),model_sha256=sha(path),context=str(context),context_sha256=sha(context),source=source,prediction=pr.tolist(),guarded=np.where(ctx['guard_flags'],ctx['fallback'],pr).tolist());save(dest,rec);out.append(rec)
            low=positive(A,y,tail=1);high=positive(A,y,tail=16);L=line(low,A);U=line(high,A);Lt=line(low,T);Ut=line(high,T);cols=np.flatnonzero(keep);j31=int(np.flatnonzero(cols==31)[0]);j15=int(np.flatnonzero(cols==15)[0]);raw=np.column_stack([A[:,j31],A[:,j15],np.log1p(A.std(1)),np.log(ctx['train_speed']),native['F'][tr,1]]);test=np.column_stack([T[:,j31],T[:,j15],np.log1p(T.std(1)),np.log(ctx['test_speed']),native['F'][te,1]])
            for ci,cfg in enumerate(proto['sensor_round2']['gate_variants']):
                method=f'gate{ci}';dest=base/(method+'.json')
                if dest.exists():out.append(read(dest));continue
                P0=np.column_stack([raw,(U-L),raw[:,0]-raw[:,1],raw[:,2]*raw[:,4]]) if cfg['extended'] else raw
                Q0=np.column_stack([test,(Ut-Lt),test[:,0]-test[:,1],test[:,2]*test[:,4]]) if cfg['extended'] else test
                mu=P0.mean(0);sd=P0.std(0);sd[sd<1e-8]=1;Pz=(P0-mu)/sd;Qz=(Q0-mu)/sd;target=(y>=20).astype(float);pw=(len(y)-target.sum())/target.sum();beta=cfg['beta'];reg=cfg['reg']
                def obj(theta):
                    z=Pz@theta[:-1]+theta[-1];g=expit(z);err=L+g*(U-L)-y;h=np.where(abs(err)<beta,.5*err**2/beta,abs(err)-.5*beta);ce=(1-target)*np.logaddexp(0,z)+pw*target*np.logaddexp(0,-z);v=np.mean(h)+.1*np.mean(ce)+reg*np.sum(theta[:-1]**2);dz=(np.clip(err/beta,-1,1)*(U-L)*g*(1-g)+.1*((1-target)*g-pw*target*(1-g)))/len(y);return v,np.r_[Pz.T@dz+2*reg*theta[:-1],dz.sum()]
                initial=np.r_[np.zeros(Pz.shape[1]),-2.];opt=minimize(obj,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8));assert np.isfinite(opt.fun) and opt.fun<=obj(initial)[0]+1e-10;pr=Lt+expit(Qz@opt.x[:-1]+opt.x[-1])*(Ut-Lt);m=dict(type='gate',low=low,high=high,mean=mu,std=sd,theta=opt.x,config=cfg,columns=[j31,j15],objective=float(opt.fun),gradient=obj(opt.x)[1]);path=base/(method+'.joblib');joblib.dump(m,path);rec=dict(track='sensor_round2',condition=family+'/'+method,fold=fold,model=str(path),model_sha256=sha(path),context=str(context),context_sha256=sha(context),source=source,prediction=pr.tolist(),guarded=np.where(ctx['guard_flags'],ctx['fallback'],pr).tolist());save(dest,rec);out.append(rec)
            save(H/'sensor-progress.json',dict(status='running',last=f'{family}/fold{fold}',completed_records=len(out)));print('sensor',family,fold,flush=True)
    save(H/'sensor-receipt.json',dict(status='complete_pending_review',records=out,protocol_sha256=sha(H/'protocol.json')))

def nested(track,sets,y,ids,folds,speed,grid,balanced=True):
    records=[];root=B/track;root.mkdir(exist_ok=True)
    for condition,X in sets.items():
        source=root/(condition+'-data.npz');np.savez_compressed(source,X=X,y=y,ids=ids,fold=folds,speed=np.zeros(len(y)) if speed is None else speed)
        for fold in sorted(set(folds)):
            dest=root/f'{condition}-{fold}.json'
            if dest.exists():records.append(read(dest));continue
            tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);splits=list(StratifiedKFold(3,shuffle=True,random_state=20261104).split(tr,np.digitize(y[tr],[12,20]))) if len(y)>30 else [(np.flatnonzero(np.arange(len(tr))!=j),np.array([j])) for j in range(len(tr))];A=np.full((len(tr),len(grid)),np.nan);T=np.full((len(te),len(grid)),np.nan);models=[]
            for ci,cfg in enumerate(grid):
                for k,(it,iv) in enumerate(splits):
                    m=fit_simple(X[tr[it]],y[tr[it]],cfg,None if speed is None else speed[tr[it]]);p=simple_predict(m,X[tr[iv]],None if speed is None else speed[tr[iv]]);A[iv,ci]=p;path=root/f'{condition}-{fold}-c{ci}-i{k}.joblib';joblib.dump(m,path);models.append(dict(candidate=ci,inner=k,path=str(path),sha256=sha(path),train=tr[it].tolist(),eval=tr[iv].tolist(),prediction=p.tolist()))
                m=fit_simple(X[tr],y[tr],cfg,None if speed is None else speed[tr]);p=simple_predict(m,X[te],None if speed is None else speed[te]);T[:,ci]=p;path=root/f'{condition}-{fold}-c{ci}-final.joblib';joblib.dump(m,path);models.append(dict(candidate=ci,inner=None,path=str(path),sha256=sha(path),train=tr.tolist(),eval=te.tolist(),prediction=p.tolist()))
            assert np.isfinite(A).all() and np.isfinite(T).all();scores=np.mean(abs(A-y[tr,None]),axis=0);selected=int(np.argmin(scores));policy={'inner_mae_selected':dict(candidate=selected,prediction=T[:,selected].tolist())}
            if balanced:
                bs=np.array([max(metric(y[tr],p)['mae']/.9,metric(y[tr],p)['p95']/3,metric(y[tr],p)['slow_mae']/3) for p in A.T]);sel=int(np.argmin(bs));policy['inner_balanced_selected']=dict(candidate=sel,prediction=T[:,sel].tolist())
            ctx=root/f'{condition}-{fold}-context.npz';np.savez_compressed(ctx,A=A,T=T,train=tr,test=te);r=dict(track=track,condition=condition,fold=int(fold),data=str(source),data_sha256=sha(source),context=str(ctx),context_sha256=sha(ctx),inner_scores=scores.tolist(),models=models,policies=policy);save(dest,r);records.append(r);save(H/(track+'-progress.json'),dict(status='running',last=f'{condition}/fold{fold}',completed=len(records)));print(track,condition,fold,flush=True)
    save(H/(track+'-receipt.json'),dict(status='complete_pending_review',records=records,protocol_sha256=sha(H/'protocol.json')))

def proxy():
    proto=freeze();d=np.load(G/'data.npz');names=read(Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn/protocol.json'))['feature_names'];sp=read(P/'protocol.json')['feature_sets'];features=dict(speed=['speed'],core3=['speed','stride_time','swing_fraction'],timing7=sp['timing'],metric12=sp['calibrated_spatiotemporal']);sets={c:d['F'][:,[names.index(n) for n in ns]] for c,ns in features.items()};sets['availability5']=np.column_stack([sets['core3'],np.isfinite(sets['timing7'][:,3]),np.isfinite(sets['timing7'][:,4])]).astype(float);assert np.all(d['F'][:,0]>0);save(H/'proxy-feature-spec.json',dict(feature_names=features,availability5='core3 plus source detector bilateral and repeated-cycle availability only. Diagnostic not phone health predictor.',limits='Sensor-derived potentially camera-computable values. No raw IMU signals or teacher CNN outputs. Medianimputation may implicitly reveal availability in7/12 arms. No camera performance claim.'));nested('proxy_round1',sets,d['y'],d['ids'],d['fold'],d['F'][:,0],proto['proxy_round1']['candidates'])

def segments(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
def stats(v,ok):
    good=v[ok]
    if len(good)<3:return [np.nan]*9
    pair=ok[1:]&ok[:-1];vel=np.diff(v)*30;lag=np.corrcoef(v[:-1][pair],v[1:][pair])[0,1] if pair.sum()>3 and np.std(v[:-1][pair])>1e-8 and np.std(v[1:][pair])>1e-8 else 0.
    longest=max(segments(ok),key=lambda ab:ab[1]-ab[0]);z=v[longest[0]:longest[1]];freq=power=np.nan
    if len(z)>=45:
        f=np.fft.rfftfreq(len(z),1/30);s=np.abs(np.fft.rfft((z-z.mean())*np.hanning(len(z))))**2;band=(f>=.5)&(f<=4)
        if np.any(band):j=np.flatnonzero(band)[np.argmax(s[band])];freq=f[j];power=s[band].sum()/max(s.sum(),1e-12)
    return [float(np.mean(good)),float(np.std(good)),float(np.quantile(good,.25)),float(np.quantile(good,.75)),float(np.median(abs(vel[pair]))) if pair.any() else np.nan,float(np.sqrt(np.mean(vel[pair]**2))) if pair.any() else np.nan,float(lag),float(freq),float(power)]
def mechanics(x,m,smooth=False):
    x=x.reshape(120,12,2).astype(float).copy()
    if smooth:
        for j in range(12):
            for lo,hi in segments(m[:,j]):
                if hi-lo>=5:x[lo:hi,j]=savgol_filter(x[lo:hi,j],5,2,axis=0)
    values=[];masks=[]
    for j in range(12):
        for a in range(2):values.append(x[:,j,a]);masks.append(m[:,j])
    for a,b,c in [(2,4,6),(3,5,7)]:
        u=x[:,a]-x[:,b];v=x[:,c]-x[:,b];cos=np.sum(u*v,axis=1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-8);values.append(np.arccos(np.clip(cos,-1,1)));masks.append(m[:,a]&m[:,b]&m[:,c])
    trunk=(x[:,0]+x[:,1]-x[:,2]-x[:,3])/2;values.append(np.arctan2(trunk[:,0],-trunk[:,1]));masks.append(np.all(m[:,:4],1));values.append(np.linalg.norm(x[:,6]-x[:,7],axis=1));masks.append(m[:,6]&m[:,7]);values.append(np.linalg.norm(x[:,10]-x[:,11],axis=1));masks.append(m[:,10]&m[:,11]);f=[]
    for v,ok in zip(values,masks):f.extend(stats(v,ok))
    for a,b in [(6,7),(8,9),(10,11)]:
        ok=m[:,a]&m[:,b]
        for k in range(2):f.append(float(np.corrcoef(x[ok,a,k],x[ok,b,k])[0,1]) if ok.sum()>5 and min(np.std(x[ok,a,k]),np.std(x[ok,b,k]))>1e-8 else np.nan)
    return np.array(f)
def optical_flow(ids,indices,data):
    sys.path.insert(0,'D:/Projects/D_P/gdi-research-harness/control/manual-sessions/20261028-camera-step-pilot/deps');import cv2
    cv2.setNumThreads(2);batch={r['person_id']:r for r in read(M/'batch14-receipt.json')['receipts']};manifest={r['person_id']:r for r in read(M/'all-bottom-videos-manifest.json')['rows']};meta={r['person_id']:r for r in read(M/'camera14-metadata.json')['records']};result=[];proof=[]
    for pid,ix in zip(ids,indices):
        dest=B/'camera-features'/(str(pid)+'-flow.npz');dest.parent.mkdir(exist_ok=True);r=batch[str(pid)];src=manifest[str(pid)];assert sha(r['output'])==r['output_sha256'] and sha(src['file'])==src['sha256']
        if dest.exists():a=np.load(dest);result.append(a['features']);proof.append(dict(id=str(pid),path=str(dest),sha256=sha(dest)));continue
        z=np.load(r['output']);start=data['window_start_ms'][ix];ii=np.flatnonzero((z['source_ms']>=start-1e-6)&(z['source_ms']<start+4000-1e-6));assert len(ii)>100 and np.all(np.diff(ii)==1);cap=cv2.VideoCapture(src['file']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(ii[0]));W=meta[str(pid)]['width'];Ht=meta[str(pid)]['height'];factor=384/Ht;ww=round(W*factor);signals=[];valid=[];prev=None;previous=None
        for i in ii:
            ok,frame=cap.read();assert ok;gray=cv2.cvtColor(cv2.resize(frame,(ww,384)),cv2.COLOR_BGR2GRAY)
            if prev is not None:
                dt=(z['source_ms'][i]-z['source_ms'][previous])/1000;flow=cv2.calcOpticalFlowFarneback(prev,gray,None,.5,3,15,3,5,1.2,0);outside=np.ones((384,ww),bool);p=z['landmarks'][previous,:,:2]*[ww,384];good=z['valid'][previous]&z['valid'][i];vals=[];vm=[];radius=max(3,round(float(data['scale_pixels'][ix])*factor*.08))
                for js in [[27,29,31],[28,30,32],[11,12,23,24]]:
                    selected=[j for j in js if good[j]];region=np.zeros((384,ww),np.uint8)
                    for j in selected:cv2.circle(region,(int(round(p[j,0])),int(round(p[j,1]))),radius,1,-1)
                    outside[region>0]=False;v=np.median(flow[region>0],axis=0) if region.any() else np.zeros(2);vals.extend(v);vm.extend([bool(region.any()) and z['provider'][i]==z['provider'][previous] and z['provider'][i]>=0]*2)
                # Camera jitter correction from outer image margins only, not other actors.
                bg=np.zeros((384,ww),bool);bg[:,:max(1,ww//10)]=True;bg[:,-max(1,ww//10):]=True;offset=np.median(flow[bg],axis=0);v=(np.array(vals).reshape(3,2)-offset)/(float(data['scale_pixels'][ix])*factor*dt);signals.append(v.reshape(-1));valid.append(vm)
            prev=gray;previous=i
        cap.release();sig=np.array(signals);mask=np.array(valid);features=[]
        for j in range(6):
            v=sig[:,j][mask[:,j]];features.extend([float(np.median(v)),float(np.std(v)),float(np.quantile(abs(v),.9))] if len(v)>=3 else [np.nan]*3)
        # Preserve local image-motion features, not acceleration or verified joint velocity.
        np.savez_compressed(dest,features=np.array(features),signals=sig,mask=mask,source_frames=ii,start_ms=start,source_video_sha256=src['sha256'],source_pose_sha256=r['output_sha256']);result.append(features);proof.append(dict(id=str(pid),path=str(dest),sha256=sha(dest)));print('flow',pid,flush=True)
    save(H/'optical-flow-receipt.json',dict(status='extracted_not_anatomically_validated',records=proof,meaning='Dense image motion near visible foot/trunk landmarks, normalized by observed leg size and source frame interval. Fixed384pxheight. No metre/IMU acceleration claim. Samefirst4s, no total duration/absolute start as model feature.'));return np.array(result)
def camera():
    proto=freeze();d=np.load(MB/'bottom14-inputs.npz');ids=np.array(sorted(set(d['person_ids'])));ix=np.array([np.flatnonzero(d['person_ids']==pid)[0] for pid in ids]);labels={r['person_id']:r['tug'] for r in read(M/'clinical-label-audit.json')['labels']};y=np.array([labels[pid] for pid in ids]);assert read(M/'first-windows-visual-review.json')['status']=='sampled_first_windows_reviewed';sets={};sets['mechanics']=np.array([mechanics(d['pelvis_stable_xy'][i],d['pelvis_mask'][i]) for i in ix]);sets['smooth_mechanics']=np.array([mechanics(d['pelvis_stable_xy'][i],d['pelvis_mask'][i],True) for i in ix]);sets['optical_flow']=optical_flow(ids,ix,d);sets['mechanics_flow']=np.column_stack([sets['smooth_mechanics'],sets['optical_flow']]);sets['quality_only']=d['pelvis_mask'][ix].mean(1);save(H/'camera-feature-spec.json',dict(first_window_indices=ix.tolist(),people=ids.tolist(),shapes={c:list(x.shape) for c,x in sets.items()},definitions='Mechanics:29projected2Dsignals each9statistics plus6left/rightcorrelations.24pelviscenteredscaledjointXY,knee2Dangles2,trunkprojectedlean1,ankles/toesdistance2. Statistics mean/std/q25/q75/medianabsvelocity/RMSvelocity/lag1corr/dominant.5..4Hzfrequency/frequencybandpower. Spectrum onlylongestvalidsegment>=45frames. Smooth version5frameSavitzkyGolay withincontiguousvalidruns only. Missing remainsNaN fortrain-onlyimputation. Flow18localimagevelocitystatistics. Quality12jointvalidfractions separate diagnostic.',limits='Nottrue3Dangles/metrictravel/IMUacceleration. First4sreleasedclock, frontal/back view, originalclockunverified. All14oneoutcomeeach; repeatedwindowsnotindependent. Qualityonlydiagnosticmustnotbeadoptedashealthscore. No futureframesoutside4s. No interpolation across pose failures.'))
    nested('camera_round1',sets,y,ids,np.arange(len(y)),None,proto['camera_round1']['candidates'],balanced=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','sensor','proxy','camera']);a=ap.parse_args();{'freeze':freeze,'sensor':sensor,'proxy':proxy,'camera':camera}[a.mode]()
