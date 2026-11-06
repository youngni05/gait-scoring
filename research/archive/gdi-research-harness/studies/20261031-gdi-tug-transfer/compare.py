"""Approved A/B/C/D comparisons; nested participant selection, CPU only."""
from pathlib import Path
import argparse,json,time
import joblib,numpy as np,torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from features import OUT,GdiCNN,channels,sha
HERE=Path(__file__).resolve().parent

def save(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)
def load():
    audit=json.loads((OUT/'input-verification.json').read_text())
    assert sha(OUT/'prepared.npz')==audit['prepared_sha256']
    assert sha(HERE/'protocol.json')==audit['protocol_sha256']
    d=dict(np.load(OUT/'prepared.npz',allow_pickle=False));assert len(set(d['ids']))==len(d['ids'])
    assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
    return d
def splits(d):
    result=[]
    for fold in range(5):
        tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold)
        assert len(te) and len(tr)>=3
        # Sort person strings exactly as in the prior participant protocol.
        tr=tr[np.argsort(d['ids'][tr])]
        inner=[(tr[a],tr[b]) for a,b in KFold(3,shuffle=True,random_state=20261031).split(tr)]
        result.append((fold,tr,te,inner))
    return result
def scalar():
    d=load();base=d['base'];y=d['y'];n=len(y);folder=OUT/'scalar';folder.mkdir(exist_ok=True)
    inputs={'A_basic_Ridge':base,'A_basic_RF':base,
            'B_GDI_score_Ridge':np.c_[base,d['gdi_score']],
            'C_GDI_features_Ridge':np.c_[base,d['gdi_features']],
            'E_random_features_Ridge':np.c_[base,d['random_features']]}
    pred={name:np.full(n,np.nan) for name in ['Median']+list(inputs)};selected=[]
    for fold,tr,te,inner in splits(d):
        pred['Median'][te]=np.median(y[tr])
        for name,x in inputs.items():
            grid=[2,5] if name.endswith('_RF') else [1.,100.]
            def model(v):return make_pipeline(StandardScaler(),RandomForestRegressor(n_estimators=128,min_samples_leaf=v,random_state=20261031,n_jobs=2) if name.endswith('_RF') else Ridge(alpha=v))
            losses=[]
            for setting in grid:
                iv=np.full(n,np.nan)
                for a,b in inner:iv[b]=model(setting).fit(x[a],y[a]).predict(x[b])
                losses.append(float(np.abs(iv[tr]-y[tr]).mean()))
            setting=grid[int(np.argmin(losses))];m=model(setting).fit(x[tr],y[tr]);pred[name][te]=m.predict(x[te])
            p=folder/('%s-fold%d.joblib'%(name,fold));joblib.dump(m,p)
            replay=joblib.load(p).predict(x[te]);np.testing.assert_allclose(replay,pred[name][te],rtol=0,atol=1e-5)
            selected.append({'fold':fold,'name':name,'train_people':d['ids'][tr].tolist(),'test_people':d['ids'][te].tolist(),
                             'setting':setting,'inner_MAE':losses,'grid':grid,'model_sha256':sha(p)})
        print('scalar fold',fold,'complete',flush=True)
    assert all(np.isfinite(v).all() for v in pred.values())
    np.savez_compressed(folder/'predictions.npz',ids=d['ids'],y=y,folds=d['folds'],**pred)
    save(folder/'selection.json',selected)
    save(folder/'complete.json',{'input_sha256':sha(OUT/'prepared.npz'),'predictions_sha256':sha(folder/'predictions.npz'),'model_count':25,'people':n})

def fit_nn(d,tr,validation,epochs,seed,path):
    torch.manual_seed(seed)
    mu=d['x'][tr].mean((0,1,2),keepdims=True);sd=np.maximum(d['x'][tr].std((0,1,2),keepdims=True),.001)
    ym=float(d['y'][tr].mean());ys=max(float(d['y'][tr].std()),.1)
    x=torch.tensor((d['x']-mu)/sd,dtype=torch.float32);y=torch.tensor((d['y']-ym)/ys,dtype=torch.float32)
    m=GdiCNN(.5);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
    def forward(ix):return m(channels(x[ix]).reshape(-1,12,124))[:,0].reshape(-1,2).mean(1)
    best=float('inf');best_epoch=epochs;state=None;history=[]
    for epoch in range(1,epochs+1):
        m.train();opt.zero_grad(set_to_none=True);loss=((forward(tr)-y[tr])**2).mean();assert torch.isfinite(loss)
        loss.backward();opt.step();item={'epoch':epoch,'loss':float(loss.detach())}
        if validation is not None:
            m.eval()
            with torch.inference_mode():p=forward(validation).numpy()*ys+ym
            score=float(np.abs(p-d['y'][validation]).mean());item['validation_MAE']=score
            if score<best:best=score;best_epoch=epoch;state={k:v.detach().clone() for k,v in m.state_dict().items()}
            if epoch-best_epoch>=12:history.append(item);break
        history.append(item)
    if state is None:state={k:v.detach().clone() for k,v in m.state_dict().items()}
    path.mkdir(parents=True,exist_ok=True)
    torch.save({'state':state,'mu':torch.tensor(mu),'sd':torch.tensor(sd),'ymean':ym,'ystd':ys,'epoch':best_epoch,'seed':seed,
                'train_people':d['ids'][tr].tolist(),'validation_people':[] if validation is None else d['ids'][validation].tolist(),
                'input_sha256':sha(OUT/'prepared.npz')},path/'model.pt')
    save(path/'history.json',history)
    return best_epoch

def scratch():
    d=load();folder=OUT/'scratch';folder.mkdir(exist_ok=True);pred={};selections=[]
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    for seed in range(20261031,20261036):
        values=np.full(len(d['y']),np.nan)
        for fold,tr,te,inner in splits(d):
            base=folder/('s%d-fold%d'%(seed,fold));marker=base/'complete.json'
            if marker.exists():
                old=json.loads(marker.read_text());assert old['input_sha256']==sha(OUT/'prepared.npz')
                values[te]=old['predictions'];selections.append(old);continue
            epochs=[fit_nn(d,a,b,80,seed+fold*10+i,base/('inner%d'%i)) for i,(a,b) in enumerate(inner)]
            chosen=max(1,int(np.median(epochs)));fit_nn(d,tr,None,chosen,seed+fold*10+9,base/'refit')
            ck=torch.load(base/'refit/model.pt',weights_only=True);m=GdiCNN(.5).eval();m.load_state_dict(ck['state'])
            x=(torch.tensor(d['x'][te])-ck['mu'])/ck['sd']
            with torch.inference_mode():p=m(channels(x).reshape(-1,12,124))[:,0].reshape(-1,2).mean(1).numpy()*ck['ystd']+ck['ymean']
            assert np.isfinite(p).all();values[te]=p
            row={'seed':seed,'fold':fold,'inner_epochs':epochs,'epoch':chosen,'predictions':p.tolist(),'test_people':d['ids'][te].tolist(),
                 'input_sha256':sha(OUT/'prepared.npz'),'checkpoint_sha256':sha(base/'refit/model.pt')}
            save(marker,row);selections.append(row);print('scratch',seed,fold,chosen,flush=True)
        pred['seed%d'%seed]=values
    pred['D_scratch_ensemble']=np.stack(list(pred.values())).mean(0)
    np.savez_compressed(folder/'predictions.npz',ids=d['ids'],y=d['y'],folds=d['folds'],**pred)
    save(folder/'selection.json',selections)
    save(folder/'complete.json',{'input_sha256':sha(OUT/'prepared.npz'),'refits':25,'inner_fits':75,'people':len(d['y'])})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['scalar','scratch']);a=ap.parse_args()
    with threadpool_limits(limits=2):scalar() if a.mode=='scalar' else scratch()
