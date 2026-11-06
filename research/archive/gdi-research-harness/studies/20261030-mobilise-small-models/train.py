"""Small MLP follow-up; preserve the completed Mobilise-D experiment inputs."""
import hashlib,json,sys,time,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE=HERE.parent/'20261030-mobilise-bout-regression'
sys.path[:0]=[str(ROOT/'outputs/motionbert-smoke-20261015/deps'),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps')]
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
FEATURES=json.loads((BASE/'protocol.json').read_text())['features']
WIDTHS=[32,64];EPOCHS=list(range(10,121,10));SEED=20261030
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def weights(ids):
    _,i,n=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/n[i];return w/w.sum()
def make(width,seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(4,width),torch.nn.ReLU(),torch.nn.Linear(width,width//2),torch.nn.ReLU(),torch.nn.Linear(width//2,1))
def fit(train,label,width,epochs,seed,valid=None):
    w=weights(train.participantid.to_numpy());x=train[FEATURES].to_numpy();y=train[label].to_numpy()
    xm=np.average(x,axis=0,weights=w);xs=np.sqrt(np.average((x-xm)**2,axis=0,weights=w));xs=np.maximum(xs,1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    model=make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    curves=[]
    if valid is not None:
        vx=torch.tensor((valid[FEATURES].to_numpy()-xm)/xs,dtype=torch.float32);vy=valid[label].to_numpy();vw=weights(valid.participantid.to_numpy())
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();pred=model(xt).squeeze(-1)
        assert pred.shape==yt.shape==wt.shape
        loss=(torch.nn.functional.smooth_l1_loss(pred,yt,reduction='none',beta=.5)*wt).sum()
        assert torch.isfinite(loss)
        loss.backward();optimizer.step()
        if valid is not None and epoch in EPOCHS:
            with torch.no_grad():vp=model(vx).squeeze(-1).numpy()*ys+ym
            curves.append(float(np.dot(vw,abs(vp-vy))))
    bundle={'x_mean':xm,'x_std':xs,'y_mean':np.array(ym),'y_std':np.array(ys)}
    for i,layer in enumerate([model[0],model[2],model[4]]):
        bundle[f'w{i}']=layer.weight.detach().numpy();bundle[f'b{i}']=layer.bias.detach().numpy()
    return model,bundle,curves
def predict(model,bundle,x):
    with torch.no_grad():
        return model(torch.tensor((x-bundle['x_mean'])/bundle['x_std'],dtype=torch.float32)).squeeze(-1).numpy()*float(bundle['y_std'])+float(bundle['y_mean'])
def run():
    torch.set_num_threads(1);started=time.time();data=pd.read_csv(BASE/'features.csv')
    protocol={'reference_study':str(BASE),'features_sha256':sha(BASE/'features.csv'),'folds_sha256':sha(BASE/'person-folds.csv'),'features':FEATURES,'models':'MLP [4,width,width/2,1], ReLU','width_candidates':WIDTHS,'epoch_candidates':EPOCHS,'optimizer':'full-batch AdamW lr=.01 weight_decay=.001','loss':'person-weighted SmoothL1 beta=.5 on training-standardized target','preprocessing':'person-weighted mean/std of X and y, training only','outer_folds':'unchanged person 5-fold','inner_folds':'same seed and person 3-fold as reference','selection':'minimum combined inner-validation person MAE over width and epoch; outer data never selects settings','draws':'fit draw0; evaluate draw0/1/2 independently without averaging predictions','seed':SEED,'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','no_new_people':True}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol);(HERE/'folds').mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    for target,band,k,fold in itertools.product(['sppb','tug'],['short','long'],[1,3,5],range(5)):
        tag=f'{target}-{band}-{k}-f{fold}'
        if (HERE/'folds'/f'{tag}.csv').exists():continue
        label='total_sppb_score' if target=='sppb' else 'tugtime'
        cell=data[data[target+'_ok']&data.band.eq(band)&data.k.eq(k)].sort_values(['draw','participantid','visit_number'])
        train=cell[cell.draw.eq(0)&cell.fold.ne(fold)].reset_index(drop=True);test=cell[cell.fold.eq(fold)].copy()
        assert not set(train.participantid)&set(test.participantid)
        people=train[['participantid','cohort']].drop_duplicates().sort_values('participantid').reset_index(drop=True)
        splits=[];inner_ids=[]
        for a,b in StratifiedKFold(3,shuffle=True,random_state=20261100+fold).split(people,people.cohort):
            tr=train[train.participantid.isin(people.iloc[a].participantid)];va=train[train.participantid.isin(people.iloc[b].participantid)]
            assert not set(tr.participantid)&set(va.participantid)
            splits.append((tr,va));inner_ids.append({'train':sorted(map(int,tr.participantid.unique())),'valid':sorted(map(int,va.participantid.unique()))})
        trials=[]
        for width in WIDTHS:
            curves=[];sizes=[]
            for inner,(tr,va) in enumerate(splits):
                _,_,curve=fit(tr,label,width,max(EPOCHS),SEED+fold*10+inner,va)
                curves.append(curve);sizes.append(va.participantid.nunique())
            scores=np.average(curves,axis=0,weights=sizes)
            trials.extend({'width':width,'epochs':epoch,'inner_person_mae':float(score)} for epoch,score in zip(EPOCHS,scores))
        best=min(trials,key=lambda z:z['inner_person_mae'])
        model,bundle,_=fit(train,label,best['width'],best['epochs'],SEED+fold*10+9)
        np.savez_compressed(HERE/'models'/f'{tag}.npz',**bundle)
        out=test[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
        out['target']=target;out['band']=band;out['k']=k;out['MLP']=predict(model,bundle,test[FEATURES].to_numpy())
        out.to_csv(HERE/'folds'/f'{tag}.csv',index=False)
        save(f'folds/{tag}.json',{'selected':best,'trials':trials,'inner_people':inner_ids,'train_people':sorted(map(int,train.participantid.unique())),'test_people':sorted(map(int,test.participantid.unique())),'parameters':sum(p.numel() for p in model.parameters())})
        print(tag,best,f'elapsed={time.time()-started:.1f}s',flush=True)
    p=pd.concat([pd.read_csv(x) for x in sorted((HERE/'folds').glob('*.csv'))],ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    keys=['participantid','visit_number','draw','fold','target','band','k']
    original=pd.read_csv(BASE/'predictions.csv')
    joined=p.merge(original[keys+['actual','baseline','Ridge','SVR','RF']],on=keys,validate='one_to_one',suffixes=('','_reference'))
    assert len(joined)==len(p)==len(original)
    np.testing.assert_allclose(joined.actual,joined.actual_reference,rtol=0,atol=1e-12)
    rows=[]
    for (target,band,k,draw),d in joined.groupby(['target','band','k','draw']):
        w=weights(d.participantid.to_numpy());baseline=float(np.dot(w,abs(d.actual-d.baseline)));svr=float(np.dot(w,abs(d.actual-d.SVR)))
        mae=float(np.dot(w,abs(d.actual-d.MLP)))
        rows.append({'target':target,'band':band,'k':int(k),'draw':int(draw),'people':int(d.participantid.nunique()),'visits':len(d),'MLP_mae':mae,'SVR_mae':svr,'baseline_mae':baseline,'baseline_improvement_percent':100*(baseline-mae)/baseline,'SVR_improvement':svr-mae})
    metrics=pd.DataFrame(rows);metrics.to_csv(HERE/'metrics.csv',index=False)
    save('completion.json',{'folds':60,'outer_models':60,'inner_fits':360,'elapsed_seconds':time.time()-started,'predictions_sha256':sha(HERE/'predictions.csv')})
    print(metrics[metrics.draw.eq(0)].round(4).to_string(index=False),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
