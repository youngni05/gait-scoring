"""Parameter-matched Deep Sets: encode each bout before mean pooling."""
import hashlib,json,sys,time,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE=HERE.parent/'20261030-mobilise-bout-regression'
MLP=HERE.parent/'20261030-mobilise-small-models'
sys.path[:0]=[str(ROOT/'outputs/motionbert-smoke-20261015/deps'),str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps')]
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
FEATURES=json.loads((BASE/'protocol.json').read_text())['features']
BAGKEY=['participantid','visit_number','draw','band']
WIDTHS=[32,64];EPOCHS=list(range(10,121,10));SEED=20261030
def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def weights(ids):
    _,i,n=np.unique(ids,return_inverse=True,return_counts=True)
    w=1/n[i];return w/w.sum()
class DeepSet(torch.nn.Module):
    def __init__(self,width):
        super().__init__()
        # Layer creation order matches the reference MLP exactly.
        self.layers=torch.nn.Sequential(torch.nn.Linear(4,width),torch.nn.ReLU(),torch.nn.Linear(width,width//2),torch.nn.ReLU(),torch.nn.Linear(width//2,1))
    def forward(self,x):
        h=self.layers[1](self.layers[0](x)).mean(dim=1)
        return self.layers[4](self.layers[3](self.layers[2](h)))
def make(width,seed):
    torch.manual_seed(seed);return DeepSet(width)
def load_data():
    bouts=pd.read_csv(BASE/'selected-bouts.csv').sort_values(BAGKEY+['rank'])
    assert not bouts.duplicated(BAGKEY+['rank']).any()
    assert bouts.groupby(BAGKEY).size().eq(5).all()
    ids=bouts[BAGKEY].drop_duplicates().reset_index(drop=True);ids['bag_id']=np.arange(len(ids))
    arrays=bouts[FEATURES].to_numpy().reshape(len(ids),5,4)
    # Repetition of each group's ID is checked before reshaping.
    actual=bouts[BAGKEY].reset_index(drop=True)
    expected=ids[BAGKEY].loc[ids.index.repeat(5)].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected)
    table=pd.read_csv(BASE/'features.csv').merge(ids,on=BAGKEY,validate='many_to_one')
    for k in [1,3,5]:
        d=table[table.k.eq(k)]
        np.testing.assert_allclose(np.median(arrays[d.bag_id.to_numpy(),:k],axis=1),d[FEATURES],rtol=1e-12,atol=1e-12)
    return table,arrays,ids
def fit(train,label,width,epochs,seed,arrays,k,valid=None):
    w=weights(train.participantid.to_numpy());x=arrays[train.bag_id.to_numpy(),:k];y=train[label].to_numpy()
    flat=x.reshape(-1,4);fw=np.repeat(w/k,k)
    xm=np.average(flat,axis=0,weights=fw);xs=np.maximum(np.sqrt(np.average((flat-xm)**2,axis=0,weights=fw)),1e-8)
    ym=float(np.average(y,weights=w));ys=max(float(np.sqrt(np.average((y-ym)**2,weights=w))),1e-8)
    xt=torch.tensor((x-xm)/xs,dtype=torch.float32);yt=torch.tensor((y-ym)/ys,dtype=torch.float32);wt=torch.tensor(w,dtype=torch.float32)
    model=make(width,seed);optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.001)
    curves=[]
    if valid is not None:
        vx=torch.tensor((arrays[valid.bag_id.to_numpy(),:k]-xm)/xs,dtype=torch.float32);vy=valid[label].to_numpy();vw=weights(valid.participantid.to_numpy())
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
    for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
        bundle[f'w{i}']=layer.weight.detach().numpy();bundle[f'b{i}']=layer.bias.detach().numpy()
    return model,bundle,curves
def predict(model,bundle,x):
    with torch.no_grad():
        return model(torch.tensor((x-bundle['x_mean'])/bundle['x_std'],dtype=torch.float32)).squeeze(-1).numpy()*float(bundle['y_std'])+float(bundle['y_mean'])
def run():
    torch.set_num_threads(1);started=time.time();data,arrays,ids=load_data()
    protocol={'reference_study':str(BASE),'reference_mlp':str(MLP),'source_hashes':{str(p):sha(p) for p in [BASE/'features.csv',BASE/'selected-bouts.csv',BASE/'person-folds.csv',BASE/'predictions.csv',MLP/'predictions.csv',MLP/'train.py']},'features':FEATURES,'input':'same selected bouts, rank<=k, [visits,k,4]; no metadata as predictors','architecture':'Linear(4,width), ReLU per bout; mean across bouts; Linear(width,width/2), ReLU, Linear(width/2,1)','parameter_matched_to_mlp':True,'width_candidates':WIDTHS,'epoch_candidates':EPOCHS,'optimizer':'full-batch AdamW lr=.01 weight_decay=.001','loss':'person-weighted SmoothL1 beta=.5 on training-standardized target','preprocessing':'training-only person-weighted X mean/std across raw selected bouts, each visit contributes weight/k to each bout; training-only weighted y mean/std','outer_folds':'unchanged person 5-fold','inner_folds':'same seed and person 3-fold as reference MLP','selection':'minimum combined inner-validation person MAE over width and epoch','draws':'fit draw0; evaluate draw0/1/2 separately, no averaging predictions','seed':SEED,'torch':torch.__version__,'numpy':np.__version__,'device':'CPU','primary_draw':0,'uncertainty':'paired person bootstrap of fixed OOF predictions; not refitting and not independent final validation','scope':'small mean-pooled Deep Sets, not all possible set architectures; raw bout summaries, not raw motion time series'}
    if (HERE/'protocol.json').exists():assert json.loads((HERE/'protocol.json').read_text())==protocol
    save('protocol.json',protocol);(HERE/'folds').mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    ids.to_csv(HERE/'bag-index.csv',index=False)
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
                _,_,curve=fit(tr,label,width,max(EPOCHS),SEED+fold*10+inner,arrays,k,va)
                curves.append(curve);sizes.append(va.participantid.nunique())
            scores=np.average(curves,axis=0,weights=sizes)
            trials.extend({'width':width,'epochs':epoch,'inner_person_mae':float(score)} for epoch,score in zip(EPOCHS,scores))
        best=min(trials,key=lambda z:z['inner_person_mae'])
        model,bundle,_=fit(train,label,best['width'],best['epochs'],SEED+fold*10+9,arrays,k)
        np.savez_compressed(HERE/'models'/f'{tag}.npz',**bundle)
        out=test[['participantid','visit_number','draw','fold',label]].rename(columns={label:'actual'}).copy()
        out['target']=target;out['band']=band;out['k']=k
        out['DeepSets']=predict(model,bundle,arrays[test.bag_id.to_numpy(),:k])
        out.to_csv(HERE/'folds'/f'{tag}.csv',index=False)
        save(f'folds/{tag}.json',{'selected':best,'trials':trials,'inner_people':inner_ids,'train_people':sorted(map(int,train.participantid.unique())),'test_people':sorted(map(int,test.participantid.unique())),'parameters':sum(p.numel() for p in model.parameters())})
        print(tag,best,f'elapsed={time.time()-started:.1f}s',flush=True)
    p=pd.concat([pd.read_csv(x) for x in sorted((HERE/'folds').glob('*.csv'))],ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    save('completion.json',{'folds':60,'outer_models':60,'inner_fits':360,'elapsed_seconds':time.time()-started,'predictions_sha256':sha(HERE/'predictions.csv')})
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
