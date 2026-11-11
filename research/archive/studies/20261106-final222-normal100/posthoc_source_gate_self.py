import json
from pathlib import Path
import numpy as np
BIG=Path('D:/gait-research-data/D_P/unseen-normal-test-20261105')
ROOT=Path('D:/Projects/D_P/studies')
gate=json.load(open(ROOT/'20261105-normal-transfer-followup/source-gate-results.json',encoding='utf-8-sig'))
Z=np.load(BIG/'inputs.npz')
meta=json.load(open(ROOT/'20261105-unseen-normal-test/preparation.json',encoding='utf-8-sig'))
nums=np.array(meta['self_numbers']);norm=np.isin(nums,meta['self_normal']);abn=np.isin(nums,meta['self_abnormal'])
def feats(x,owner,n):
    w=np.concatenate([x.mean(2),x.std(2),np.ptp(x,axis=2),np.abs(np.diff(x,axis=2)).mean(2)],axis=2).mean(1)
    f=np.zeros((n,48));np.add.at(f,owner,w);return f/np.bincount(owner,minlength=n)[:,None]
def prob(f):
    out=[]
    for m in gate['models']:
        l=((f-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coef'])+m['intercept'];out.append(1/(1+np.exp(-l)))
    return np.array(out)
for name,grp in (('live',None),('self',1)):
    x=Z[name+'__x'].astype(np.float64);owner=Z[name+'__owner'];n=int(Z[name+'__n'])
    p=prob(feats(x,owner,n))
    for fam in ('CNN','STT'):
        P=np.load(BIG/f'pred-{fam}.npz')
        base=np.array([P[f'{name}__baseline__{f}__refit'] for f in range(5)])
        gated=np.where(p[:,:,None]>=.5,100.,base).mean(0).mean(1); b=base.mean(0).mean(1)
        if name=='live':
            ids=np.array(meta['live_people']);q=sorted(set(ids.tolist()))
            print(name,fam,'gate person mean',round(np.mean([gated[ids==i].mean() for i in q]),2),'(expected CNN 97.91 STT 98.04)')
        else:
            print(name,fam,'base normal',round(b[norm].mean(),2),'abn',round(b[abn].mean(),2),'| gate normal',round(gated[norm].mean(),2),'abn',round(gated[abn].mean(),2),'gap',round(gated[norm].mean()-gated[abn].mean(),2))
            print('   gate per video',dict(zip(nums.tolist(),np.round(gated,1).tolist())))
    print(name,'mean p by video',np.round(p.mean(0),2).tolist(),'folds>=.5 counts',(p>=.5).sum(0).tolist())
