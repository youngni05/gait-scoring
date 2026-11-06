"""Shared immutable participant plans, inputs and metrics; never resplit test people."""
import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT_SHA='115f5c87b74218276f8b8db71c2c747322d2358ab3b3b5bd066c719318002330'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def data(path):
    assert sha(path)==INPUT_SHA
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    assert d['x'].shape==(220,2,6,124) and np.isfinite(d['x']).all()
    np.testing.assert_array_equal(d['owner'],np.arange(220))
    assert len(np.unique(d['ids']))==44
    for pid in np.unique(d['ids']):
        assert np.count_nonzero(d['ids']==pid)==5
        assert len(np.unique(d['y'][d['ids']==pid]))==len(np.unique(d['folds'][d['ids']==pid]))==1
    return d
def mask(d,ids):return np.isin(d['ids'],ids)
def people(d,m):return sorted(np.unique(d['ids'][m]).tolist())
def hist(d,ids):return {str(k):sum(float(d['y'][d['ids']==p].mean())==k for p in ids) for k in range(5,13)}
def make_plan(d):
    out=[]
    for fold in range(5):
        test=people(d,d['folds']==fold);available=people(d,d['folds']!=fold)
        orders=[]
        for draw in range(3):
            rng=np.random.RandomState(20261110+fold*100+draw)
            bands=[[],[],[]]
            for p in available:
                y=float(d['y'][d['ids']==p].mean());bands[0 if y<=8 else (1 if y<=10 else 2)].append(p)
            for b in bands:rng.shuffle(b)
            sizes=np.array([len(b) for b in bands]);used=np.zeros(3,int);order=[]
            for n in range(len(available)):
                deficit=(n+1)*sizes/len(available)-used
                deficit[used==sizes]=-np.inf
                k=int(np.argmax(deficit));order.append(bands[k][used[k]]);used[k]+=1
            orders.append(order)
        for size in (8,16,24,'full'):
            for draw in (range(3) if size!='full' else [-1]):
                selected=sorted(orders[draw][:size]) if size!='full' else available
                shuffled=np.array(selected);np.random.RandomState(20261031).shuffle(shuffled)
                inners=[]
                for validation in np.array_split(shuffled,3):
                    va=sorted(validation.tolist());tr=sorted(set(selected)-set(va))
                    inners.append(dict(train=tr,validation=va))
                out.append(dict(key='n%s-d%s'%(size,draw),fold=fold,size=size,draw=draw,people=selected,test=test,
                                inner=inners,score_histogram=hist(d,selected),final_training_people=len(selected)))
    return dict(schema=1,input_sha256=INPUT_SHA,cases=out)
def check_plan(plan,d):
    assert plan==make_plan(d),'Participant plan differs from declared deterministic algorithm'
    assert len(plan['cases'])==50
    for c in plan['cases']:
        assert not set(c['people'])&set(c['test'])
        assert set(c['test'])==set(people(d,d['folds']==c['fold']))
        for j in c['inner']:
            assert not set(j['train'])&set(j['validation'])
            assert set(j['train'])|set(j['validation'])==set(c['people'])
    for f in range(5):
        for draw in range(3):
            sets=[set(next(c['people'] for c in plan['cases'] if c['fold']==f and c['size']==n and c['draw']==draw)) for n in (8,16,24)]
            assert sets[0]<sets[1]<sets[2]
def person_rows(d,record_predictions,selected):
    return [dict(id=p,y=float(d['y'][d['ids']==p].mean()),prediction=float(record_predictions[d['ids']==p].mean())) for p in sorted(selected)]
def metrics(rows):
    y=np.array([r['y'] for r in rows]);p=np.array([r['prediction'] for r in rows]);e=p-y
    return dict(MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),
                within1=float(np.mean(np.abs(e)<=1)),within2=float(np.mean(np.abs(e)<=2)),
                rounded_exact=float(np.mean(np.floor(p+.5)==y)),low5_8_MAE=float(np.abs(e[y<=8]).mean()) if np.any(y<=8) else None,
                high11_12_MAE=float(np.abs(e[y>=11]).mean()) if np.any(y>=11) else None)
