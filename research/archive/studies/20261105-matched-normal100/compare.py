"""Report only independently audited, complete five-fold matched comparisons."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/matched-normal100-20261105');OLD=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'child/data.npz') as z:d={k:z[k][:1061] for k in ['y','person_id','fold','age','source']}
y=d['y'];ids=d['person_id'];assert np.all(d['source']=='Stanford');provenance=[]
def load(arm,family):
    if arm=='baseline':root=OLD/('runs/baseline' if family=='RF' else 'received/data/runs/baseline')
    else:
        if family=='RF':
            audit=HERE/('rf-audit-'+arm+'.json')
            if not audit.exists() or not read(audit)['passed']:return None
            root=BIG/arm/'runs/mixed'
        else:
            covered=set()
            for audit in (HERE/'audits'/arm).glob(family+'-*.json'):
                v=read(audit)
                if v['passed']:covered.update(r['fold'] for r in v['folds'] if r['passed'])
            if covered!=set(range(5)):return None
            root=BIG/'received/data'/arm/'runs/mixed'
    result={k:np.full_like(y,np.nan,dtype=float) for k in (['refit'] if family=='RF' else ['refit','bag'])}
    for f in range(5):
        folder=root/family/f'fold{f}';done=read(folder/'complete.json');path=folder/'predictions.npz';assert sha(path)==done['prediction_sha256']
        with np.load(path) as z:
            ix=z['row_index'];use=ix<1061;ix=ix[use];assert np.array_equal(ix,np.flatnonzero(d['fold']==f))
            for k in result:result[k][ix]=z[k][use]
        provenance.append(dict(arm=arm,family=family,fold=f,path=str(path),sha256=sha(path)))
    assert all(np.isfinite(v).all() for v in result.values());return result
predictions={}
for arm in ['baseline','adult','child']:
    p={}
    for family in ['CNN','RF','STT','bilateral']:
        values=load(arm,family)
        if values is None:continue
        for k,v in values.items():
            if family=='bilateral' and k=='bag':continue
            p[family if family=='RF' else family+'_'+k]=v
    if all(k in p for k in ['CNN_bag','STT_bag','bilateral_refit']):
        fusion=np.empty_like(y,dtype=float)
        for row in read(HERE/'bundle/ensemble.json'):
            mask=d['fold']==row['outer'];w=row['cnn_weight'];t=row['fraction']
            b=w*p['CNN_bag'][mask]+(1-w)*p['STT_bag'][mask];mean=b.mean(axis=1)
            diff=(1-t)*(b[:,0]-b[:,1])+t*(p['bilateral_refit'][mask,0]-p['bilateral_refit'][mask,1])
            fusion[mask]=np.stack([mean+diff/2,mean-diff/2],axis=1)
        p['fusion']=fusion
    predictions[arm]=p
groups={'all':np.ones_like(y,bool),'GDI_ge90':y>=90,'GDI_ge100':y>=100,'GDI_lt70':y<70}
def measures(pred,mask):
    people=np.unique(ids[mask.any(axis=1)]);e=pred-y
    per={k:[] for k in ['mae','bias','shortfall','underprediction_fraction']}
    for p in people:
        z=e[mask&(ids==p)[:,None]]
        for k,v in [('mae',np.abs(z)),('bias',z),('shortfall',np.maximum(-z,0)),('underprediction_fraction',z<0)]:per[k].append(float(v.mean()))
    # Separate flat weighted calculation validates person/side aggregation.
    weights=np.zeros_like(y,dtype=float)
    for p in people:
        m=mask&(ids==p)[:,None];weights[m]=1/(m.sum()*len(people))
    assert abs(sum((np.abs(e)*weights).ravel())-np.mean(per['mae']))<1e-10
    return {k:np.array(v) for k,v in per.items()}
models={}
for name in ['CNN_refit','RF','STT_refit','CNN_bag','STT_bag','bilateral_refit','fusion']:
    results={}
    for g,mask in groups.items():
        per={arm:measures(p[name],mask) for arm,p in predictions.items() if name in p}
        result=dict(people=len(np.unique(ids[mask.any(axis=1)])),records=int(mask.any(axis=1).sum()),sides=int(mask.sum()),arms={arm:{k:float(v.mean()) for k,v in m.items()} for arm,m in per.items()},contrasts={})
        for before,after in [('baseline','adult'),('baseline','child'),('adult','child')]:
            if before not in per or after not in per:continue
            rng=np.random.default_rng(20261105);draw=rng.integers(0,result['people'],size=(10000,result['people']));contrast={}
            for k in ['mae','bias','shortfall']:
                diff=per[after][k]-per[before][k];contrast[k]=dict(delta=float(diff.mean()),paired95=np.quantile(diff[draw].mean(axis=1),[.025,.975]).tolist())
            result['contrasts'][after+'_minus_'+before]=contrast
        results[g]=result
    models[name]=dict(status='verified_three_arm_comparison' if all(name in p for p in predictions.values()) else 'neural_training_pending',groups=results)
out=dict(status='Only completed and audited models reported',scope='Same Stanford measured-GDI development OOF people; all added people excluded. Original final223 unopened.',uncertainty='Paired10000 person bootstrap conditional on fitted models; descriptive multiple comparisons, no seed or adult-sample uncertainty.',models=models,prediction_provenance=provenance)
(HERE/'comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({name:v for name,v in models.items() if v['status']=='verified_three_arm_comparison'},indent=2))
