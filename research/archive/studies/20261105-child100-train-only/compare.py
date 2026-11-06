"""Compare identical held-out Stanford people; group by actual record-side GDI."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
BIG=Path('D:/gait-research-data/D_P/child100-train-only-20261105')
OLD=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files if k!='x'}
assert sha(BIG/'data.npz')==read(HERE/'bundle/study-protocol.json')['data_sha256']
n=1061;y=d['y'][:n];ids=d['person_id'][:n];age=d['age'][:n]
assert (d['source'][:n]=='Stanford').all() and (d['fold'][n:]==-1).all()
provenance=[]

def load_family(root,family):
    result={k:np.full((n,2),np.nan) for k in (['refit'] if family=='RF' else ['refit','bag'])}
    for f in range(5):
        folder=root/family/('fold%d'%f)
        if not (folder/'complete.json').exists():return None
        done=read(folder/'complete.json');p=folder/'predictions.npz'
        assert sha(p)==done['prediction_sha256']
        with np.load(p) as z:
            ix=z['row_index'];m=ix<n;ix=ix[m]
            assert np.array_equal(ix,np.flatnonzero(d['fold'][:n]==f))
            for k in result:result[k][ix]=z[k][m]
        provenance.append(dict(path=str(p),sha256=sha(p)))
    assert all(np.isfinite(p).all() for p in result.values())
    return result

preds={}
for arm in ['baseline','mixed']:
    p={}
    for fam in ['RF','CNN','STT','bilateral']:
        root=(OLD/('runs/baseline' if fam=='RF' else 'received/data/runs/baseline')) if arm=='baseline' else (BIG/('runs/mixed' if fam=='RF' else 'received/data/runs/mixed'))
        values=load_family(root,fam)
        if values is not None:
            for k,v in values.items():
                if fam=='bilateral' and k=='bag':continue
                p[fam if fam=='RF' else fam+'_'+k]=v
    if all(k in p for k in ['CNN_bag','STT_bag','bilateral_refit']):
        fusion=np.empty_like(y)
        for r in read(HERE/'bundle/ensemble.json'):
            ix=d['fold'][:n]==r['outer'];w=r['cnn_weight'];t=r['fraction']
            base=w*p['CNN_bag'][ix]+(1-w)*p['STT_bag'][ix]
            common=base.mean(axis=1);bil=p['bilateral_refit'][ix]
            diff=(1-t)*(base[:,0]-base[:,1])+t*(bil[:,0]-bil[:,1])
            fusion[ix]=np.stack([common+diff/2,common-diff/2],axis=1)
        p['fusion']=fusion
    preds[arm]=p

def per_person(pred,eligible):
    persons=np.unique(ids[eligible.any(axis=1)])
    abs_err=[];signed=[]
    for person in persons:
        m=eligible&(ids==person)[:,None];e=(pred-y)[m]
        abs_err.append(np.abs(e).mean());signed.append(e.mean())
    return np.array(abs_err),np.array(signed),persons

groups={'all':np.ones_like(y,dtype=bool),'GDI_ge90':y>=90,'GDI_ge100':y>=100,'GDI_lt70':y<70,
        'age5to12':np.broadcast_to(((age>=5)&(age<13))[:,None],y.shape)}
out={}
for name in ['CNN_refit','RF','STT_refit','CNN_bag','STT_bag','bilateral_refit','fusion']:
    before=preds['baseline'][name];after=preds['mixed'].get(name);result={}
    for group,eligible in groups.items():
        b,bb,persons=per_person(before,eligible)
        r=dict(people=len(persons),records=int(eligible.any(axis=1).sum()),side_observations=int(eligible.sum()),
               before_mae=float(b.mean()),before_bias=float(bb.mean()))
        if after is not None:
            a,ab,ap=per_person(after,eligible);assert np.array_equal(ap,persons)
            diff=a-b;rng=np.random.default_rng(20261105)
            draws=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(axis=1)
            r.update(after_mae=float(a.mean()),after_bias=float(ab.mean()),mae_delta=float(diff.mean()),
                     paired_person_bootstrap95=np.quantile(draws,[.025,.975]).tolist())
        result[group]=r
    out[name]=dict(status='predictions_available' if after is not None else 'child_training_pending',groups=result)
assert abs(out['RF']['groups']['all']['before_mae']-7.1818329439934)<1e-6
result=dict(metric='Eligible measured-GDI side-record errors averaged within person, then equal-person mean. Negative after-minus-before MAE delta is improvement.',
            scope='Five development outer folds only; this is not untouched final evaluation.',models=out,provenance=provenance)
(HERE/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if v['status']=='predictions_available'},indent=2))
