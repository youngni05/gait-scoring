"""Source-separated, paired person-level comparisons on identical held-out rows."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/ljmu100-mixed-20261104')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
ap=argparse.ArgumentParser();ap.add_argument('--rf-only',action='store_true');ap.add_argument('--available',action='store_true');a=ap.parse_args()
with np.load(BIG/'data.npz') as z:d={k:z[k] for k in z.files if k!='x'}
ids=d['person_id'];y=d['y'];fold=d['fold'];sources=d['source']
people=np.unique(ids);source=np.array([sources[ids==p][0] for p in people]);true=np.array([y[ids==p].mean() for p in people])
def perperson(p):
    return {'error':np.array([np.abs(p[ids==q]-y[ids==q]).mean() for q in people]),
            'bias':np.array([(p[ids==q]-y[ids==q]).mean() for q in people]),
            'predicted':np.array([p[ids==q].mean() for q in people])}
def loadpred(arm,family,kind):
    root=BIG/'runs' if family=='RF' else BIG/'received/data/runs'
    pred=np.full_like(y,np.nan,dtype=float)
    for f in range(5):
        folder=root/arm/family/f'fold{f}';done=read(folder/'complete.json');path=folder/'predictions.npz'
        assert sha(path)==done['prediction_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['row_index'],np.flatnonzero(fold==f));pred[z['row_index']]=z[kind]
    assert np.isfinite(pred).all();return pred
models={'RF':('RF','refit')}
if not a.rf_only:models={'CNN_refit':('CNN','refit'),'RF':('RF','refit'),'STT_refit':('STT','refit'),'CNN_bag':('CNN','bag'),'STT_bag':('STT','bag'),'bilateral_refit':('bilateral','refit')}
if a.available:
    audited={}
    for p in (HERE/'audits').glob('*.json'):
        audit=read(p)
        if audit.get('passed'):
            audited.setdefault(audit['family'],set()).update(v['fold'] for v in audit['folds'])
    models={name:value for name,value in models.items() if audited.get(value[0],set())==set(range(5))}
preds={arm:{name:loadpred(arm,*value) for name,value in models.items()} for arm in ['baseline','mixed']}
if all(name in models for name in ['CNN_bag','STT_bag','bilateral_refit']):
    for arm in preds:
        fusion=np.empty_like(y,dtype=float)
        for v in read(HERE/'bundle/ensemble.json'):
            m=fold==v['outer'];w=v['cnn_weight'];t=v['fraction'];p=preds[arm]
            b=w*p['CNN_bag'][m]+(1-w)*p['STT_bag'][m];bil=p['bilateral_refit'][m]
            common=b.mean(axis=1);diff=(1-t)*(b[:,0]-b[:,1])+t*(bil[:,0]-bil[:,1])
            fusion[m]=np.stack([common+diff/2,common-diff/2],axis=1)
        preds[arm]['fusion']=fusion
metrics={arm:{name:perperson(p) for name,p in values.items()} for arm,values in preds.items()}
rows=[];groups=[]
for name in preds['baseline']:
    for s in ['Stanford','LJMU']:
        ix=source==s;b=metrics['baseline'][name];m=metrics['mixed'][name];delta=m['error'][ix]-b['error'][ix]
        rng=np.random.default_rng(20261104);samples=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1);ci=np.quantile(samples,[.025,.975])
        rows.append({'model':name,'source':s,'people':int(ix.sum()),'records':int((sources==s).sum()),
                     'before_MAE':float(b['error'][ix].mean()),'after_MAE':float(m['error'][ix].mean()),
                     'delta_after_minus_before':float(delta.mean()),'delta_95pct_low':float(ci[0]),'delta_95pct_high':float(ci[1]),
                     'before_bias':float(b['bias'][ix].mean()),'after_bias':float(m['bias'][ix].mean()),
                     'before_mean_prediction':float(b['predicted'][ix].mean()),'after_mean_prediction':float(m['predicted'][ix].mean()),
                     'label_kind':'measured_GDI' if s=='Stanford' else 'assumed_100'})
    for lo,hi,label in [(-np.inf,60,'below60'),(60,75,'60to75'),(75,90,'75to90'),(90,np.inf,'90plus')]:
        ix=(source=='Stanford')&(true>=lo)&(true<hi)
        groups.append({'model':name,'group':label,'people':int(ix.sum()),**{arm+'_MAE':float(metrics[arm][name]['error'][ix].mean()) for arm in metrics}})
suffix='rf-comparison' if a.rf_only else 'comparison-available' if a.available else 'comparison'
result={'status':'RF_complete_neural_pending' if a.rf_only else 'partial_verified_models' if a.available else 'complete','data_sha256':sha(BIG/'data.npz'),
        'metric':'Equal-person average of record/side absolute errors; same held-out people before and after.',
        'uncertainty':'Paired 10000 person bootstrap, seed20261104. Conditional on these fitted OOF models; development reuse, not independent final validation.',
        'LJMU_interpretation':'Distance from assumed100. No measured LJMU GDI available.',
        'rows':rows,'Stanford_true_person_mean_GDI_groups':groups}
save(HERE/(suffix+'.json'),result)
with (HERE/(suffix+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(BIG/(suffix+'-predictions.npz'),**{arm+'__'+name:p for arm,values in preds.items() for name,p in values.items()},person_id=ids,y=y,source=sources,fold=fold)
print(json.dumps(rows,indent=2))
