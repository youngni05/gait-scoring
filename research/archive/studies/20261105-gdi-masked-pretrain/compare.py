"""Report completed, independently replayed A-D predictions on the same people."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=BIG/'received/runs');a=ap.parse_args()
p=read(HERE/'bundle/protocol.json');assert sha(BIG/'data.npz')==p['data_sha256']
with np.load(BIG/'data.npz') as z:y=z['y'][:1061];ids=z['person_id'][:1061];fold=z['fold'][:1061]
assert len(np.unique(ids))==692 and np.isfinite(y).all()
audited={}
for path in (HERE/'audits').glob('main-*.json'):
    v=read(path);assert v['passed']
    for r in v['audits']:
        if r['passed']:audited[(r['family'],r['condition'],r['fold'])]=r
assert len(audited)==60,'All60 fitted models require independent replay first'
rows=[];contrasts=[];budgets=[];provenance=[]
groups={'all':np.ones_like(y,bool),'GDI_ge90':y>=90,'GDI_ge100':y>=100,'GDI_lt70':y<70}
metrics=['mae','underprediction_shortfall','bias','underprediction_fraction']
for family in p['families']:
    predictions={}
    for condition in p['conditions']:
        prediction=np.full_like(y,np.nan,dtype=float);raw=np.full_like(y,np.nan,dtype=float)
        for f in p['folds']:
            folder=a.root/'main'/family/condition/('fold%d'%f);done=read(folder/'complete.json');audit=audited[(family,condition,f)]
            assert sha(folder/'complete.json')==audit['complete_sha256'] and sha(folder/'predictions.npz')==done['prediction_sha256']
            with np.load(folder/'predictions.npz') as z:
                ix=z['evaluation_rows'];assert np.array_equal(ix,np.flatnonzero(fold==f));prediction[ix]=z['prediction'];raw[ix]=z['evaluation_raw']
            budgets.append(dict(family=family,condition=condition,fold=f,gdi_updates=done['optimizer_steps'],gdi_examples=done['examples'],pretrain_updates=done['pretraining_steps'],head_initial_sha256=done['head_initial_sha256']))
            provenance.append(dict(path=str(folder/'predictions.npz'),sha256=done['prediction_sha256']))
        assert np.isfinite(prediction).all();predictions[condition]={'calibrated':prediction,'raw':raw}
    for f in p['folds']:
        b=[v for v in budgets if v['family']==family and v['fold']==f]
        assert len({v['gdi_updates'] for v in b})==len({v['gdi_examples'] for v in b})==len({v['head_initial_sha256'] for v in b})==1
    for output in ['calibrated','raw']:
      for group,eligible in groups.items():
        persons=np.unique(ids[eligible.any(1)]);individual={}
        for condition,pp in predictions.items():
            errors=pp[output]-y;values=[]
            for person in persons:
                e=errors[eligible&(ids==person)[:,None]];values.append([np.abs(e).mean(),np.maximum(-e,0).mean(),e.mean(),(e<0).mean()])
            individual[condition]=np.array(values)
            # Independently aggregate eligible side/record observations with inverse-person-count weights.
            ri,si=np.nonzero(eligible);count={q:int((ids[ri]==q).sum()) for q in persons}
            w=np.array([1/(len(persons)*count[q]) for q in ids[ri]]);ee=errors[ri,si]
            expected=np.array([np.abs(ee)@w,np.maximum(-ee,0)@w,ee@w,(ee<0)@w]);assert np.max(np.abs(expected-individual[condition].mean(0)))<1e-10
            rows.append(dict(family=family,condition=condition,output=output,group=group,people=len(persons),records=int(eligible.any(1).sum()),side_observations=int(eligible.sum()),**dict(zip(metrics,map(float,expected)))))
        rng=np.random.default_rng(20261105);draws=rng.integers(len(persons),size=(10000,len(persons)))
        pairs=[('A','B_adult'),('A','B_child'),('A','C'),('C','D_adult'),('C','D_child'),('B_adult','D_adult'),('B_child','D_child'),('D_adult','D_child'),('A','D_adult'),('A','D_child')]
        for before,after in pairs:
            diff=individual[after]-individual[before]
            for k,m in enumerate(metrics):
                lo,hi=np.quantile(diff[:,k][draws].mean(1),[.025,.975]);contrasts.append(dict(family=family,output=output,group=group,before=before,after=after,metric=m,delta=float(diff[:,k].mean()),low95=float(lo),high95=float(hi)))
out=dict(status='All60 models independently replayed',rows=rows,contrasts=contrasts,budgets=budgets,prediction_provenance=provenance,protocol_sha256=sha(HERE/'bundle/protocol.json'),
    primary_output='calibrated',uncertainty='Paired10000 person bootstrap; descriptive unadjusted95% intervals, conditional on fitted models. No initialization/normal-sample uncertainty; reused development cohort, final223 unopened.',
    labels='Only measured Stanford record-side GDI. External people absent from all metrics. Underprediction=max(y-p,0), including zeros for overprediction.')
(HERE/'comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
with (HERE/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps([r for r in rows if r['output']=='calibrated'],indent=2))
