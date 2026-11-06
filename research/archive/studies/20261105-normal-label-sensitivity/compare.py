"""Compare40 new fits and20 frozen-budget A/B references only after all audits."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'studies/20261105-gdi-masked-pretrain-v3';BIG=Path('D:/gait-research-data/D_P/normal-label-sensitivity-20261105');BASE=Path('D:/gait-research-data/D_P/gdi-masked-pretrain-v3-20261105')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(HERE/'bundle/protocol.json');assert sha(BASE/'data.npz')==p['data_sha256']
with np.load(BASE/'data.npz') as z:y=z['y'][:1061];ids=z['person_id'][:1061];fold=z['fold'][:1061]
audits={}
for study in [HERE,OLD]:
 for path in (study/'audits').glob('main-*.json'):
  a=read(path);assert a['passed']
  for r in a['audits']:audits[(study.name,r['family'],r['condition'],r['fold'])]=r
conditions=['A']+p['conditions'];rows=[];contrasts=[];provenance=[];metrics=['mae','shortfall','bias','under_fraction','overprediction']
groups={'all':np.ones_like(y,bool),'GDI_ge90':y>=90,'GDI_ge100':y>=100,'GDI_lt70':y<70}
for family in p['families']:
 preds={};budget={}
 for condition in conditions:
  reuse=condition in ['A','B_y100_w1_adult'];study=OLD if reuse else HERE;root=BASE if reuse else BIG;actual='B_adult' if condition=='B_y100_w1_adult' else condition
  pred=np.full_like(y,np.nan,dtype=float);raw=pred.copy()
  for f in range(5):
   a=audits[(study.name,family,actual,f)];folder=root/'received/runs/main'/family/actual/('fold%d'%f);done=read(folder/'complete.json')
   assert a['passed'] and a['complete_sha256']==sha(folder/'complete.json');assert sha(folder/'predictions.npz')==done['prediction_sha256']
   with np.load(folder/'predictions.npz') as z:
    ix=z['evaluation_rows'];assert np.array_equal(ix,np.flatnonzero(fold==f));pred[ix]=z['prediction'];raw[ix]=z['evaluation_raw']
   b=(done['epochs'],done['optimizer_steps'],done['examples'],done['head_initial_sha256'])
   if f in budget:assert budget[f]==b
   else:budget[f]=b
   provenance.append(dict(family=family,condition=condition,fold=f,path=str(folder/'predictions.npz'),sha256=done['prediction_sha256'],reused=reuse))
  assert np.isfinite(pred).all();preds[condition]={'calibrated':pred,'raw':raw}
 for output in ['calibrated','raw']:
  for group,mask in groups.items():
   people=np.unique(ids[mask.any(1)]);individual={}
   for condition,outputs in preds.items():
    err=outputs[output]-y;v=[]
    for person in people:
     a=err[mask&(ids==person)[:,None]];v.append([np.abs(a).mean(),np.maximum(-a,0).mean(),a.mean(),(a<0).mean(),np.maximum(a,0).mean()])
    v=np.array(v);individual[condition]=v
    ri,si=np.nonzero(mask);count={q:int((ids[ri]==q).sum()) for q in people};w=np.array([1/(len(people)*count[q]) for q in ids[ri]]);ee=err[ri,si]
    expected=np.array([np.abs(ee)@w,np.maximum(-ee,0)@w,ee@w,(ee<0)@w,np.maximum(ee,0)@w]);assert np.max(np.abs(expected-v.mean(0)))<1e-10
    rows.append(dict(family=family,condition=condition,output=output,group=group,people=len(people),**dict(zip(metrics,map(float,expected)))))
   draws=np.random.default_rng(20261105).integers(len(people),size=(10000,len(people)))
   for reference in ['A','B_y100_w1_adult']:
    for condition in conditions:
     if condition==reference:continue
     delta=individual[condition]-individual[reference]
     for i,m in enumerate(metrics):
      ci=np.quantile(delta[:,i][draws].mean(1),[.025,.975]);contrasts.append(dict(family=family,output=output,group=group,reference=reference,condition=condition,metric=m,delta=float(delta[:,i].mean()),low95=float(ci[0]),high95=float(ci[1])))
out=dict(passed=True,rows=rows,contrasts=contrasts,provenance=provenance,protocol_sha256=sha(HERE/'bundle/protocol.json'),uncertainty='Descriptive paired10000-person bootstrap conditional on fixed fits; multiple comparisons, seed and normal-sample uncertainty not included; no best-arm selection or clinical claim')
(HERE/'comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print('COMPARISON_COMPLETE',len(provenance),'fits',len(rows),'rows')
