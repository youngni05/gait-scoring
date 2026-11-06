from prepare import *
from scipy.stats import rankdata
from collections import Counter

def corr(a,b):
 a=np.asarray(a);b=np.asarray(b)
 if len(a)<4 or np.ptp(a)==0 or np.ptp(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def rho(a,b):return corr(rankdata(a),rankdata(b))
def partial(a,b,speed):
 design=np.column_stack([np.ones(len(speed)),rankdata(speed)])
 a=rankdata(a);b=rankdata(b);a=a-design@np.linalg.lstsq(design,a,rcond=None)[0];b=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
 return corr(a,b)
def describe(v):
 v=np.asarray(v);return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),min_max=[float(v.min()),float(v.max())])
def interval(values):
 v=[x for x in values if x is not None and np.isfinite(x)];return dict(ci95=np.quantile(v,[.025,.975]).tolist() if v else None,valid=len(v))
def main():
 rows=read(HERE/'rows.json');people=read(HERE/'people.json');meta={p['id']:p for p in people};fp=np.load(BIG/'frozen-predictions.npz');features=np.load(BIG/'features.npz');owner=features['owner'];starts=features['start'];new=np.load(BIG/'new-model-oof.npz');newids=list(new['person_id']);models=['fusion','CNN_bag','CNN_refit'];primary={};perperson={};trial=[]
 for m in models:
  primary[m]={}
  for r,row in enumerate(rows):
   ix=np.flatnonzero(owner==r)
   if not len(ix):continue
   value=fp[m][ix].mean(axis=0)
   if row['primary']:
    primary[m].setdefault(row['person'],[]).append(float(value.mean()))
    if m=='fusion':trial.append(dict(person=row['person'],member=row['member'],windows=len(ix),left_right=value.tolist()))
  perperson[m]={p:float(np.mean(v)) for p,v in primary[m].items()}
 clinical={normid(p['id']):p for p in read(BASE/'speed-control/participants.json')}
 ids=sorted(set(perperson['fusion'])&set(clinical)&set(newids));ix=np.array([newids.index(p) for p in ids]);speed=np.array([clinical[p]['speed']['mean_mps'] for p in ids]);poma=np.array([clinical[p]['POMA'] for p in ids])
 compare={'people':ids,'n':len(ids),'speed_vs_POMA':rho(speed,poma),'models':{}}
 arrays={m:np.array([perperson[m][p] for p in ids]) for m in models};arrays['new_age65plus']=-new['age65plus_prediction'][ix];arrays['new_allhealthy']=-new['allhealthy_prediction'][ix];arrays['direct3D_fold_reference']=-new['age65plus_truth'][ix]
 rng=np.random.default_rng(20261103);draws=rng.integers(0,len(ids),(2000,len(ids))) if ids else []
 for m,v in arrays.items():
  entry={'higher_is_better':True,'POMA_spearman':rho(v,poma),'speed_spearman':rho(v,speed),'POMA_partial_spearman_speed':partial(v,poma,speed),'POMA_spearman_bootstrap':interval([rho(v[j],poma[j]) for j in draws]),'POMA_partial_bootstrap':interval([partial(v[j],poma[j],speed[j]) for j in draws])}
  for outcome in ['FAC','TIS']:entry[outcome+'_spearman']=rho(v,[clinical[p][outcome] for p in ids])
  compare['models'][m]=entry
 delta=[]
 for j in draws:
  a=partial(arrays['new_age65plus'][j],poma[j],speed[j]);b=partial(arrays['fusion'][j],poma[j],speed[j]);delta.append(None if a is None or b is None else a-b)
 compare['new_minus_fusion_partial_rho']={'value':compare['models']['new_age65plus']['POMA_partial_spearman_speed']-compare['models']['fusion']['POMA_partial_spearman_speed'],'bootstrap':interval(delta)}
 lookup={(r['person'],r['member'],r['camera'],r['toe_offset_mm']):i for i,r in enumerate(rows)};sensitivity=[]
 for camera,offset in [(c,o) for c in proj.CONDITIONS for o in [0,30,60] if (c,o)!=('side_pan',0)]:
  differences={};matches=[]
  for key,r0 in lookup.items():
   p,member,cam,toe=key
   if (cam,toe)!=('side_pan',0) or (p,member,camera,offset) not in lookup:continue
   r1=lookup[p,member,camera,offset];a={int(starts[j]):int(j) for j in np.flatnonzero(owner==r0)};b={int(starts[j]):int(j) for j in np.flatnonzero(owner==r1)};shared=sorted(set(a)&set(b))
   if not shared:continue
   diff=float(np.mean([fp['fusion'][b[s]].mean()-fp['fusion'][a[s]].mean() for s in shared]));differences.setdefault(p,[]).append(diff);matches.append(dict(person=p,member=member,common_starts=shared,difference=diff))
  pdiff=[np.mean(v) for v in differences.values()]
  sensitivity.append(dict(camera=camera,toe_offset_mm=offset,people=len(pdiff),records=len(matches),signed_difference=describe(pdiff) if pdiff else None,absolute_difference=describe(np.abs(pdiff)) if pdiff else None,matches=matches))
 scores=[]
 for p in sorted(set(perperson['fusion'])|set(newids)):
  entry={**meta.get(p,{}),'id':p};entry['frozen']={m:perperson[m].get(p) for m in models}
  if p in newids:
   j=newids.index(p);entry['new_deviation']=float(new['age65plus_prediction'][j]);entry['true_deviation']=float(new['age65plus_truth'][j]);entry['fold']=int(new['fold'][j])
  if p in clinical:entry['clinical']=clinical[p]
  scores.append(entry)
 distributions={g:describe([v for p,v in perperson['fusion'].items() if p in meta and meta[p]['cohort']==g and meta[p]['age']>=65]) for g in ['healthy','stroke']}
 result=dict(protocol_sha256=sha(HERE/'protocol.json'),frozen_prediction_sha256=sha(BIG/'frozen-predictions.npz'),new_prediction_sha256=sha(BIG/'new-model-oof.npz'),frozen_people=len(perperson['fusion']),new_model_people=len(newids),older_frozen_distributions=distributions,clinical_comparison=compare,sensitivity=sensitivity,original_GDI_accuracy_evaluated=False,synthetic_only=True)
 save(HERE/'scores.json',scores);save(HERE/'trial-predictions.json',trial);save(HERE/'results.json',result)
 print(json.dumps({k:v for k,v in result.items() if k!='sensitivity'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

