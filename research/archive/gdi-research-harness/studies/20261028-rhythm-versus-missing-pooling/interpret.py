import json,csv,hashlib
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(S/'features.npz',allow_pickle=False) as z:feat={k:z[k].copy() for k in z.files}
with np.load(S/'fold_models.npz',allow_pickle=False) as z:models={k:z[k].copy() for k in z.files}
rows=list(csv.DictReader((S/'predictions.csv').open(encoding='utf-8')));index={str(v):i for i,v in enumerate(feat['video_ids'])};ids=feat['patient_ids'];folds={r['video_id']:int(r['fold']) for r in rows}
def person(a):return np.array([a[ids==p].mean() for p in sorted(set(ids))])
output={};checks=[]
for method in ('rhythm','combined'):
    n=10 if method=='rhythm' else 522;terms=np.zeros((203,2,n))
    for v,i in index.items():
        prefix=method+'_fold'+str(folds[v])+'__'
        if method=='rhythm':front,last=feat['front_rhythm'][i],feat['last_rhythm'][i]
        else:front,last=np.r_[feat['front_baseline'][i],feat['front_rhythm'][i]],np.r_[feat['last_baseline'][i],feat['last_rhythm'][i]]
        terms[i]=models[prefix+'coef']*(last-front)[None,:]/models[prefix+'scaler_scale'][None,:]
        got={r['position']:np.array([float(r['prediction_L']),float(r['prediction_R'])]) for r in rows if r['video_id']==v and r['method']==method}
        error=float(np.max(np.abs(terms[i].sum(-1)-(got['last']-got['front']))));assert error<1e-8;checks.append(error)
    block=terms if method=='rhythm' else terms[:,:,512:]
    output[method]=dict(mean_absolute_total_change=float(person(np.abs(terms.sum(-1)).mean(1)).mean()),temporal_block_mean_absolute_change=float(person(np.abs(block.sum(-1)).mean(1)).mean()),feature_mean_absolute_terms={str(name):float(person(np.abs(block[:,:,j]).mean(1)).mean()) for j,name in enumerate(feat['feature_names'])})
    if method=='combined':output[method]['original_block_mean_absolute_change']=float(person(np.abs(terms[:,:,:512].sum(-1)).mean(1)).mean())
freq={}
for j,n in zip((0,3,6),('ankle_distance','right_knee','left_knee')):
    diff=np.abs(feat['last_rhythm'][:,j]-feat['front_rhythm'][:,j]);freq[n]=dict(records_changed=int((diff>0).sum()),record_fraction=float((diff>0).mean()),person_mean_fraction=float(person((diff>0).astype(float)).mean()),person_mean_abs_cycles_per_frame=float(person(diff).mean()))
result=dict(people=175,records=203,additional_fits=0,linear_decomposition_max_error=max(checks),condition_details=output,dominant_frequency_changes=freq,not_a_clinical_feature_importance_or_causal_fraction=True,source_sha256={p.name:sha(p) for p in (Path(__file__),S/'interpretation_plan.md',S/'features.npz',S/'fold_models.npz',S/'predictions.csv')})
with (S/'interpretation.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
