"""Fixed10/20 point boundaries on existing verified person-held-out scores."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;S=H.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=read(S/'robust-aggregation692/analysis/comparison.json');p=Path(report['predictions_path']);assert sha(p)==report['predictions_sha256']
with np.load(p) as z:a={k:z[k].copy() for k in z.files}
ids=a['person_id'];people=np.unique(ids);result={}
for width in (10,20):
    boundaries=np.arange(0,161,width);truth=np.searchsorted(boundaries,a['y'],side='right');part={}
    for name in ['existing_CNN_STT','existing_mean_difference','inner_selected']:
        pred=np.searchsorted(boundaries,a[name],side='right');per=lambda v:float(np.mean([v[ids==q].mean() for q in people]))
        support=[]
        for k in np.unique(truth):
            rows=truth==k;person_weights=np.array([1/np.sum(ids==i) for i in ids])[:,None]/2
            denom=(person_weights*rows).sum();support.append(dict(band_index=int(k),independent_people=len(set(ids[rows.any(1)])),
                recall=float((person_weights*rows*(pred==truth)).sum()/denom)))
        part[name]=dict(exact=per(pred==truth),two_or_more=per(abs(pred-truth)>=2),band_recall=support)
    result[str(width)]=part
out=H/'checks/baselines.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(dict(passed=True,people=692,records=1061,boundaries='0..160 step10/20 with open tails',metrics=result,
    source=str(p),source_sha256=sha(p),new_training=False),indent=2)+'\n')
print(json.dumps({width:{k:v['exact'] for k,v in part.items()} for width,part in result.items()}))
