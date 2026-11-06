"""Conservative sensitivity: WITHHOLD unverified times, not verified zero time."""
from audit import *
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ALPHAS=[.1,1.,10.,100.]
def save(n,v):
    p=HERE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def fit(x,y,tr,te,alpha,view):
    xx=x[:,:-1];offset=x[:,-1];target=y-offset
    sc=StandardScaler().fit(xx[tr]);model=Ridge(alpha=alpha,solver='svd').fit(sc.transform(xx[tr]),target[tr]);part=model.predict(sc.transform(xx[te]))
    return np.maximum(part,0)+offset[te],part+offset[te],(sc,model)
def prepare():
    assert not (HERE/'protocol.json').exists()
    annotations=json.loads((HERE/'timing-annotations.json').read_text());assert all(r['clinical_timing_status']=='unverified' and r['confirmed_preparation_intervals'] is None for r in annotations)
    cohorts={};sources=[]
    for c in ['exclude283','include283']:
        out=HERE/c;out.mkdir(exist_ok=True);d=np.load(PRIOR/c/'input.npz');views={}
        for b in [10,20,30]:
            for kind,prior in [('Gait','Gait'),('Withheld','Reallocated'),('Candidate','Prep')]:views[f'{kind}_{b}']=d[f'{prior}_{b}'].copy()
        np.savez_compressed(out/'input.npz',ids=d['ids'],y=d['y'],**views)
        for n in ['manifest.json','splits.json']:(out/n).write_bytes((PRIOR/c/n).read_bytes())
        cohorts[c]={'people':len(d['y']),'long_people':int(sum(d['y']>=20)),'views':list(views),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json']}}
        sources.extend(PRIOR/c/n for n in ['input.npz','manifest.json','splits.json','predictions.csv'])
    protocol={'cohorts':cohorts,'scope':'CONSERVATIVE SENSITIVITY ONLY; actual requested confirmed-preparation experiment lacks verified timer/trial evidence for83 people',
      'mapping':{'Gait':'previous Gait','Withheld':'previous Reallocated; same prep+gait observed clips, no unverified prep OR core rise duration is added or used as feature','Candidate':'previous Prep; all automatically observed prep+rise duration is added'},
      'observation':'exact same previous per-person clips and total durations,10/20/30 caps; Withheld and Candidate identical clipped speed and observed gait stop; only candidate time addition differs',
      'split':'byte-identical previous82/83 person folds; no re-selection of people or records',
      'model':'Ridge SVD .1,1,10,100 chosen within3inner folds using finalTUG MAE;5partitionsx5outer; training-only scaling and residual target; no evaluation label timing corrections',
      'semantics':'withholding uncertain time is a policy, not assertion that true timed preparation=0; core rise proxy is also not independently timed annotation',
      'not_new_independent_validation':'reruns previous relevant conditions for explicit conservative audit; results expected to reproduce, not new independent clinical evidence',
      'code_sha256':sha(HERE/'study.py'),'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in sources}}
    save('protocol.json',protocol);print(json.dumps(cohorts),flush=True)
if __name__=='__main__':prepare()
