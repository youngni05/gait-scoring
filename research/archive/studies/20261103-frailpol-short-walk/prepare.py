from signal_pilot import HERE,BIG,save
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=read(HERE/'protocol.json');assert sha(HERE/'extract.py')==p['extraction']['code_sha256']
    f=read(BIG/'features.json');assert len(f)==668 and len(set(r['id'] for r in f))==668
    a=pd.read_excel(BIG/'raw/participants.xlsx');a.columns=a.columns.str.strip();a=a[a.PID.isin([r['id'] for r in f])].set_index('PID');assert len(a)==668 and a.index.is_unique
    b=pd.read_csv(BIG/'raw/gait_parameters_3_class.csv').set_index('participant_id');assert len(b)==668
    items=['Weight loss (0/1)','Weakness (0/1)','Slowness (0/1)','Physical Activity (0/1)','Muscular strength (0/1)'];rows=[]
    ids=np.array(sorted(a.index),int);folds={}
    for k,(_,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=20261103).split(ids,a.loc[ids,'Targets'])):
        for i in ids[te]:folds[int(i)]=k
    byid={r['id']:r for r in f}
    for pid in ids:
        r=a.loc[pid];assert int(r.Targets)==int(b.loc[pid,'Targets'])
        q=r[items].to_numpy(float);complete=bool(np.isfinite(q).all());consistent=bool(complete and sum(q)==r.FI)
        t=float(r['TUG time (s)']);t=t if np.isfinite(t) and t>0 else None
        rows.append(dict(id=int(pid),fold=folds[int(pid)],age=float(r.Age),tug=t,frailty=int(r.Targets),FI=int(r.FI),items=[float(v) if np.isfinite(v) else None for v in q],component_consistent=consistent,non_slowness=float(q[[0,1,3,4]].sum()) if consistent else None,status=byid[int(pid)]['status'],reason=byid[int(pid)].get('reason'),features=byid[int(pid)].get('features'),window=byid[int(pid)].get('window')))
    save(BIG/'cohorts.json',rows)
    groups={}
    for c in range(3):
        rr=[r for r in rows if r['frailty']==c];groups[str(c)]=dict(total=len(rr),measurable=sum(r['status']=='ok' for r in rr))
    reasons={s:sum(r.get('reason')==s for r in rows) for s in set(r.get('reason') for r in rows) if s}
    ok=[r for r in rows if r['status']=='ok'];dur=np.array([r['window']['support_seconds'] for r in ok])
    audit=dict(total=668,measurable=len(ok),age_range=[min(r['age'] for r in rows),max(r['age'] for r in rows)],aged65_total=sum(r['age']>=65 for r in rows),aged65_measurable=sum(r['age']>=65 for r in ok),by_frailty=groups,reasons=reasons,missing_tug=[r['id'] for r in rows if r['tug'] is None],inconsistent_components=[{k:r[k] for k in ['id','FI','items']} for r in rows if not r['component_consistent']],target_n={t:sum(r[t] is not None for r in ok) for t in ['tug','frailty','non_slowness']},support_seconds=dict(min=float(dur.min()),median=float(np.median(dur)),max=float(dur.max())),features=list(ok[0]['features']))
    save(HERE/'data-audit.json',audit);save(HERE/'preparation-receipt.json',dict(protocol_sha256=sha(HERE/'protocol.json'),cohorts_sha256=sha(BIG/'cohorts.json'),features_sha256=sha(BIG/'features.json'),weights_manifest_sha256=sha(HERE/'weights-manifest.json')));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
