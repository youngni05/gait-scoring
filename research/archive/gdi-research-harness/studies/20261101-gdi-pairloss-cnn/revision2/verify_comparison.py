"""Independent weighted reconstruction of the final two-arm report; no selection."""
from pathlib import Path
import json,math,hashlib
from collections import Counter
import numpy as np
H=Path(__file__).resolve().parent
AUDIT=H.parents[1]/'20261101-gdi-longitudinal-audit'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=read(H/'comparison.json');receipt=read(H/'checks/full-receipt.json');specs=read(AUDIT/'prepared-pairs.json')['pairs']
    metadata=read(H/'seed0/visit_metadata.json');counts=Counter(r['person_id'] for r in specs);errors=[]
    assert len(counts)==229 and len(specs)==581
    for arm in (0,1):
        rec=next(r for r in receipt['seeds'] if r['seed']==arm);path=Path(rec['extracted'])/'runs/main/oof_predictions.npz'
        assert sha(path)==report['source_sha256'][str(path)]
        with np.load(path) as z:
            y=z['y'];p=z['predictions'][:,0];ids=z['person_id'];vids=z['video_id'];fold=z['fold']
        index={v:i for i,v in enumerate(vids)};terms=[]
        for r in specs:
            a,b=index[r['lower_age_video_ids'][0]],index[r['higher_age_video_ids'][0]]
            assert ids[a]==ids[b]==r['person_id'] and fold[a]==fold[b]
            for s in (0,1):terms.append((r['person_id'],s,float(y[b,s]-y[a,s]),float(p[b,s]-p[a,s])))
        for side in (None,0,1):
            ts=[r for r in terms if side is None or r[1]==side];ns=2 if side is None else 1
            mae=math.fsum(abs(t-pred)/(len(counts)*ns*counts[person]) for person,s,t,pred in ts)
            changed=Counter(person for person,s,t,pred in ts if t!=0)
            direction=math.fsum((.5 if pred==0 else float((t>0)==(pred>0)))/(len(changed)*changed[person]) for person,s,t,pred in ts if t!=0)
            expected=report['arms']['arm%d'%arm]['change' if side is None else 'sides']
            if side is not None:expected=expected['left' if side==0 else 'right']
            errors.extend([abs(mae-expected['person_mean_absolute_delta_error']),abs(direction-expected['person_mean_direction_agreement'])])
        first=[]
        for person in sorted(set(ids)):
            ix=[int(i) for i in np.flatnonzero(ids==person)]
            first.append(ix[0] if len(ix)==1 else sorted(ix,key=lambda i:metadata[i]['age'])[0])
        first_error=math.fsum(abs(float(p[i,s]-y[i,s]))/(692*2) for i in first for s in (0,1))
        errors.append(abs(first_error-report['arms']['arm%d'%arm]['first_available_or_lower_age_visit_mae']))
        all_error=math.fsum(float(abs(p[ids==person]-y[ids==person]).mean())/692 for person in set(ids))
        errors.append(abs(all_error-report['arms']['arm%d'%arm]['all_visit_mae']))
    assert max(errors)<1e-10
    result=dict(passed=True,maximum_metric_difference=max(errors),tolerance=1e-10,training_or_inference_rerun=False,comparison_sha256=sha(H/'comparison.json'),script_sha256=sha(Path(__file__)))
    (H/'comparison-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
