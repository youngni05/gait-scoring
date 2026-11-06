"""Prespecified follow-up on existing straight clips, retain mixed-window results."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import video_measurement as v
HERE=v.HERE;BIG=v.BIG
def freeze():
    assert not(HERE/'straight-cycle-protocol.json').exists()
    v.save(HERE/'straight-cycle-protocol.json',dict(code_sha256=v.sha(__file__),measurement_code_sha256=v.sha(Path(v.__file__)),
      motivation='Initial4sec fixedwindows include turns; retain them and audit existing14straight2sec windows separately. This is a declared exploratoryfollowup after seeing mixedwindowresults,not replacementprimaryanalysis.',
      selection='Use all28 previouslyfrozen14person2camerabothstraightclips,nonechosenbymeasurementaccuracy. Sameheightonlyimageestimator+spectralperiod andindependent3Dkinematicpeakreference. Existingqualityruleunchanged; noevents=>unmeasurable.',
      evaluation='Speed,kinematiccycle,kinematicstrideproxy,coverage,bias,p95,personbootstrap5000. No expert/forceplateHSreference,so cannotclaimclinicalstep/stridelengthaccuracy.' ))
def run():
    p=json.loads((HERE/'straight-cycle-protocol.json').read_text());assert p['code_sha256']==v.sha(__file__) and p['measurement_code_sha256']==v.sha(Path(v.__file__))
    sys.path.insert(0,str(v.CAD));import study as cadence
    source=pd.read_csv('D:/Projects/D_P/gdi-research-harness/studies/20261101-camera-height-speed/main-clips.csv');source=source[source.condition=='straight'];rows=[]
    for _,r in source.iterrows():
        path=v.HEIGHT/(r['key']+'.npz');assert v.sha(path)==r.array_sha256;z=np.load(path);lm=z['lm'];xy=lm[:,:,:2]*[800,600]
        speed=float(np.linalg.norm(v.slope(xy[:,[23,24]].mean(1)))*float(z['height'])/np.median(np.ptp(xy[:,:,1],axis=1)))
        per=cadence.estimate(xy[:,27]-xy[:,28]);cycle=120/per['cadence'] if per['quality'] else None;truth=v.kin_reference(z['reference'])
        rows.append(dict(subject=int(r.subject),camera=r.camera,speed=speed,cycle=cycle,stride=speed*cycle if cycle is not None else None,reference=truth,quality=per))
    summaries=[]
    for cam in ['PG1','PG2']:
      for field in ['speed','cycle','stride']:
        valid=[r for r in rows if r['camera']==cam and r[field] is not None and r['reference'][field] is not None]
        summaries.append(dict(camera=cam,metric=field,eligible=14,**v.metrics([r['reference'][field] for r in valid],[r[field] for r in valid])))
    v.save(HERE/'straight-cycle-results.json',dict(estimates=rows,metrics=summaries,scope='kinematic3Dproxy forcycle/stride,notclinicalHSgroundtruth; exploratoryfollowup'))
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':{'freeze':freeze,'run':run}[sys.argv[1]]()
