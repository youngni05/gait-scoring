"""Independent replay checks for the explanatory calculations."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;OUT=HERE/'diagnosis'
sys.path.insert(0,str(BASE/'feature-analysis'));import study
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((OUT/'summary.json').read_text());assert d['passed']
for p,h in d['source_sha256'].items():assert sha(BASE/p)==h
a=np.load(OUT/'arrays.npz');v=np.load(HERE/'scores.npz');records=json.loads((OUT/'leaf-records.json').read_text());data=np.load(BASE/'data.npz');keys=a['keys'].tolist()
assert a['label_mass'].shape==(30,10,4) and a['contributions'].shape==(30,10,128) and len(records)==300
pred=a['label_mass']@np.arange(4);ci=[list(v['keys']).index(k) for k in keys]
leaf_error=float(abs(pred-v['predictions'][:,ci]).max());assert leaf_error<=1e-10
roots=np.array([r['root_mean'] for r in records]).reshape(30,10)
path_error=float(abs(roots+a['contributions'].sum(2)-pred).max());assert path_error<=1e-10
camera=pred[:,1::2]-pred[:,::2]
assert abs(camera.mean()-sum(r['mean_camera_contribution'] for r in d['camera_attribution']['statistics']))<1e-10
assert abs(camera.mean()-sum(r['mean_camera_contribution'] for r in d['camera_attribution']['joints']))<1e-10
oof_errors=[]
for f in range(6):
    tr=data['person'][data['fold']!=f];te=data['person'][data['fold']==f];assert not set(tr)&set(te)
    for si,s in enumerate(study.SEEDS):
        z=np.load(BASE/'feature-analysis/fits/motion3d'/('fold%d_seed%d.npz'%(f,s)))
        oof_errors.append(float(abs(a['oof'][si,z['indices']]-z['prediction']).max()))
assert max(oof_errors)<=1e-12
# Recompute a complete training-fold RGB five-second distance comparison using
# direct squared differences rather than the pairwise-distance implementation.
f=0;x=a['x5'];valid=np.isfinite(x).all(1)&(data['cohort']!='BMCLab');tr=np.flatnonzero(valid&(data['fold']!=f));te=np.flatnonzero(valid&(data['fold']==f))
w=study.weights(data['person'][tr]);mu=np.average(x[tr],axis=0,weights=w);sd=np.maximum(np.sqrt(np.average((x[tr]-mu)**2,axis=0,weights=w)),1e-6)
xt=(x[tr]-mu)/sd;xe=(x[te]-mu)/sd;xv=(v['features'][ci]-mu)/sd
reference=np.array([np.sqrt(np.mean((xt-r)**2,axis=1)).min() for r in xe]);video=np.array([np.sqrt(np.mean((xt-r)**2,axis=1)).min() for r in xv]);wt=study.weights(data['person'][te])
nn=json.loads((OUT/'nearest.json').read_text());nn_error=[];percentile_error=[]
for i,k in enumerate(keys):
    r=next(r for r in nn if r['fold']==f and r['window']=='first5s' and r['subset']=='RGB' and r['key']==k)
    nn_error.append(abs(video[i]-r['nearest_distance']));percentile_error.append(abs(np.average(reference<=video[i],weights=wt)-r['heldout_distance_percentile']))
assert max(nn_error)<=1e-10 and max(percentile_error)<=1e-12
heading=json.loads((OUT/'heading.json').read_text());clips=json.loads((HERE/'clips.json').read_text());heading_errors=[]
for r,c in zip(heading['rows'],clips[::2]):
    f=Path('D:/gait-research-data/MoVi/experiments/walking-expansion')/('subject-%02d'%c['subject'])/'heavy/PG1_800_comparison.npz';assert sha(f)==r['reference_sha256']
    z=np.load(f);m=(z['frames']>=c['frames_one_based'][0])&(z['frames']<=c['frames_one_based'][1]);ref=z['reference_3d'][m];vec=ref[:,7,:2]-ref[:,6,:2];yaw=np.unwrap(np.arctan2(vec[:,1],vec[:,0]))*180/np.pi
    heading_errors.append(abs(np.quantile(yaw,.95)-np.quantile(yaw,.05)-r['heading_p95_p05_deg']))
assert max(heading_errors)<=1e-10
result=dict(passed=True,RF_models=30,video_people=5,video_clips=10,leaf_reconstruction_max_error=leaf_error,path_reconstruction_max_error=path_error,CARE_oof_replay_max_error=max(oof_errors),direct_nearest_distance_max_error=float(max(nn_error)),reference_percentile_max_error=float(max(percentile_error)),heading_recalculation_max_error=float(max(heading_errors)),summary_sha256=sha(OUT/'summary.json'),diagnostic_sources_sha256={n:sha(OUT/n) for n in ['arrays.npz','distributions.json','nearest.json','leaf-records.json','heading.json']},limits='Exact accounting and observational distribution checks; no causal allocation of HMR2 vs population/turns, no new clinical accuracy or corrected-camera claim.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
