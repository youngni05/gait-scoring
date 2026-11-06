"""Audit real GPU pilot and write the pinned approval gate only after success."""
from pathlib import Path
import argparse,sys,tarfile,json,math
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'bundle'))
import run as r
import torch
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True);ap.add_argument('--sha256',required=True);a=ap.parse_args()
    archive=H/('pilot-review-'+a.job+'.tar.gz');dest=H/('received-pilot-'+a.job)
    assert r.sha(archive)==a.sha256
    with tarfile.open(archive) as tf:tf.extractall(dest,filter='data')
    assert r.sha(dest/'manifest.json')==r.sha(H/'bundle/manifest.json')
    r.setup(20261029,'cpu');d=r.data();c=r.load(H/'bundle/protocol.json')
    for check in ['synthetic','nested','windows','structure']:
        assert r.load(dest/('checks/%s-%s/verification.json'%(check,a.job)))['passed']
    assert r.load(dest/('checks/gpu-%s/check.json'%a.job))['windows']==11261
    p=dest/'runs/pilot';s=r.load(p/'pilot_summary.json')
    assert s['passed'] and s['pilot_only'] and s['epochs']==3 and not s['outer_evaluation_used']
    envs=[r.load(f) for f in p.glob('environment-*.json')]
    assert len(envs)==1 and envs[0]['job']==a.job and envs[0]['device']=='cuda'
    tr,va,te=r.split(d,0,1);steps=3*math.ceil(2*len(d['x'][tr].values)/c['batch_size'])
    assert [len(set(d['person_id'][m])) for m in (tr,va,te)]==[416,138,138]
    assert s['optimizer_steps']==steps and s['checkpoint_sha256']==r.sha(p/'fit/latest.pt')
    ck=torch.load(p/'fit/latest.pt',map_location='cpu',weights_only=True)
    ident=dict(bundle_sha256=r.sha(H/'bundle/manifest.json'),seed=c['seed']+1,epochs=3,outer_fold=0,inner_fold=1,
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist())
    for prefix,mask in [('train',tr),('validation',va)]:
        for field,key in [('x','x'),('y','y'),('ids','person_id')]:ident[prefix+'_'+field+'_sha']=r.digest_array(d[key][mask])
    assert ck['identity']==ident==r.load(p/'fit/identity.json')
    history=r.load(p/'fit/curve.json');assert history==ck['history'] and len(history)==3
    complete=r.load(p/'fit/fit_complete.json')
    assert complete['checkpoint_sha256']==r.sha(p/'fit/latest.pt') and complete['curve_sha256']==r.sha(p/'fit/curve.json') and complete['optimizer_steps']==steps
    for row in history:
        path=p/'fit/raw_predictions'/('epoch%04d.npz'%row['epoch']);assert r.sha(path)==row['raw_predictions_sha256']
        with np.load(path) as z:
            coef=r.calibration(z['train_raw'],d['y'][tr]);np.testing.assert_allclose(coef,row['calibration'],rtol=0,atol=1e-10)
            for mask,raw,prefix in [(tr,z['train_raw'],'train'),(va,z['validation_raw'],'validation')]:
                assert raw.shape==(int(mask.sum()),2)
                assert abs(r.metric(d['y'][mask],raw,d['person_id'][mask])-row[prefix+'_raw_person_mae'])<1e-10
                assert abs(r.metric(d['y'][mask],r.corrected(raw,coef),d['person_id'][mask])-row[prefix+'_person_mae'])<1e-10
    assert ck['lo']==float(np.float32(d['y'][tr].min())) and ck['span']==float(np.float32(np.ptp(d['y'][tr])))
    assert all(int(v['step'])==steps for v in ck['optimizer']['state'].values())
    m=r.new_model();m.load_state_dict(ck['model'],strict=True)
    train_raw=r.predict(m,d['x'][tr],ck['lo'],ck['span'],'cpu');coef=r.calibration(train_raw,d['y'][tr])
    raw=r.predict(m,d['x'][va],ck['lo'],ck['span'],'cpu');pred=r.corrected(raw,coef)
    with np.load(p/'validation_predictions.npz') as z:
        np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va))
        for key in ('y','person_id'):np.testing.assert_array_equal(z[key],d[key][va])
        assert abs(r.metric(z['y'],z['predictions'],z['person_id'])-s['validation_person_mae'])<1e-10
        raw_error=float(np.max(np.abs(raw-z['raw'])));pred_error=float(np.max(np.abs(pred-z['predictions'])))
    # Keep the original float32 cross-device result visible. Early OLS slope
    # amplifies a few ULPs; do not raise the tolerance or alter trained weights.
    float32_passed=raw_error<1e-4 and pred_error<1e-4
    m64=r.new_model().double();m64.load_state_dict(ck['model'],strict=True);m64.eval()
    def reference64(x):
        values=[]
        with torch.inference_mode():
            for start in range(0,len(x.values),64):
                t=torch.from_numpy(x.values[start:start+64]).double()
                values.append((m64(t)*ck['span']+ck['lo']).reshape(-1,2).numpy())
        return x.aggregate(np.concatenate(values))
    train64=reference64(d['x'][tr]);coef64=r.calibration(train64,d['y'][tr]);raw64=reference64(d['x'][va]);pred64=r.corrected(raw64,coef64)
    with np.load(p/'validation_predictions.npz') as z:
        raw64_error=float(np.max(np.abs(raw64-z['raw'])));pred64_error=float(np.max(np.abs(pred64-z['predictions'])))
    assert raw64_error<1e-4 and pred64_error<1e-4,(raw64_error,pred64_error)
    gpu=r.load(H/'gpu_checkpoint_replay.json')
    assert gpu['passed'] and gpu['checkpoint_sha256']==s['checkpoint_sha256'] and gpu['job_id']==a.job
    assert gpu['raw_max_error']==gpu['corrected_max_error']==gpu['train_raw_max_error']==0
    mean_seconds=float(np.mean(s['epoch_seconds']))
    ratios=[]
    base_steps=steps/3
    for folds in c['parallel_workers']:
        ratios.append(sum(math.ceil(2*len(d['x'][r.split(d,f,j)[0]].values)/c['batch_size'])/base_steps for f in folds for j in [None]+[k for k in range(5) if k!=f]))
    result=dict(passed=True,job_id=a.job,archive_sha256=a.sha256,epochs=3,optimizer_steps=steps,train_people=416,validation_people=138,
        train_windows=len(d['x'][tr].values),validation_windows=len(d['x'][va].values),outer_evaluation_used=False,
        float32_cpu_replay_passed=float32_passed,float32_raw_max_error=raw_error,float32_corrected_max_error=pred_error,
        float64_cpu_reference_passed=True,raw_cpu_replay_max_error=raw64_error,corrected_cpu_replay_max_error=pred64_error,tolerance=1e-4,
        gpu_checkpoint_replay_exact=True,reference_note='Float32 and float64 CPU differences both recorded; same learned weights, original tolerance unchanged.',
        validation_person_mae=s['validation_person_mae'],epoch_seconds=s['epoch_seconds'],peak_allocated_MiB=s['peak_allocated_bytes']/1024**2,
        rough_worker_max_training_hours=[mean_seconds*c['max_epochs']*v/3600 for v in ratios],
        time_estimate_note='Pilot-based extrapolation; assumes every refit uses 200 epochs, IO and node contention vary.',real_local_optimizer_steps=0)
    r.save(H/'gpu_pilot_verification.json',result)
    r.save(H/'approve-main.json',dict(approved=True,bundle_sha256=r.sha(H/'bundle/manifest.json'),pilot_summary_sha256=r.sha(p/'pilot_summary.json'),pilot_checkpoint_sha256=s['checkpoint_sha256'],local_verification_sha256=r.sha(H/'gpu_pilot_verification.json')))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
