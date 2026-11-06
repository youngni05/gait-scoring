"""Audit completed GPU pilot on CPU without additional optimization or outer inference."""
from pathlib import Path
import sys,json,tarfile,math
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'bundle'))
import run
import torch

def main():
    archive=HERE/'pilot-review-31964.tar.gz';r=HERE/'received-31964'
    assert run.sha(archive)=='efa37d6d39031f111aaafe7270e0dd181ec2ffa037a2c3bd89706b3efc68ed4d'
    with tarfile.open(archive) as tf:tf.extractall(r,filter='data')
    assert run.sha(r/'manifest.json')==run.sha(HERE/'bundle/manifest.json')
    run.setup(20261029,'cpu');d=run.data();c=run.load(HERE/'bundle/protocol.json')
    p=r/'runs/pilot';s=run.load(p/'pilot_summary.json');env=run.load(p/'environment.json')
    assert s['passed'] and s['pilot_only'] and not s['outer_evaluation_used'] and s['epochs']==3
    assert env['job']=='31964' and env['device']=='cuda' and env['hostname']=='DIS03'
    assert env['torch']=='2.4.0+cu121' and env['gpu']=='NVIDIA RTX A6000'
    assert 'COMPLETED' in (r/'logs/accounting-31964.txt').read_text()
    assert 'STT_PILOT_FINISHED' in (r/'logs/pilot-31964.log').read_text()
    assert run.load(r/'checks/gpu-31964/check.json')['passed']
    synthetic=run.load(r/'checks/synthetic-31964/verification.json')
    assert synthetic['passed'] and synthetic['exact_model_optimizer_scheduler_resume']
    assert synthetic['validation_input_and_label_changes_do_not_change_fit']
    tr,va,te=run.split(d,0,1)
    assert [len(set(d['person_id'][m])) for m in (tr,va,te)]==[416,138,138]
    assert s['optimizer_steps']==3*math.ceil(2*int(tr.sum())/c['batch_size'])==30
    assert run.sha(p/'fit/latest.pt')==s['checkpoint_sha256']
    ck=torch.load(p/'fit/latest.pt',map_location='cpu',weights_only=True)
    ident=dict(bundle_sha256=run.sha(HERE/'bundle/manifest.json'),seed=c['seed']+1,epochs=3,outer_fold=0,inner_fold=1,
        train_people=np.unique(d['person_id'][tr]).tolist(),validation_people=np.unique(d['person_id'][va]).tolist(),outer_evaluation_people=np.unique(d['person_id'][te]).tolist(),
        train_x_sha=run.digest_array(d['x'][tr]),train_y_sha=run.digest_array(d['y'][tr]),train_ids_sha=run.digest_array(d['person_id'][tr]))
    assert ck['identity']==ident
    h=run.load(p/'fit/curve.json');assert h==ck['history'] and [v['epoch'] for v in h]==[1,2,3]
    assert ck['lo']==float(np.float32(d['y'][tr].min()))
    assert ck['span']==float(np.float32(d['y'][tr].max()-d['y'][tr].min()))
    m=run.new_model();m.load_state_dict(ck['model'],strict=True)
    assert sum(q.numel() for q in m.parameters())==s['parameters']==27022
    assert all(int(state['step'])==30 for state in ck['optimizer']['state'].values())
    train_raw=run.predict(m,d['x'][tr],ck['lo'],ck['span'],'cpu')
    coef=run.calibration(train_raw,d['y'][tr])
    raw=run.predict(m,d['x'][va],ck['lo'],ck['span'],'cpu');pred=raw*coef[0]+coef[1]
    with np.load(p/'validation_predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va))
        for k in ('y','person_id'):np.testing.assert_array_equal(z[k],d[k][va])
        saved_mae=run.metric(z['y'],z['predictions'],z['person_id'])
        assert abs(saved_mae-s['validation_person_mae'])<1e-10
        raw_error=float(np.max(np.abs(raw-z['raw'])))
        pred_error=float(np.max(np.abs(pred-z['predictions'])))
    assert raw_error<1e-4 and pred_error<1e-4,(raw_error,pred_error)
    run.setup(ident['seed'],'cpu');initial=run.new_model()
    changed=sum(not torch.equal(value,initial.state_dict()[key]) for key,value in ck['model'].items())
    assert changed>0
    cap=run.load(r/'logs/capacity-after-31964.json');assert cap['total_free']>=1
    result=dict(passed=True,job='31964',state='COMPLETED',pilot_only=True,outer_evaluation_used=False,
        archive_sha256=run.sha(archive),checkpoint_sha256=run.sha(p/'fit/latest.pt'),
        train_people=416,validation_people=138,untouched_outer_people=138,epochs=3,real_optimizer_steps=30,
        raw_cpu_replay_max_error=raw_error,prediction_cpu_replay_max_error=pred_error,
        training_only_scale_calibration_recomputed=True,changed_parameter_tensors=changed,
        validation_person_mae=saved_mae,peak_allocated_MiB=s['peak_allocated_bytes']/1024**2,
        epoch_seconds=s['epoch_seconds'],calibration=h[-1]['calibration'],
        free_GPU_at_submission=4,free_GPU_after_allocation=3,free_GPU_after_completion=cap['total_free'],
        full_training_submitted=False,performance_comparison_not_valid_from_three_epoch_inner_pilot=True)
    run.save(HERE/'gpu_pilot_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
