"""Audit the received paired pilot without training or opening outer predictions."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'bundle'))
import torch,run as r
from audit import audit_fit

def main():
    h=H/'received-pilot-32985';u=h/'unpacked';out=H/'analysis/pilot-32985';out.mkdir(parents=True,exist_ok=True)
    receipt=r.load(h/'receipt.json')
    assert r.sha(h/'pilot-review.tar.gz')==receipt['archive_sha256']
    assert r.sha(h/'pilot-worker-log-1341.txt')==receipt['log_sha256']
    assert r.sha(u/'manifest.json')==r.sha(H/'bundle/manifest.json')
    gate=r.load(u/'control/approve-main-pair.json');assert gate['approved'] and gate['bundle_sha256']==r.sha(u/'manifest.json')
    for name,digest in gate['proof_sha256'].items():assert r.sha(u/name)==digest
    r.setup(7,'cpu');d=r.data();base=r.config();tr,va,te=r.split(d,0,1);rows={}
    for variant in ('window_control','record_mean'):
        c=dict(base,loss_variant=variant,name='cnn_'+variant+'_packed_records');folder=u/'runs/pilot'/variant/'pilot'
        curve=audit_fit(r,d,c,0,1,folder,3,'pilot_not_performance')
        ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=True)
        model=r.GdiCNN(c['dropout']);model.load_state_dict(ck['model']);model.eval()
        train=r.predict(model,d['x'][tr],ck['lo'],ck['span'],'cpu');valid=r.predict(model,d['x'][va],ck['lo'],ck['span'],'cpu')
        coef=r.calibrate(train,d['y'][tr]);pred=r.corrected(valid,coef)
        with np.load(folder/'raw_predictions/epoch0003.npz') as z:
            delta=float(np.max(abs(pred-r.corrected(z['validation_raw'],curve[-1]['calibration']))))
            raw_delta=float(np.max(abs(valid-z['validation_raw'])))
        assert delta<1e-4
        metric=float(np.mean([abs(pred[d['person_id'][va]==q]-d['y'][va][d['person_id'][va]==q]).mean() for q in np.unique(d['person_id'][va])]))
        assert abs(metric-curve[-1]['validation_person_mae'])<1e-4
        gpu=r.load(u/('checks/pilot-%s-gpu.json'%variant));cpu=r.load(u/('checks/pilot-%s-cpu.json'%variant))
        for report in (gpu,cpu):
            assert report['passed'] and report['checkpoint_sha256']==r.sha(folder/'latest.pt') and report['max_corrected_prediction_error']<1e-4
        summary=r.load(u/'runs/pilot'/variant/'pilot_summary.json')
        rows[variant]=dict(passed=True,epochs=3,optimizer_steps=r.load(folder/'fit_complete.json')['optimizer_steps'],
            train_people=len(np.unique(d['person_id'][tr])),train_records=int(tr.sum()),train_windows=len(d['x'][tr].values),validation_people=len(np.unique(d['person_id'][va])),
            corrected_validation_mae=curve[-1]['validation_person_mae'],raw_validation_mae=curve[-1]['validation_raw_person_mae'],calibration=curve[-1]['calibration'],
            local_cpu_max_corrected_error=delta,local_cpu_max_raw_error=raw_delta,server_cpu_max_corrected_error=cpu['max_corrected_prediction_error'],server_gpu_max_error=gpu['max_corrected_prediction_error'],
            peak_allocated_MiB=summary['peak_allocated_bytes']/1024**2,mean_epoch_seconds=summary['mean_epoch_seconds'],rough_max_training_hours=summary['rough_max_training_seconds']/3600,checkpoint_sha256=r.sha(folder/'latest.pt'))
        print(json.dumps({variant:rows[variant]}),flush=True)
    report=dict(passed=True,pilot_only=True,variants=rows,archive_sha256=receipt['archive_sha256'],all_six_epoch_metrics_verified=True,training_only_scaling_and_calibration=True,
        matching_people_splits_input=True,same_seed=True,outer_evaluation_used=False,full_OOF_comparison_performed=False,local_optimizer_steps=0,protected_final_data_opened=False)
    r.save(out/'verification.json',report);print('PILOT_RECEIPT_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
