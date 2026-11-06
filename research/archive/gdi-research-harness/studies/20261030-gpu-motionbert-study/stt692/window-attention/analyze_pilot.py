"""Audit received paired pilot artifacts and summarize inner-validation diagnostics only."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile
import numpy as np
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H/'bundle'))
import run as r
from bags import audit_weights,weight_summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extract',action='store_true');a=ap.parse_args()
    receipt=r.load(H/'received/transfer.json')
    for name,digest in receipt['sha256'].items():assert r.sha(H/'received'/name)==digest,name
    dest=H/'received/unpacked'
    if a.extract:
        with tarfile.open(H/'received/pilot-review.tar.gz') as tf:tf.extractall(dest,filter='data')
        assert r.sha(dest/'manifest.json')==r.sha(H/'bundle/manifest.json')
        assert r.sha(dest/'protocol.json')==r.sha(H/'bundle/protocol.json')
        print('Received archive hashes and frozen protocol verified');return
    d=r.data();tr,va,te=r.split(d,0,1);pair=r.load(dest/'runs/pilot/pair_summary.json')
    assert pair['passed'] and pair['pilot_only'] and not pair['full_training_started']
    assert r.load(H/'received/worker-pilot.json')['phase']=='pilot_complete'
    assert receipt['accounting']['state']=='COMPLETED' and receipt['accounting']['exit_code']=='0:0'
    assert 'POOLING_PAIR_PILOT_FINISHED' in (H/'received/worker-32854.log').read_text()
    rows={};identity=[];curves={};preds={}
    for mode in ['mean','attention']:
        p=dest/'runs/pilot'/mode;s=r.load(p/'pilot_summary.json');assert s==pair['variants'][mode]
        assert s['epochs']==3 and s['optimizer_steps']==240 and not s['outer_evaluation_used']
        envs=[r.load(v) for v in p.glob('environment-*.json')];assert len(envs)==1 and envs[0]['job']=='32854' and envs[0]['device']=='cuda'
        for folder in ['gpu-pooling','synthetic-'+mode,'replay-'+mode]:assert r.load(dest/'checks'/folder/'verification.json')['passed']
        cpu=r.load(H/'analysis'/('cpu-'+mode)/'verification.json');assert cpu['passed']
        gpu=r.load(dest/'checks'/('replay-'+mode)/'verification.json');identity.append(r.load(p/'fit/identity.json'))
        curves[mode]=r.load(p/'fit/curve.json')
        with np.load(p/'validation_predictions.npz') as z:
            for key in ['person_id','y']:np.testing.assert_array_equal(z[key],d[key][va])
            np.testing.assert_array_equal(z['row_index'],np.flatnonzero(va));preds[mode]=z['predictions'].copy()
            raw_mae=r.metric(z['y'],z['raw'],z['person_id']);corrected_mae=r.metric(z['y'],z['predictions'],z['person_id'])
            assert abs(corrected_mae-s['validation_person_mae'])<1e-10
            with np.load(p/'window_weights.npz') as wd:
                audit_weights(d['x'][va],wd,z['raw'],mode);weight_report=weight_summary(d['x'][va],wd)
                saved_summary=r.load(p/'weight_summary.json')
                exact_cpu_summary=True
                try:assert weight_report==saved_summary
                except AssertionError:
                    # Preserve the failed cross-NumPy exact check. Independently
                    # demand the same exact criterion in the source environment.
                    exact_cpu_summary=False
                    server=r.load(H/'analysis/weight-summary-replay.json')['variants'][mode]
                    assert server['passed'] and server['exact_match']
                    assert server['weights_sha256']==r.sha(p/'window_weights.npz') and server['summary_sha256']==r.sha(p/'weight_summary.json')
                    assert server['summary']==saved_summary
                summary_diff={k:weight_report[k]-v for k,v in saved_summary.items() if isinstance(v,(int,float))}
                deviation=float(abs(wd['weights']-1/d['x'][va].counts[wd['window_record'],None]).max())
        with np.load(p/'fit/raw_predictions/epoch0003.npz') as raw_epoch:
            train_raw=raw_epoch['train_raw'];vr=raw_epoch['validation_raw']
            raw_diagnostics=dict(training_raw_std=float(train_raw.std()),training_target_std=float(d['y'][tr].std()),training_raw_target_correlation=float(np.corrcoef(train_raw.reshape(-1),d['y'][tr].reshape(-1))[0,1]),validation_raw_min=float(vr.min()),validation_raw_max=float(vr.max()))
        rows[mode]=dict(raw_inner_person_mae=raw_mae,corrected_inner_person_mae=corrected_mae,calibration=curves[mode][-1]['calibration'],epoch_seconds=s['epoch_seconds'],mean_epoch_seconds=float(np.mean(s['epoch_seconds'])),peak_MiB=s['peak_allocated_bytes']/1024**2,parameters=s['parameters'],optimizer_steps=s['optimizer_steps'],weight_diagnostics=weight_report,max_weight_deviation_from_uniform=deviation,gpu_replay=gpu,cpu_replay=cpu,raw_diagnostics=raw_diagnostics,exact_cpu_weight_summary_passed=exact_cpu_summary,cpu_weight_summary_difference=summary_diff)
    for key in ['seed','epochs','outer_fold','inner_fold','train_people','validation_people','outer_evaluation_people','train_x_sha','train_y_sha','train_ids_sha','validation_x_sha','validation_y_sha','validation_ids_sha']:assert identity[0][key]==identity[1][key],key
    assert set(identity[0]['train_people']).isdisjoint(identity[0]['validation_people'])
    assert set(identity[0]['outer_evaluation_people']).isdisjoint(set(identity[0]['train_people'])|set(identity[0]['validation_people']))
    result=dict(passed=True,job_id='32854',pilot_only=True,main_training_started=False,cohort_people=692,cohort_records=1061,windows=11261,
        train_people=len(set(d['person_id'][tr])),train_records=int(tr.sum()),train_windows=len(d['x'][tr].values),validation_people=len(set(d['person_id'][va])),validation_records=int(va.sum()),validation_windows=len(d['x'][va].values),outer_people=len(set(d['person_id'][te])),outer_evaluation_used=False,
        rows=rows,attention_minus_mean_inner_mae=rows['attention']['corrected_inner_person_mae']-rows['mean']['corrected_inner_person_mae'],training_mean_constant_inner_mae=r.metric(d['y'][va],np.full_like(d['y'][va],d['y'][tr].mean()),d['person_id'][va]),
        transfer_sha256_verified=True,all6_epoch_metrics_verified=True,matched_people_input_order_and_seed=True,
        same_initial_encoder_and_prediction=r.load(H/'checks/pooling/verification.json')['zero_gate_equals_mean_exact'],
        notes=['Three-epoch inner-validation pilot only; no outer-fold or full-OOF ranking.', 'GDI values are pediatric public-data references, not older-adult functional or clinical validation.', 'Window weights describe model use, not established visibility or clinical importance.', 'Cross-NumPy exact entropy-summary mismatch retained when present; source NumPy environment exact replay required, original score/weight tolerances unchanged.'])
    r.save(H/'analysis/pilot_comparison.json',result);print(json.dumps(result))
if __name__=='__main__':main()
