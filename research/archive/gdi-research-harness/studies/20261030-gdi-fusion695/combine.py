"""Collect the predeclared comparisons after all phases have finished."""
from common import HERE, ROOT, arrays, data, load, save, sha, summary
import csv
from datetime import datetime, timezone
import numpy as np


def main():
    d=data();methods={};proofs={}
    assert load(HERE/'frame_verification.json')['verified']
    phases=[('rf',HERE/'rf_predictions.npz',HERE/'rf_summary.json',HERE/'rf_verification.json'),
            ('raw',HERE/'cnn/raw/oof_predictions.npz',HERE/'cnn/raw/summary.json',HERE/'cnn/raw/verification.json'),
            ('temporal',HERE/'cnn/temporal/oof_predictions.npz',HERE/'cnn/temporal/summary.json',HERE/'cnn/temporal/verification.json')]
    for phase,prediction,report,verification in phases:
        proof=load(verification)
        assert proof['verified'] and proof['summary_sha256']==sha(report)
        for name,value in arrays(prediction).items():
            if name in methods:np.testing.assert_array_equal(methods[name],value)
            else:methods[name]=value
        proofs[phase]={'summary_sha256':sha(report),'verification_sha256':sha(verification),'prediction_sha256':sha(prediction)}
    assert len(methods)==11
    np.savez_compressed(HERE/'all_predictions.npz',**methods,person_id=d['person_id'],video_id=d['video_id'],fold=d['fold'],y=d['y'])
    with (HERE/'all_predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream)
        writer.writerow(['method','person_id','video_id','fold','side','actual_GDI','predicted_GDI'])
        for name,pred in methods.items():
            for i in range(1064):
                for j,side in enumerate(['left','right']):
                    writer.writerow([name,d['person_id'][i],d['video_id'][i],int(d['fold'][i]),side,float(d['y'][i,j]),float(pred[i,j])])
    result={**summary(d,methods),'people':695,'records':1064,'methods':list(methods),
            'phase_evidence':proofs,'completed_utc':datetime.now(timezone.utc).isoformat(),
            'existing_CNN_RF_status_before_this_study':load(HERE/'existing_experiment_audit.json'),
            'independent_final_evaluation':False,'protected_numeric_data_read':False,
            'all_new_CNN_training_device':'CPU','new_encoder_training':False,
            'all_predictions_sha256':sha(HERE/'all_predictions.npz'),
            'execution_note':'Phase elapsed_seconds values measure the last invocation. Raw CNN resumed after a transient Windows log-file lock; all 20 inner curves and five selected refits were checked. Source learning functions were unchanged by the IO-only retry patch.',
            'code_sha256':{name:sha(HERE/name) for name in ['common.py','prepare.py','train_rf.py','train_cnn.py','verify.py','verify_frames.py','combine.py','cnn_source/run.py','cnn_source/model.py','cnn_execution.json','io_patch.json']}}
    bands=[]
    for name,pred in methods.items():
        for band in sorted(set(np.floor(d['y'].reshape(-1)/10).astype(int))):
            mask=np.floor(d['y']/10)==band
            errors=[];biases=[]
            for person in np.unique(d['person_id']):
                rows=d['person_id']==person
                selected=mask[rows]
                if selected.any():
                    residual=(pred[rows]-d['y'][rows])[selected]
                    errors.append(float(abs(residual).mean()));biases.append(float(residual.mean()))
            bands.append({'method':name,'actual_bin_low':int(band*10),'people':len(errors),
                          'person_MAE':float(np.mean(errors)),'bias_prediction_minus_truth':float(np.mean(biases))})
    save('score_bands.json',{'rule':'Actual 10-point bins; records/sides first averaged within each eligible person. A person may appear in multiple bins. Descriptive, not clinical grades.','rows':bands})
    result['score_bands_sha256']=sha(HERE/'score_bands.json')
    save('summary.json',result)
    print(json_message(result))


def json_message(result):
    import json
    return json.dumps({'completed':True,'people':695,'records':1064,
                       'MAE':{k:round(v['MAE'],6) for k,v in result['metrics'].items()}},ensure_ascii=False)


if __name__=='__main__':main()
