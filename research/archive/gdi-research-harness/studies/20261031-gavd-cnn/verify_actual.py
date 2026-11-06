"""Independent checks on actual CNN windows, calibration, means and native timing."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np
from audit_gavd import HERE,DATA,write_json
from prepare_comparison import read_csv,BUNDLE
def main(args):
    models=json.loads((BUNDLE/'models.json').read_text());checks=[]
    for run,shift in [(args.primary_run,0),(args.sensitivity_run,1)]:
        droot=DATA/'runs'/run;meta=json.loads((droot/'predictions/run.json').read_text());records=read_csv(droot/'predictions/record_predictions.csv');windows=read_csv(droot/'predictions/window_predictions.csv');review={r['record_id']:r for r in read_csv(HERE/('reviewed_shift%d.csv'%shift))}
        assert meta['folds']==list(range(5)) and meta['training_performed']==False and meta['calibration_refitted']==False and meta['resampling_performed']==False
        assert meta['input_sha256']==hashlib.sha256((droot/'prepared/prepared.npz').read_bytes()).hexdigest()
        group=collections.defaultdict(list)
        for w in windows:
            key=(w['record_id'],int(w['fold']));group[key].append(w)
            assert int(w['frame_last'])-int(w['frame_first'])==123
            assert 0<=float(w['missing_fraction'])<=.25
            r=review[w['record_id']];assert int(w['frame_first'])>=int(r['annotation_frame_first']) and int(w['frame_last'])<=int(r['annotation_frame_last'])
        maxwindow=0;maxcal=0
        for r in records:
            ws=group[(r['record_id'],int(r['fold']))];assert len(ws)==int(r['windows'])
            slope,intercept=models['folds'][int(r['fold'])]['calibration_slope_intercept']
            for side in ['left','right']:
                maxwindow=max(maxwindow,abs(sum(float(w[side+'_GDI']) for w in ws)/len(ws)-float(r[side+'_GDI'])))
                maxcal=max(maxcal,abs(float(r[side+'_raw'])*slope+intercept-float(r[side+'_GDI'])))
        assert maxwindow<1e-8 and maxcal<1e-10
        with np.load(droot/'prepared/prepared.npz',allow_pickle=False) as z:
            assert z['x'].shape[1:]==(2,124,12) and np.isfinite(z['x']).all()
            assert all(float(z['fps'][i])==float(review[str(k)]['fps']) for i,k in enumerate(z['record_id']))
            shape=list(z['x'].shape)
        checks.append(dict(run=run,records=meta['records'],windows=meta['windows'],array_shape=shape,all_five_folds=True,native_fps_exact=True,frame_bounds_and_missingness_passed=True,window_to_record_max_error=maxwindow,saved_calibration_max_error=maxcal))
    pairchecks=[]
    for shift in [0,1]:
        cond={r['record_id']:r for r in read_csv(HERE/('conditions_shift%d.csv'%shift))};used=set()
        for pair in read_csv(HERE/('matched_pairs_shift%d.csv'%shift)):
            n=cond[pair['normal_record']];a=cond[pair['abnormal_record']]
            assert n['label']=='Normal Gait' and a['label']=='Abnormal Gait'
            assert all(n[k]==a[k] for k in ['view','environment','camera']) and abs(float(n['fps'])-float(a['fps']))<.001
            assert all(.5<=float(a[k])/float(n[k])<=2 for k in ['height','bbox_height_pixels'])
            k=(a['cluster_id'],a['view']);assert k not in used;used.add(k)
        pairchecks.append(dict(shift=shift,pairs=len(used),all_condition_constraints_passed=True))
    write_json(HERE/args.out/'actual_verification.json',dict(status='passed',runs=checks,condition_matching=pairchecks,decoder_exact_equivalence='Additional strict 4-ULP test did not pass; see decoder_verification.json. This result does not erase that separate limitation.'))
    print(json.dumps(dict(runs=checks,pairs=pairchecks)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--primary-run',default='primary');p.add_argument('--sensitivity-run',default='shift1');p.add_argument('--out',default='results');main(p.parse_args())
