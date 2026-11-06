"""Audit provider algebra and our ingestion separately from physical accuracy."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
G=Path('D:/gait-research-data/D_P/planar-tug-improvement-20261104/gstride')
BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    verified_source_members=0
    for name in ['gstride-parameters-acquisition.json','gstride-segmentation-acquisition.json']:
        receipt=json.loads((HERE.parent/'20261104-planar-tug-improvement'/name).read_text())
        for row in receipt['members']:
            assert sha(G/'processed'/row['member'])==row['sha256'];verified_source_members+=1
    raw_receipt=json.loads((HERE/'raw-audit-acquisition.json').read_text())
    for row in raw_receipt['members']:assert sha(row['path'])==row['sha256']
    paths=sorted((G/'processed/_GAIT/_PARAMETERS').glob('*.csv'));assert len(paths)==158
    maximum={k:0. for k in ['speed_length_over_time','cadence_120_over_time','swing_fraction_from_events','swing_plus_stance_percent']}
    total=0;bad={k:0 for k in maximum};bad_event_order=0;first_turn_zero=0;files=[]
    for path in paths:
        pid=path.name[:4];pp=list(csv.DictReader(path.open()));ep=G/'processed/_GAIT/_SEGMENTATION'/f'{pid}_GAIT_SEGMENTATION.csv'
        ee=list(csv.DictReader(ep.open()));pm={int(float(q['index'])):q for q in pp};em={int(float(q['index'])):q for q in ee}
        assert len(pm)==len(pp) and len(em)==len(ee) and set(pm)==set(em)
        implied=[]
        for idx in pm:
            q=pm[idx];ev=em[idx];dt=float(q['CycleDuration[s]']);length=float(q['StrideLength[m]']);speed=float(q['Stride_Velocity[m/s]']);cad=float(q['Cadence[steps/min]']);sw=float(q['Per_Swing[%CD]']);stance=float(q['Per_Stance[%CD]'])
            hs0=float(ev['HS_ini[sample]']);hs1=float(ev['HS_fin[sample]']);toe=float(ev['TO[sample]']);ff0=float(ev['FF_ini[sample]']);ff1=float(ev['FF_fin[sample]'])
            errors={'speed_length_over_time':abs(speed-length/dt),'cadence_120_over_time':abs(cad-120/dt),'swing_fraction_from_events':abs(sw/100-(hs1-toe)/(hs1-hs0)),'swing_plus_stance_percent':abs(sw+stance-100)}
            for k,v in errors.items():
                assert np.isfinite(v);maximum[k]=max(maximum[k],v);bad[k]+=v>1e-7
            bad_event_order+=not(hs0<ff0<=ff1<toe<hs1);implied.append((hs1-hs0)/dt);total+=1
        first_turn_zero+=float(pp[0]['TurningAngle[degrees]'])==0
        files.append({'id':pid,'cycles':len(pp),'implied_sampling_rate_median':float(np.median(implied)),'implied_sampling_rate_spread':float(np.ptp(implied)),'parameters_sha256':sha(path),'events_sha256':sha(ep)})
    prepared=json.loads((BIG/'G-features.json').read_text());input_error=0.
    for r in prepared:
        q=next(q for q in csv.DictReader((G/'processed/_GAIT/_PARAMETERS'/f'{r["id"]}_GAIT_PARAMETERS.csv').open()) if int(float(q['index']))==r['index'])
        expected=[float(q['Stride_Velocity[m/s]']),float(q['CycleDuration[s]']),float(q['Per_Swing[%CD]'])/100]
        input_error=max(input_error,max(abs(a-b) for a,b in zip(r['X'],expected)))
    raw_results=[]
    for pid in ['V001','V026','V138']:
        raw=BIG/'gstride-raw-audit'/f'{pid}.csv'
        if not raw.exists():continue
        with raw.open() as f:
            reader=csv.DictReader(f);names=reader.fieldnames;arr=np.array([[float(q[n]) for n in names] for q in reader])
        ti=names.index('timestamp[s]');t=arr[:,ti];diff=np.diff(t);assert np.isfinite(arr).all() and (diff>0).all()
        pp=list(csv.DictReader((G/'processed/_GAIT/_PARAMETERS'/f'{pid}_GAIT_PARAMETERS.csv').open()));ee=list(csv.DictReader((G/'processed/_GAIT/_SEGMENTATION'/f'{pid}_GAIT_SEGMENTATION.csv').open()));errors=[];time_errors_one=[];examples=[]
        for q,ev in zip(pp,ee):
            assert q['index']==ev['index'];a=int(float(ev['HS_ini[sample]']));b=int(float(ev['HS_fin[sample]']));dt=float(q['CycleDuration[s]'])
            if b>=len(t):continue
            errors.append(abs((t[b]-t[a])-dt));time_errors_one.append(abs((t[b-1]-t[a-1])-dt))
            if len(examples)<2:examples.append({'index':q['index'],'source_cycle_seconds':dt,'timestamp_seconds_zero_based':float(t[b]-t[a]),'timestamp_seconds_one_based':float(t[b-1]-t[a-1]),'start_seconds_zero_based':float(t[a]),'source_length_m':float(q['StrideLength[m]']),'source_speed_mps':float(q['Stride_Velocity[m/s]'])})
        raw_results.append({'id':pid,'columns':names,'samples':len(t),'duration_seconds':float(t[-1]-t[0]),'effective_sampling_rate':float(1/np.median(diff)),'timestamp_step_quantiles':np.quantile(diff,[0,.5,1]).tolist(),'max_cycle_time_error_zero_based':max(errors),'max_cycle_time_error_one_based':max(time_errors_one),'samples_checked':len(errors),'acc_norm_median_mps2':float(np.median(np.linalg.norm(arr[:,:3],axis=1))),'sha256':sha(raw),'examples':examples})
    out={'status':'audit_complete','official_processed_member_sha_checks':verified_source_members,'raw_crc_verified_member_sha_checks':len(raw_receipt['members']),'provider_people_with_processed_data':len(paths),'cycles_checked':total,'maximum_formula_errors':maximum,'formula_error_counts_above_1e-7':bad,'event_order_violations':bad_event_order,'first_turn_angle_zero_files':first_turn_zero,'prepared_people_checked':len(prepared),'prepared_input_max_difference':input_error,'raw_timestamp_pilots':raw_results,'files':files,
      'limits':'Provider algebra consistency and ingestion do not establish physical displacement/heelcontact accuracy. No independent synchronized distance truth. Paper2023 processing differs fromV3; exact workbook aggregation mask not reproduced. No rawINS reintegration or newmodeltraining.'}
    (HERE/'gstride-numeric-audit.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
