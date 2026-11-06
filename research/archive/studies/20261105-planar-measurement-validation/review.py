"""Independent replay of units, events, measurements, aggregation and within-record changes."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
from scipy.signal import savgol_filter,find_peaks
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
import measurement as m
H=m.H
FIELDS=['speed','step_length','step_time','stride_length','step_length_asymmetry','step_time_asymmetry']

def independently_measure(foot,events,fs):
    steps=[];strides=[]
    for i in range(1,len(events)):
        now=events[i];previous=events[i-1];elapsed=(now['index']-previous['index'])/fs;side=now['side']
        if previous['side']!=side and .2<=elapsed<=1.5:steps.append({'side':side,'time':now['index']/fs,'step_time':elapsed,'step_length':float(foot[now['index'],side]-foot[now['index'],1-side])})
    for side in [0,1]:
        indices=[r['index'] for r in events if r['side']==side]
        for a,b in zip(indices,indices[1:]):
            if .8<=(b-a)/fs<=3:strides.append(float(foot[b,side]-foot[a,side]))
    values={'step_time':float(np.mean([r['step_time'] for r in steps])) if len(steps)>=2 else None,'step_length':float(np.mean([r['step_length'] for r in steps])) if len(steps)>=2 else None,'stride_length':float(np.mean(strides)) if strides else None}
    for field in ['step_time','step_length']:
        left=[r[field] for r in steps if r['side']==0];right=[r[field] for r in steps if r['side']==1]
        values[field+'_asymmetry']=float(200*abs(np.mean(left)-np.mean(right))/max(np.mean(left)+np.mean(right),2e-6)) if len(left)>=2 and len(right)>=2 else None
    return values,steps

def assert_values(actual,expected):
    for key in expected:
        if expected[key] is None:assert actual[key] is None
        else:
            assert actual[key] is not None,('mismatched_availability',key,actual,expected)
            np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-10)

def independent_events(foot,pelvis,fs):
    width=7 if fs==30 else 29;signal=savgol_filter(foot,width,2,axis=0)-savgol_filter(pelvis,width,2)[:,None];out=[]
    for side in [0,1]:
        prominence=max(.02,.15*(signal[:,side].max()-signal[:,side].min()));indices=find_peaks(signal[:,side],distance=.4*fs,prominence=prominence)[0]
        out.extend({'side':side,'index':int(index),'time':float(index/fs)} for index in indices)
    return sorted(out,key=lambda r:r['time'])

def cluster_stats(items):
    if not items:return {'people':0,'windows':0,'MAE':None,'MAE_CI95':None}
    groups=collections.defaultdict(list)
    for sid,truth,pred in items:groups[sid].append(float(pred-truth))
    per=np.array([np.mean(abs(np.array(e))) for e in groups.values()]);signed=np.array([np.mean(e) for e in groups.values()]);allerr=np.array([p-y for _,y,p in items]);rng=np.random.default_rng(20261105);bs=per[rng.integers(0,len(per),(10000,len(per)))].mean(1)
    return {'people':len(per),'windows':len(items),'MAE':float(per.mean()),'MAE_CI95':np.quantile(bs,[.025,.975]).tolist() if len(per)>=2 else None,'bias_person_equal':float(signed.mean()),'p95_AE_windows':float(np.quantile(abs(allerr),.95)),'descriptive_window_LoA95':(allerr.mean()+np.array([-1,1])*1.96*allerr.std(ddof=1)).tolist() if len(allerr)>1 else None}

def same_to_frozen_precision(a,b):
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_to_frozen_precision(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(same_to_frozen_precision(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):return abs(a-b)<=1e-10
    return a==b

def main():
    protocol=m.read(H/'protocol.json');assert protocol['code_sha256']==m.sha(H/'measurement.py');audit=m.read(H/'input-audit.json');data=m.read(H/'measurements.json');assert data['input_audit_sha256']==m.sha(H/'input-audit.json') and data['code_sha256']==protocol['code_sha256']
    for path,digest in audit['source_hashes'].items():assert m.sha(path)==digest
    index={r['key']:r for r in audit['records']};rows=data['records'];assert len(index)==len(rows);replayed=0;rejected=0
    for row in rows:
        source=index[row['key']];assert m.sha(source['array'])==source['sha256'];d=np.load(source['array']);np.testing.assert_array_equal(d['native_index'][::4],(d['frames']-1)*4)
        if row['reference_status']!='ok':assert not row['reference'] and not row['prediction'];rejected+=1;continue
        lm=d['lm'];xy=lm[:,:,:2]*[800,600];hip=(xy[:,23]+xy[:,24])/2;scale=float(d['height'])/np.median(xy[:,:,1].max(1)-xy[:,:,1].min(1));t=np.arange(len(xy))/30;pv=np.array([linregress(t,hip[:,axis]).slope for axis in [0,1]])*scale
        np.testing.assert_allclose(row['raw_speed'],np.linalg.norm(pv),rtol=0,atol=1e-10)
        pelvis=d['pelvis120'][:,:2];centered=pelvis-pelvis.mean(0);eigenvalues,vectors=np.linalg.eigh(centered.T@centered);axis=vectors[:,-1]
        if np.dot(axis,pelvis[-1]-pelvis[0])<0:axis=-axis
        foot=np.einsum('ntk,k->nt',d['heels120'][:,:,:2],axis);forward=pelvis@axis;ev=independent_events(foot,forward,120);assert ev==row['reference_events'];ref,steps=independently_measure(foot,ev,120);t=np.arange(len(pelvis))/120;ref['speed']=float(np.linalg.norm([linregress(t,pelvis[:,a]).slope for a in [0,1]]));assert_values(row['reference'],ref)
        assert len(steps)==len(row['reference_steps'])
        for aa,bb in zip(steps,row['reference_steps']):
            assert aa['side']==bb['side']
            for key in ['time','step_time','step_length']:np.testing.assert_allclose(aa[key],bb[key],rtol=0,atol=1e-10)
        direction=1 if hip[-1,0]>=hip[0,0] else -1;foot=xy[:,[29,30],0]*scale*direction;forward=hip[:,0]*scale*direction;ev=independent_events(foot,forward,30);assert ev==row['predicted_events'];pr,steps=independently_measure(foot,ev,30);pr['speed']=float(np.linalg.norm(pv))
        speedgood=np.isfinite(lm).all() and (d['pose_counts']>0).all() and np.median(lm[:,[23,24],3])>=.5;footgood=speedgood and np.median(lm[:,29,3])>=.5 and np.median(lm[:,30,3])>=.5 and abs(hip[-1,0]-hip[0,0])*scale>=.1
        assert bool(speedgood)==row['speed_quality'] and bool(footgood)==row['foot_quality']
        # Rebuild same-foot event matching from frame indices, without calling the extractor.
        errors=[]
        if footgood:
            for side in [0,1]:
                pp=[e['index']/30 for e in row['predicted_events'] if e['side']==side]
                rr=[e['index']/120 for e in row['reference_events'] if e['side']==side]
                if pp and rr:
                    costs=np.array([[abs(a-b) for b in rr] for a in pp]);ii,jj=linear_sum_assignment(costs)
                    errors.extend(pp[i]-rr[j] for i,j in zip(ii,jj) if costs[i,j]<=.15)
        observed=row['event_match'];assert observed['TP']==len(errors)
        assert observed['FP']==(len(row['predicted_events'])-len(errors) if footgood else 0)
        assert observed['FN']==len(row['reference_events'])-len(errors)
        np.testing.assert_allclose(observed['timing_errors'],errors,rtol=0,atol=1e-10)
        if not speedgood:pr={key:None for key in pr}
        elif not footgood:pr={key:val if key=='speed' else None for key,val in pr.items()}
        for key in ['step_length','stride_length']:
            if pr[key] is not None and pr[key]<=0:pr[key]=None
        assert_values(row['prediction'],pr);assert len(steps)==len(row['predicted_steps'])
        for aa,bb in zip(steps,row['predicted_steps']):
            assert aa['side']==bb['side']
            for key in ['time','step_time','step_length']:np.testing.assert_allclose(aa[key],bb[key],rtol=0,atol=1e-10)
        replayed+=1
    # Coverage extension must leave every original primary/4s estimate identical.
    old=m.read(H/'measurements-before-2s-pooling-extension.json')['records'];new={r['key']:r for r in rows};corrected=[];roundoff_only=[]
    for row in old:
        if row!=new[row['key']]:
            boundary=False
            for key,fs in [('predicted_events',30),('reference_events',120)]:
                ev=row.get(key,[])
                for a,b in zip(ev,ev[1:]):
                    if a['side']!=b['side']:
                        x=b['time']-a['time'];y=(b['index']-a['index'])/fs
                        boundary|=(.2<=x<=1.5)!=(.2<=y<=1.5)
                for side in [0,1]:
                    same=[e for e in ev if e['side']==side]
                    for a,b in zip(same,same[1:]):
                        x=b['time']-a['time'];y=(b['index']-a['index'])/fs;boundary|=(.8<=x<=3)!=(.8<=y<=3)
            if boundary:corrected.append(row['key'])
            else:
                assert same_to_frozen_precision(row,new[row['key']]),('unexplained_change',row['key']);roundoff_only.append(row['key'])
        for key in ['key','subject','camera','group','first_frame','last_frame','reference_status']:assert row[key]==new[row['key']][key]
    summaries={};paired_rows=[]
    for group in ['primary2s','extra2s','extra4s']:
      summaries[group]={}
      for camera in ['PG1','PG2']:
        subset=[r for r in rows if r['group']==group and r['camera']==camera];valid=[r for r in subset if r['reference_status']=='ok'];results={}
        for field in FIELDS:
            ref_people=set(r['subject'] for r in valid if r['reference'].get(field) is not None);pairs=[(r['subject'],r['reference'][field],r['prediction'][field]) for r in valid if r['reference'].get(field) is not None and r['prediction'].get(field) is not None]
            results[field]={**cluster_stats(pairs),'registered_people':len(set(r['subject'] for r in subset)),'geometry_reference_valid_people':len(set(r['subject'] for r in valid)),'metric_reference_eligible_people':len(ref_people)}
            paired_rows.extend({'subject':sid,'group':group,'camera':camera,'metric':field,'reference':y,'prediction':pr} for sid,y,pr in pairs)
        tp=sum(r['event_match']['TP'] for r in valid);fp=sum(r['event_match']['FP'] for r in valid);fn=sum(r['event_match']['FN'] for r in valid);err=[e for r in valid for e in r['event_match']['timing_errors']]
        summaries[group][camera]={'registered_windows':len(subset),'reference_valid_windows':len(valid),'measures':results,'events':{'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,'F1':2*tp/(2*tp+fp+fn) if tp+fp+fn else None,'matched_event_MAE_seconds':float(np.mean(abs(np.array(err)))) if err else None}}
    pooled={};pool_records=[]
    for camera in ['PG1','PG2']:
        pooled[camera]={}
        for field in ['step_length','step_time']:
            eligible=[r for r in rows if r['group']=='extra2s' and r['camera']==camera and r['reference_status']=='ok'];by=collections.defaultdict(list)
            for row in eligible:by[row['subject']].append(row)
            pairs=[]
            for sid,subjectrows in by.items():
                common=[r for r in subjectrows if r['foot_quality'] and not(field=='step_length' and r['reason']=='nonpositive_forward_length')]
                ref={s:[v[field] for r in common for v in r['reference_steps'] if v['side']==s] for s in [0,1]};pr={s:[v[field] for r in common for v in r['predicted_steps'] if v['side']==s] for s in [0,1]}
                if min(len(ref[0]),len(ref[1]),len(pr[0]),len(pr[1]))<2:continue
                def asym(z):return float(200*abs(np.mean(z[0])-np.mean(z[1]))/max(np.mean(z[0])+np.mean(z[1]),2e-6))
                yy=asym(ref);pp=asym(pr);pairs.append((sid,yy,pp));pool_records.append({'subject':sid,'camera':camera,'metric':field+'_asymmetry','reference':yy,'prediction':pp,'windows_used':[r['key'] for r in common],'reference_side_steps':[len(ref[s]) for s in [0,1]],'predicted_side_steps':[len(pr[s]) for s in [0,1]]})
            pooled[camera][field+'_asymmetry']={**cluster_stats(pairs),'registered_people':30,'reference_geometry_people':len(by),'units':'percentagepoints','status':'exploratorywithin_recordpooledsteps_notretest'}
    changes={};change_records=[]
    for camera in ['PG1','PG2']:
        changes[camera]={}
        for field in ['speed','step_length','step_time']:
            by=collections.defaultdict(list)
            for row in rows:
                if row['group']=='extra2s' and row['camera']==camera and row['reference_status']=='ok' and row['reference'].get(field) is not None and row['prediction'].get(field) is not None:by[row['subject']].append(row)
            pairs=[]
            for sid,subjectrows in by.items():
                subjectrows.sort(key=lambda r:r['first_frame'])
                for a,b in zip(subjectrows,subjectrows[1:]):
                    yy=b['reference'][field]-a['reference'][field];pp=b['prediction'][field]-a['prediction'][field];pairs.append((sid,yy,pp));change_records.append({'subject':sid,'camera':camera,'metric':field,'first':a['key'],'second':b['key'],'reference_change':yy,'predicted_change':pp})
            changes[camera][field]=cluster_stats(pairs)
    if paired_rows:
        with (H/'measurement-pairs.csv').open('w',newline='',encoding='utf8') as file:writer=csv.DictWriter(file,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    m.save(H/'pooled-symmetry-records.json',pool_records);m.save(H/'within-record-change-pairs.json',change_records)
    result={'status':'independent_marker_reference_validation_not_clinical_gold','summary':summaries,'pooled_symmetry':pooled,'within_record_change':changes,'ages':{'min':min(r['age'] for r in audit['people']),'max':max(r['age'] for r in audit['people']),'older65':0},'clinical_contact_truth':'unavailable:forceplate/expertcontactlabelsabsent','true_test_retest_or_longitudinal_decline':'not_evaluated:noindependentrepeatcaptureinacquiredFround','change_limits':'Withinrecordwindowsnotindependentcapture;naturalvariationnotknownclinicalchange;referencekinematiceventalgorithmhasitsownerror.'}
    m.save(H/'results.json',result);m.save(H/'verification.json',{'status':'passed','source_hash_files':len(audit['source_hashes']),'registered_windows_views':len(rows),'numeric_measurements_replayed':replayed,'reference_rejected_windows_views':rejected,'original132_windows_and_people_unchanged':True,'old132_arithmetic_corrected_records':corrected,'old132_roundoff_only_at_frozen_1e10_tolerance':roundoff_only,'exact_native_to_cached_joint_replay':True,'prediction_and_reference_rebuilt_independently':True,'results_sha256':m.sha(H/'results.json'),'measurements_sha256':m.sha(H/'measurements.json'),'review_code_sha256':m.sha(__file__)})
    print(json.dumps({'primary':{cam:{field:{k:v for k,v in val.items() if k in ['people','MAE','MAE_CI95']} for field,val in q['measures'].items()} for cam,q in summaries['primary2s'].items()},'pooled':pooled,'changes':changes},indent=2),flush=True)

if __name__=='__main__':main()
