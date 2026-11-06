from common import *
from inspect_inputs import detect
import importlib.util
spec=importlib.util.spec_from_file_location('old_prepare',SOURCE/'prepare.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
TURN_NAMES=['observed_turn_seconds','yaw_speed_mean','yaw_speed_std','yaw_speed_p95','yaw_path_ratio','horizontal_speed_mean','horizontal_speed_std','horizontal_speed_change','pelvis_list_std','pelvis_list_range','lumbar_bending_std','vertical_speed_std']
def qc(a,start,end):
 b=a[(a['time']>=start-1e-7)&(a['time']<=end+1e-7)];t=b['time']
 if len(t)<3:return {'passed':False,'reason':'too_short'}
 yaw=np.rad2deg(np.unwrap(np.deg2rad(b['pelvis_rotation'])))
 pos=np.column_stack([b['pelvis_tx'],b['pelvis_ty'],b['pelvis_tz']])
 v=np.linalg.norm(np.diff(pos[:,[0,2]],axis=0),axis=1)/np.diff(t)
 jump=float(np.max(abs(np.diff(yaw))));speed=float(max(v))
 jointjump=float(np.max(abs(np.diff(np.column_stack([b[n] for n in old.ANGLES]),axis=0))))
 return {'passed':bool(speed<=6 and jump<=30 and jointjump<=45),'max_speed':speed,'max_yaw_step':jump,'max_joint_step':jointjump}
def turn_features(a,start,end):
 # Only samples inside the selected interval contribute to model features.
 b=a[(a['time']>=start-1e-7)&(a['time']<=end+1e-7)];t=b['time']
 grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*30+1e-5))+1)/30
 yaw=np.interp(grid,t,np.rad2deg(np.unwrap(np.deg2rad(b['pelvis_rotation']))))
 positions=np.stack([np.interp(grid,t,b[n]) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']])
 velocity=np.gradient(positions,1/30,axis=1);v=np.hypot(velocity[0],velocity[2]);omega=abs(np.gradient(yaw,1/30))
 lateral=np.interp(grid,t,b['pelvis_list']);trunk=np.interp(grid,t,b['lumbar_bending'])
 return np.array([end-start,omega.mean(),omega.std(),np.quantile(omega,.95),np.sum(abs(np.diff(yaw)))/max(abs(yaw[-1]-yaw[0]),1),v.mean(),v.std(),abs(np.diff(v)).mean(),lateral.std(),np.ptp(lateral),trunk.std(),velocity[1].std()])
def summaries(x):return np.r_[x.mean(1),x.std(1),abs(np.diff(x,axis=1)).mean(1)]
def walk_window(a,start,seconds):
 # Keep the old sample interval and exact duration, with no padding or stitching.
 index=int(np.searchsorted(a['time'],start-1e-7));x=old.window(a,index,seconds)
 s=summaries(x);return s[[17,36,55]],s
def freeze_protocol():
 protocol={'target':'same person AND visit clinical tft_tug seconds; not inferred scores',
 'population':'subset of prior 298 people at their already selected visit; no visit or person replacement',
 'task_choice':'first QC-eligible matching original session preferred, then cone before line, then session/name; all candidates audited',
 'turn_detection':'pelvis yaw relative to median first 0.5s, unwrapped and 0.25s moving average; first 90degree crossing at >60% of 99th-percentile radial excursion from initial position; preceding20 and following160deg crossings, padded0.25s on both ends; excursion>=1.5m, endpoints>=45% excursion, subsequent return<=65%; duration0.3..27s before padding',
 'qc':{'inside_selected_windows_only':True,'native_max_horizontal_speed_m_s':6,'native_max_yaw_step_deg':30,'native_max_other_joint_step_deg':45,'source_max_time_gap_s':.05,'straight_yaw_range_max_deg':45,'interpretation':'heuristic kinematic plausibility screen, not original RGB tracking validation'},
 'event_matched':{'walk':'first3s old input and first(3+D)s from SAME old start','turn':'complete detected central140degree turn plus0.5s margins; D its duration','observation':'WalkMatched3+D seconds versus Walk3Turn3+D seconds, two separate recordings; no full TUG elapsed time feature','views':['Walk3Speed','WalkMatchedSpeed','WalkMatchedFull','Walk3TurnTime','Walk3TurnShape','Walk3Turn']},
 'fixed6':{'walk':'first3s and first6s from same old start','turn':'3s centered on first90degree crossing; may not encompass full turn','views':['Walk3Speed','WalkMatchedSpeed','WalkMatchedFull','Walk3Turn'],'observation':'6 seconds for matched straight and 3+3; event-matched cohort evaluated separately'},
 'turn_feature_names':TURN_NAMES,'training':'RF128trees leaf2/5 selected by inner MAE; explicit uniform sample weights; train-only standardization; 5 frozen person partitions x3 seeds; old outer and inner splits restricted to cohort, never reallocated',
 'tail':'TUG>=20seconds descriptive band, not a clinical threshold; alleligible people predicted regardless of target',
 'analysis':'MAE, signedbias, positive underestimation, RMSE per band and per person; paired person bootstrap of across-run errors conditional on fits; no claims of independent final validation',
 'limits':['same-visit label does not establish which cone/line trial the stopwatch label refers to','TUG turn is a part of the target assessment, not independent SPPB inference','window location and source IK use full recordings; observation budget refers to model-input motion, not recording duration','quality filtering changes population; excluded long people explicitly enumerated'],
 'model_seeds':SEEDS,'source_manifest_sha256':sha(PREV/'manifest.csv'),'label_sha256':sha(SOURCE/'raw/participant_info.csv')}
 save('protocol.json',protocol)
def build():
 assert not (HERE/'preparation.json').exists()
 freeze_protocol()
 manifest=pd.read_csv(PREV/'manifest.csv');clinical=pd.read_csv(SOURCE/'raw/participant_info.csv');assert not clinical.duplicated(['subid','visit']).any()
 labels=clinical.set_index(['subid','visit']);dold=dict(np.load(PREV/'input.npz'))
 receipts=json.loads((HERE/'acquisition.json').read_text());selected=json.loads((HERE/'selected-files.json').read_text());assert len(receipts)==len(selected)
 bypair={};audit=[]
 for receipt in receipts:
  name=receipt['name'];p=HERE/'raw'/name;assert sha(p)==receipt['sha256'];pid,v,s=identity(name)
  r={'file':name,'id':pid,'visit':v,'session':s,'sha256':receipt['sha256'],'task':'cone' if 'tug_cone' in name else 'line'}
  try:a=read(p);r.update(detect(a));r['detection']='passed';bypair.setdefault((pid,v),[]).append((r,a))
  except ValueError as e:r['detection']=str(e)
  audit.append(r)
 save('turn-audit.json',audit)
 plans=json.loads((PREV/'splits.json').read_text());outcomes={}
 for cohort in ['event_matched','fixed6']:
  rows=[];rejected=[];data=[]
  for oi,m in manifest.iterrows():
   assert str(dold['ids'][oi])==m.id
   y=float(labels.loc[(m.id,int(m.visit)),'tft_tug']);assert abs(y-m.target)<1e-8 and abs(y-dold['y'][oi])<1e-8
   wp=SOURCE/'raw'/m.file;assert sha(wp)==m.source_sha256
   a=read(wp);start=float(m.start_time);candidates=bypair.get((m.id,int(m.visit)),[])
   candidates=sorted(candidates,key=lambda pair:(pair[0]['session']!=m.session,pair[0]['task']!='cone',pair[0]['session'],pair[0]['file']))
   reasons=[];chosen=None
   for r,b in candidates:
    if cohort=='event_matched':ts,te=np.ceil(r['start']*30-1e-6)/30,np.floor(r['end']*30+1e-6)/30
    else:
     center=round(r['mid']*30)/30;ts,te=center-1.5,center+1.5
    duration=float(te-ts);budget=3+duration;end=start+budget
    if ts<b['time'][0] or te>b['time'][-1]:reasons.append('turn_clip_too_short');continue
    spans=old.active(a)
    if not any(a['time'][l]<=start+1e-6 and a['time'][h]>=end-1e-6 for l,h,_ in spans):reasons.append('matched_straight_too_short');continue
    qtw=qc(b,ts,te);qsw=qc(a,start,end)
    if not qtw['passed']:reasons.append('turn_quality');continue
    if not qsw['passed']:reasons.append('straight_quality');continue
    part=a[(a['time']>=start-1e-7)&(a['time']<=end+1e-7)]
    yaw=np.rad2deg(np.unwrap(np.deg2rad(part['pelvis_rotation'])))
    if np.ptp(yaw)>45:reasons.append('straight_not_straight');continue
    base,full3=walk_window(a,start,3);long,full=walk_window(a,start,budget)
    # Confirm old baseline features exactly reproduce the already frozen source.
    expected=summaries(dold['x'][oi])[[17,36,55]];assert np.allclose(base,expected,atol=2e-6)
    tf=turn_features(b,ts,te)
    z={'Walk3Speed':base,'WalkMatchedSpeed':long,'WalkMatchedFull':full,'Walk3TurnTime':np.r_[base,duration],'Walk3TurnShape':np.r_[base,tf[1:]],'Walk3Turn':np.r_[base,tf]}
    chosen={'old_sample':int(oi),'id':m.id,'visit':int(m.visit),'target':y,'walk_file':m.file,'walk_sha256':sha(wp),'walk_start':start,'walk_end':end,'turn_file':r['file'],'turn_sha256':r['sha256'],'task':r['task'],'original_session':int(m.session),'turn_session':r['session'],'turn_start':ts,'turn_end':te,'turn_mid':r['mid'],'turn_observed_seconds':duration,'matched_input_seconds':budget,'turn_qc':qtw,'walk_qc':qsw,'walk_yaw_range':float(np.ptp(yaw))}
    data.append(z);break
   if chosen is None:rejected.append({'id':m.id,'visit':int(m.visit),'target':y,'reasons':sorted(set(reasons)) or ['no_detected_same_visit_turn']})
   else:rows.append(chosen)
  out=HERE/cohort;out.mkdir(exist_ok=True);keep=[r['old_sample'] for r in rows];mapping={o:i for i,o in enumerate(keep)}
  filtered=[]
  for rep in plans:
   ss=[]
   for f in rep['splits']:
    remap=lambda arr:[mapping[i] for i in arr if i in mapping]
    ss.append({'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]})
   filtered.append({**rep,'splits':ss})
  views=json.loads((HERE/'protocol.json').read_text())[cohort]['views']
  arrays={v:np.stack([z[v] for z in data]) for v in views}
  np.savez_compressed(out/'input.npz',ids=np.array([r['id'] for r in rows]),y=np.array([r['target'] for r in rows]),old_sample=np.array(keep),**arrays)
  save(cohort+'/manifest.json',rows);save(cohort+'/excluded.json',rejected);save(cohort+'/splits.json',filtered)
  outcomes[cohort]={'people':len(rows),'long_people':sum(r['target']>=20 for r in rows),'excluded_long':[r for r in rejected if r['target']>=20],'task_counts':pd.Series([r['task'] for r in rows]).value_counts().to_dict(),'seconds_range':[min(r['matched_input_seconds'] for r in rows),max(r['matched_input_seconds'] for r in rows)],'seconds_median':float(np.median([r['matched_input_seconds'] for r in rows])),'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json','excluded.json']}}
  print(cohort,json.dumps(outcomes[cohort]),flush=True)
 save('preparation.json',{'cohorts':outcomes,'protocol_sha256':sha(HERE/'protocol.json'),'code_sha256':{p.name:sha(p) for p in [HERE/'prepare.py',HERE/'inspect_inputs.py',HERE/'common.py']},'source_splits_sha256':sha(PREV/'splits.json')})
if __name__=='__main__':build()
