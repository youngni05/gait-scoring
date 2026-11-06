"""Pre-training sensitivity: retain flagged kinematics, never repair or hide them."""
from common import *
from prepare import qc,walk_window,turn_features,old
assert not (HERE/'preparation-sensitivity.json').exists()
cohort='event_flagged';out=HERE/cohort;out.mkdir(exist_ok=True)
protocol={'declared_before_training':True,'reason':'Strict quality filtering retains only4 of9 prior long-TUG people; assess exclusion effect without calling flagged data valid','views':['Walk3Speed','WalkMatchedSpeed','Walk3Turn'],'parent_protocol_sha256':sha(HERE/'protocol.json'),'change':'retain max-speed/yaw-step/joint-step flags; still require complete finite files, detected turn, matched straight duration and yaw range<=45deg','same_visits_and_frozen_splits':True}
save(cohort+'/protocol.json',protocol)
audit=json.loads((HERE/'turn-audit.json').read_text());manifest=pd.read_csv(PREV/'manifest.csv');lookup={}
for r in audit:
 if r['detection']=='passed':lookup.setdefault((r['id'],r['visit']),[]).append(r)
rows=[];zs=[];excluded=[]
for oi,m in manifest.iterrows():
 a=read(SOURCE/'raw'/m.file);start=float(m.start_time);spans=old.active(a);chosen=None;reasons=[]
 for r in sorted(lookup.get((m.id,int(m.visit)),[]),key=lambda r:(r['session']!=m.session,r['task']!='cone',r['session'],r['file'])):
  ts=float(np.ceil(r['start']*30-1e-6)/30);te=float(np.floor(r['end']*30+1e-6)/30);duration=te-ts;end=start+3+duration
  if not any(a['time'][l]<=start+1e-6 and a['time'][h]>=end-1e-6 for l,h,_ in spans):reasons.append('matched_straight_too_short');continue
  part=a[(a['time']>=start-1e-7)&(a['time']<=end+1e-7)];yaw=np.rad2deg(np.unwrap(np.deg2rad(part['pelvis_rotation'])))
  if np.ptp(yaw)>45:reasons.append('straight_not_straight');continue
  b=read(HERE/'raw'/r['file']);base,_=walk_window(a,start,3);long,_=walk_window(a,start,3+duration);tf=turn_features(b,ts,te)
  qtw=qc(b,ts,te);qsw=qc(a,start,end)
  chosen={'old_sample':int(oi),'id':m.id,'visit':int(m.visit),'target':float(m.target),'walk_file':m.file,'walk_sha256':sha(SOURCE/'raw'/m.file),'walk_start':start,'walk_end':end,'turn_file':r['file'],'turn_sha256':r['sha256'],'task':r['task'],'turn_session':r['session'],'turn_start':ts,'turn_end':te,'turn_mid':r['mid'],'turn_observed_seconds':duration,'matched_input_seconds':3+duration,'turn_qc':qtw,'walk_qc':qsw,'walk_yaw_range':float(np.ptp(yaw)),'quality_flagged':not(qtw['passed'] and qsw['passed'])}
  zs.append({'Walk3Speed':base,'WalkMatchedSpeed':long,'Walk3Turn':np.r_[base,tf]});break
 if chosen is None:excluded.append({'id':m.id,'target':float(m.target),'reasons':sorted(set(reasons)) or ['no_detected_same_visit_turn']})
 else:rows.append(chosen)
mapping={r['old_sample']:i for i,r in enumerate(rows)};plans=[]
for rep in json.loads((PREV/'splits.json').read_text()):
 ss=[]
 for f in rep['splits']:
  remap=lambda arr:[mapping[i] for i in arr if i in mapping]
  ss.append({'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]})
 plans.append({**rep,'splits':ss})
np.savez_compressed(out/'input.npz',ids=np.array([r['id'] for r in rows]),y=np.array([r['target'] for r in rows]),old_sample=np.array(list(mapping)),**{v:np.stack([z[v] for z in zs]) for v in protocol['views']})
save(cohort+'/manifest.json',rows);save(cohort+'/splits.json',plans);save(cohort+'/excluded.json',excluded)
record={'people':len(rows),'long_people':sum(r['target']>=20 for r in rows),'flagged_people':sum(r['quality_flagged'] for r in rows),'excluded_long':[r for r in excluded if r['target']>=20],'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json','excluded.json','protocol.json']}}
save('preparation-sensitivity.json',{'cohorts':{cohort:record},'code_sha256':sha(HERE/'prepare_sensitivity.py')});print(json.dumps(record),flush=True)
