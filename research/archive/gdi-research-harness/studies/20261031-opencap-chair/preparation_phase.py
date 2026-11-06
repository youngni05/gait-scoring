"""Exploratory preparation extension motivated by input/clinical timing audit."""
from context import *
from segment import features

def extend(a,event):
 t=a['time'];dt=np.median(np.diff(t));channels=np.column_stack([a['pelvis_tilt'],a['lumbar_extension'],(a['hip_flexion_l']+a['hip_flexion_r'])/2]);smooth=uniform_filter1d(channels,max(3,round(.25/dt)),axis=0,mode='nearest');baseline=np.median(smooth[t<=t[0]+.75],axis=0)
 changed=np.sum(abs(smooth-baseline)>=8,axis=1)>=2;eligible=(t>=t[0]+.75)&(t<event[0]);hold=max(1,round(.25/dt));valid=np.convolve((changed&eligible).astype(int),np.ones(hold,dtype=int),mode='valid')>=hold;starts=np.flatnonzero(valid)
 left=event[0]
 if len(starts):left=max(float(np.ceil(t[0]*30-1e-6)/30),float(np.floor((t[starts[0]]-.25)*30+1e-6)/30))
 return [min(left,event[0]),event[1]]

def main():
 cohort='prep_event';out=STUDY/cohort;assert not (out/'completion.json').exists();rows=[];zs=[];rejected=[];audit=[]
 protocol={'exploratory':True,'reason':'Input audit showed posture change before pelvis rise in long cases and disagreement between core-boundary span and clinical time; added before reading model performance','detector':'0.25s smoothed pelvis tilt, lumbar extension, mean left/right hip flexion; deviation>=8deg in at least2of3 from initial0.75s median for0.25s; earliest crossing before core rise with0.25s margin; does not use clinical target','observation':'unchanged same3s walk+turnD; prep-rise and sit versus same added time of non-chair TUG context; max total30s; same raw files and original person folds','quality':'same vertical<=3m/s; unused angular jump recorded and flagged rather than silently repaired; phase boundaries are proxies without command timestamps or RGB labels','views':['TurnBase','TurnContext','PrepTime','PrepAll'],'no_whole_TUG_elapsed_feature':True,'code_sha256':sha(STUDY/'preparation_phase.py')}
 save(cohort+'/protocol.json',protocol);manifests=json.loads((PRIOR/'geometry/manifest.json').read_text());events={r['id']:r for r in json.loads((STUDY/'transition-audit.json').read_text())};old=dict(np.load(PRIOR/'geometry/input.npz'))
 for oi,m in enumerate(manifests):
  try:
   ev=events[m['id']]
   if any('window' not in ev[k] for k in ['rise','sit']):raise ValueError('missing_required_transition')
   a=read(PRIOR/'raw'/m['turn_file']);rw=extend(a,ev['rise']['window']);sw=ev['sit']['window'];windows={'rise':rw,'sit':sw};peaks={}
   for k,w in windows.items():
    b=a[(a['time']>=w[0]-1e-7)&(a['time']<=w[1]+1e-7)];peaks[k]=float(max(abs(np.diff(b['pelvis_ty'])/np.diff(b['time']))))
   audit.append({'id':m['id'],'target':m['target'],'old_rise':ev['rise']['window'],'new_rise':rw})
   if max(peaks.values())>3:raise ValueError('chair_vertical_quality')
   extra=rw[1]-rw[0]+sw[1]-sw[0];budget=3+m['turn_observed_seconds']+extra
   if budget>30:raise ValueError('over30seconds')
   cw=central(a,m,m['turn_observed_seconds']+extra,[rw[1],sw[0]]);pre=[cw[0],m['turn_start']];post=[m['turn_end'],cw[1]]
   if min(pre[1]-pre[0],post[1]-post[0])<.2:raise ValueError('context_side_too_short')
   for w in [pre,post]:
    b=a[(a['time']>=w[0]-1e-7)&(a['time']<=w[1]+1e-7)]
    if max(abs(np.diff(b['pelvis_ty'])/np.diff(b['time'])))>3:raise ValueError('context_vertical_quality')
   base=old['Walk3Turn'][oi];f=[features(a,w) for w in windows.values()];cf=np.r_[features(a,pre)[1:],features(a,post)[1:],geom.features(a,cw[0],cw[1],m['axis'])[1:]]
   z={'TurnBase':base,'TurnContext':np.r_[base,cf],'PrepTime':np.r_[base,[v[0] for v in f]],'PrepAll':np.r_[base,*f]};part=a[(a['time']>=rw[0]-1e-7)&(a['time']<=rw[1]+1e-7)];anglejump=float(np.max(abs(np.diff(np.column_stack([part[k] for k in ['pelvis_tilt','lumbar_extension','hip_flexion_l','hip_flexion_r']]),axis=0))))
   rows.append({**m,'prior_sample':oi,'phase_windows':windows,'phase_max_vertical_speed':peaks,'max_preparation_angle_step':anglejump,'flagged_angle':anglejump>45,'context_window':cw,'context_pre':pre,'context_post':post,'observed_seconds':budget,'core_rise':ev['rise']['window']});zs.append(z)
  except ValueError as e:rejected.append({'id':m['id'],'target':m['target'],'reason':str(e)})
 mapping={r['prior_sample']:i for i,r in enumerate(rows)};plan=[]
 for rep in json.loads((PRIOR/'geometry/splits.json').read_text()):
  ss=[]
  for f in rep['splits']:
   remap=lambda arr:[mapping[i] for i in arr if i in mapping]
   ss.append({'fold':f['fold'],'train':remap(f['train']),'test':remap(f['test']),'inner':[{'train':remap(v['train']),'validation':remap(v['validation'])} for v in f['inner']]})
  plan.append({**rep,'splits':ss})
 save(cohort+'/manifest.json',rows);save(cohort+'/splits.json',plan);save(cohort+'/excluded.json',rejected);save(cohort+'/detection-audit.json',audit)
 np.savez_compressed(out/'input.npz',ids=np.array([r['id'] for r in rows]),y=np.array([r['target'] for r in rows]),**{v:np.stack([z[v] for z in zs]) for v in protocol['views']})
 info={'people':len(rows),'long_people':sum(r['target']>=20 for r in rows),'views':protocol['views'],'long':[{'id':r['id'],'rise':r['phase_windows']['rise'],'core_rise':r['core_rise'],'seconds':r['observed_seconds'],'angle_flag':r['flagged_angle']} for r in rows if r['target']>=20],'excluded_long':[r for r in rejected if r['target']>=20],'hashes':{n:sha(out/n) for n in ['input.npz','manifest.json','splits.json','excluded.json','protocol.json']}}
 save('preparation-extension.json',{'cohorts':{cohort:info}});print(json.dumps(info),flush=True)
if __name__=='__main__':main()
