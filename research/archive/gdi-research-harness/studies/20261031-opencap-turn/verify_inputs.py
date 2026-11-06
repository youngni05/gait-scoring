from common import *
import zlib
ANGLES=['pelvis_tilt','pelvis_list']+[f'{n}_{s}' for s in ['r','l'] for n in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']]+['lumbar_extension','lumbar_bending','lumbar_rotation']
def derivative(v):
 result=np.empty_like(v,dtype=float);result[...,0]=(v[...,1]-v[...,0])*30;result[...,-1]=(v[...,-1]-v[...,-2])*30;result[...,1:-1]=(v[...,2:]-v[...,:-2])*15
 return result
def walk(a,start,seconds):
 t=a['time'];first=np.searchsorted(t,start-1e-7);last=np.searchsorted(t,t[first]+seconds);tt=t[first:last+1];grid=t[first]+np.arange(round(seconds*30)+1)/30
 assert grid[-1]<=tt[-1]+1e-6
 angle=np.vstack([np.interp(grid,tt,a[n][first:last+1]) for n in ANGLES]);pos=np.vstack([np.interp(grid,tt,a[n][first:last+1]) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]);vel=derivative(pos)
 x=np.vstack([angle,np.sqrt(vel[0]**2+vel[2]**2),vel[1]]).astype(np.float32)
 f=np.r_[np.average(x,axis=1),np.std(x,axis=1),np.mean(abs(x[:,1:]-x[:,:-1]),axis=1)]
 return f[[17,36,55]],f
def turn(a,start,end):
 m=(a['time']>=start-1e-7)&(a['time']<=end+1e-7);b=a[m];t=b['time'];grid=t[0]+np.arange(int(np.floor((t[-1]-t[0])*30+1e-5))+1)/30
 assert grid[-1]<=t[-1]+1e-6
 theta=np.interp(grid,t,np.unwrap(b['pelvis_rotation']*np.pi/180)*180/np.pi);w=abs(derivative(theta))
 pos=np.vstack([np.interp(grid,t,b[n]) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]);vel=derivative(pos);speed=np.sqrt(vel[0]**2+vel[2]**2)
 lateral=np.interp(grid,t,b['pelvis_list']);lumbar=np.interp(grid,t,b['lumbar_bending'])
 return np.array([end-start,np.average(w),np.std(w),np.percentile(w,95),np.sum(abs(theta[1:]-theta[:-1]))/max(abs(theta[-1]-theta[0]),1),np.average(speed),np.std(speed),np.average(abs(speed[1:]-speed[:-1])),np.std(lateral),max(lateral)-min(lateral),np.std(lumbar),np.std(vel[1])])
def main():
 prep=json.loads((HERE/'preparation.json').read_text());assert prep['protocol_sha256']==sha(HERE/'protocol.json')
 prep['cohorts'].update(json.loads((HERE/'preparation-sensitivity.json').read_text())['cohorts'])
 receipts=json.loads((HERE/'acquisition.json').read_text());assert len(receipts)==356
 for r in receipts:
  p=HERE/'raw'/r['name'];b=p.read_bytes();assert len(b)==r['bytes'] and zlib.crc32(b)==r['crc'] and sha(p)==r['sha256']
 clinical=pd.read_csv(SOURCE/'raw/participant_info.csv');assert not clinical.duplicated(['subid','visit']).any();lookup=clinical.set_index(['subid','visit'])
 old=dict(np.load(PREV/'input.npz'));oldplan=json.loads((PREV/'splits.json').read_text());cache={};result={}
 for cohort,info in prep['cohorts'].items():
  out=HERE/cohort
  for name,h in info['hashes'].items():assert sha(out/name)==h
  d=dict(np.load(out/'input.npz'));rows=json.loads((out/'manifest.json').read_text());plan=json.loads((out/'splits.json').read_text());n=len(rows)
  assert len(set(d['ids']))==n==len(d['y']);worst=0.
  for i,r in enumerate(rows):
   oi=int(d['old_sample'][i]);assert old['ids'][oi]==r['id']==d['ids'][i] and old['y'][oi]==d['y'][i]==r['target']==lookup.loc[(r['id'],r['visit']),'tft_tug']
   assert identity(r['walk_file'])[:2]==identity(r['turn_file'])[:2]==(r['id'],r['visit'])
   wp=SOURCE/'raw'/r['walk_file'];tp=HERE/'raw'/r['turn_file'];assert sha(wp)==r['walk_sha256'] and sha(tp)==r['turn_sha256']
   for p in [wp,tp]:
    if str(p) not in cache:cache[str(p)]=read(p)
   a=cache[str(wp)];b=cache[str(tp)];seconds=r['turn_end']-r['turn_start'];assert abs(r['walk_end']-r['walk_start']-3-seconds)<1e-6
   if cohort=='fixed6':assert abs(seconds-3)<1e-6
   base,unused=walk(a,r['walk_start'],3);long,full=walk(a,r['walk_start'],3+seconds);tf=turn(b,r['turn_start'],r['turn_end'])
   expected={'Walk3Speed':base,'WalkMatchedSpeed':long,'WalkMatchedFull':full,'Walk3TurnTime':np.r_[base,seconds],'Walk3TurnShape':np.r_[base,tf[1:]],'Walk3Turn':np.r_[base,tf]}
   for name in expected:
    if name in d:
     error=float(np.max(abs(expected[name]-d[name][i])));worst=max(worst,error);assert np.allclose(expected[name],d[name][i],rtol=2e-6,atol=2e-5),(cohort,r['id'],name,error)
   if cohort!='event_flagged':assert r['walk_qc']['passed'] and r['turn_qc']['passed']
   assert r['walk_yaw_range']<=45
  for ri,rep in enumerate(plan):
   tests=[]
   for fi,f in enumerate(rep['splits']):
    tr=set(f['train']);te=set(f['test']);assert not tr&te and tr|te==set(range(n));tests.extend(f['test'])
    oldf=oldplan[ri]['splits'][fi];assert set(d['old_sample'][f['test']])==set(oldf['test'])&set(d['old_sample'])
    vals=[]
    for ii,inn in enumerate(f['inner']):
     it=set(inn['train']);iv=set(inn['validation']);assert not it&iv and it|iv==tr and not te&(it|iv);vals.extend(inn['validation'])
     assert set(d['old_sample'][inn['validation']])==set(oldf['inner'][ii]['validation'])&set(d['old_sample'])
    assert sorted(vals)==sorted(tr)
   assert sorted(tests)==list(range(n))
  result[cohort]={'people':n,'passed':True,'largest_feature_difference':worst,'same_prior_outer_inner_person_splits':True,'same_visit_actual_labels':True,'matched_observation_seconds':True}
 save('input-verification.json',{'passed':True,'CRC_and_SHA_verified_files':len(receipts),'cohorts':result,'code_sha256':sha(HERE/'verify_inputs.py')});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
