from common import *
from verify_inputs import walk,derivative
def independently_turn(a,start,end,axis):
 t=a['time'];lo=np.searchsorted(t,start-1e-7);hi=np.searchsorted(t,end-1e-7);tt=t[lo:hi+1];grid=tt[0]+np.arange(round((end-start)*30)+1)/30;assert grid[-1]<=tt[-1]+1e-6
 p=np.stack([np.interp(grid,tt,a[n][lo:hi+1]) for n in ['pelvis_tx','pelvis_tz']]);v=derivative(p);speed=np.sqrt(np.sum(v**2,axis=0))
 smooth=np.stack([np.convolve(np.pad(row,(3,3),mode='edge'),np.ones(7)/7,mode='valid') for row in p]);sv=derivative(smooth);moving=np.sqrt(np.sum(sv**2,axis=0))>.1
 theta=np.unwrap(np.arctan2(sv[1],sv[0]));w=abs(derivative(theta))*180/np.pi
 first=np.mean(sv[:,:min(10,len(grid))],axis=1);last=np.mean(sv[:,-min(10,len(grid)):],axis=1);angle=np.degrees(np.arccos(np.clip(np.sum(first*last)/max(np.sqrt(np.sum(first**2)*np.sum(last**2)),1e-12),-1,1)))
 lat=-axis[1]*p[0]+axis[0]*p[1]
 return np.array([end-start,speed.mean(),speed.std(),np.mean(abs(np.diff(speed))),np.sum(np.sqrt(np.sum(np.diff(p,axis=1)**2,axis=0))),lat.max()-lat.min(),angle,w[moving].mean(),np.quantile(w[moving],.95),np.mean(speed<.2)])
def main():
 info=json.loads((HERE/'preparation-geometry.json').read_text())['cohorts']['geometry'];out=HERE/'geometry'
 for n,h in info['hashes'].items():assert sha(out/n)==h
 d=dict(np.load(out/'input.npz'));rows=json.loads((out/'manifest.json').read_text());old=dict(np.load(PREV/'input.npz'));clinical=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit']);worst=0.
 assert len(set(d['ids']))==len(rows)==238 and sum(d['y']>=20)==9
 for i,r in enumerate(rows):
  oi=int(d['old_sample'][i]);assert r['id']==d['ids'][i]==old['ids'][oi];assert r['target']==d['y'][i]==old['y'][oi]==clinical.loc[(r['id'],r['visit']),'tft_tug']
  assert identity(r['walk_file'])[:2]==identity(r['turn_file'])[:2]==(r['id'],r['visit'])
  wp=SOURCE/'raw'/r['walk_file'];tp=HERE/'raw'/r['turn_file'];assert sha(wp)==r['walk_sha256'] and sha(tp)==r['turn_sha256'];a=read(wp);b=read(tp)
  seconds=r['turn_end']-r['turn_start'];assert abs(r['walk_end']-r['walk_start']-3-seconds)<1e-6
  base,_=walk(a,r['walk_start'],3);long,_=walk(a,r['walk_start'],3+seconds);tf=independently_turn(b,r['turn_start'],r['turn_end'],r['axis']);v=old['x'][oi,17];orig=np.array([v.mean(),v.std(),np.mean(abs(np.diff(v)))])
  expected={'OldWalk3Speed':orig,'Walk3Speed':base,'WalkMatchedSpeed':long,'Walk3TurnTime':np.r_[base,seconds],'Walk3TurnShape':np.r_[base,tf[1:]],'Walk3Turn':np.r_[base,tf]}
  for name,z in expected.items():
   err=float(max(abs(z-d[name][i])));worst=max(worst,err);assert np.isfinite(z).all() and np.allclose(z,d[name][i],rtol=2e-6,atol=2e-5),(r['id'],name,err)
  for src,s,e,key in [(a,r['walk_start'],r['walk_end'],'walk_max_speed'),(b,r['turn_start'],r['turn_end'],'turn_max_speed')]:
   part=src[(src['time']>=s-1e-7)&(src['time']<=e+1e-7)];position=np.column_stack([part['pelvis_tx'],part['pelvis_tz']]);peak=float(max(np.linalg.norm(np.diff(position,axis=0),axis=1)/np.diff(part['time'])));assert abs(peak-r[key])<1e-7 and peak<=6
  assert r['straight_ratio']>=.85
 plan=json.loads((out/'splits.json').read_text());oldplan=json.loads((PREV/'splits.json').read_text());keep=set(d['old_sample']);n=len(rows)
 for ri,rep in enumerate(plan):
  tested=[]
  for fi,f in enumerate(rep['splits']):
   tr=set(f['train']);te=set(f['test']);assert not tr&te and tr|te==set(range(n));tested+=f['test'];oldf=oldplan[ri]['splits'][fi];assert set(d['old_sample'][f['test']])==keep&set(oldf['test']);vv=[]
   for ii,inn in enumerate(f['inner']):
    a=set(inn['train']);b=set(inn['validation']);assert not a&b and a|b==tr and not te&(a|b);vv+=inn['validation'];assert set(d['old_sample'][inn['validation']])==keep&set(oldf['inner'][ii]['validation'])
   assert sorted(vv)==sorted(tr)
  assert sorted(tested)==list(range(n))
 save('geometry-input-verification.json',{'passed':True,'people':n,'long_people':9,'largest_feature_difference':worst,'same_visit_labels':True,'frozen_outer_inner_person_splits':True,'equal_observation_budget':True,'selected_position_quality':True,'code_sha256':sha(HERE/'verify_geometry.py')});print('GEOMETRY_INPUT_VERIFIED',n,worst,flush=True)
if __name__=='__main__':main()
