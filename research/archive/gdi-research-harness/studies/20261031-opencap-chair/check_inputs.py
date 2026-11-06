from context import *
from verify_geometry import independently_turn
from verify_inputs import walk
from common import identity

def independent_height(a,w):
 lo,hi=w;t=a['time'];b=a[(t>=lo-1e-7)&(t<=hi+1e-7)];grid=b['time'][0]+np.arange(round((hi-lo)*30)+1)/30
 assert grid[-1]<=b['time'][-1]+1e-6
 h=np.interp(grid,b['time'],b['pelvis_ty']);v=np.empty_like(h);v[0]=(h[1]-h[0])*30;v[-1]=(h[-1]-h[-2])*30;v[1:-1]=(h[2:]-h[:-2])*15
 return np.array([hi-lo,h[-1]-h[0],h.max()-h.min(),np.mean(v),np.std(v),np.percentile(abs(v),95),np.mean(abs(v[1:]-v[:-1])),np.mean(abs(v)<.05)])

def main():
 prep=json.loads((STUDY/'preparation.json').read_text());assert sha(STUDY/'protocol.json')==prep['protocol_sha256'];priorplan=json.loads((PRIOR/'geometry/splits.json').read_text());old=dict(np.load(PRIOR/'geometry/input.npz'));labels=pd.read_csv(SOURCE/'raw/participant_info.csv').set_index(['subid','visit']);cache={};result={}
 if (STUDY/'preparation-extension.json').exists():prep['cohorts'].update(json.loads((STUDY/'preparation-extension.json').read_text())['cohorts'])
 for cohort,info in prep['cohorts'].items():
  out=STUDY/cohort
  for n,h in info['hashes'].items():assert sha(out/n)==h
  d=dict(np.load(out/'input.npz'));rows=json.loads((out/'manifest.json').read_text());worst=0.;n=len(rows);mapping=np.array([r['prior_sample'] for r in rows]);assert len(set(d['ids']))==n
  for i,r in enumerate(rows):
   oi=r['prior_sample'];assert d['ids'][i]==r['id']==old['ids'][oi];assert d['y'][i]==r['target']==old['y'][oi]==labels.loc[(r['id'],r['visit']),'tft_tug'];assert identity(r['turn_file'])[:2]==(r['id'],r['visit'])
   if r['id'] not in cache:
    tp=PRIOR/'raw'/r['turn_file'];wp=SOURCE/'raw'/r['walk_file'];assert sha(tp)==r['turn_sha256'] and sha(wp)==r['walk_sha256'];cache[r['id']]=(read(tp),read(wp))
   a,w=cache[r['id']];base=np.r_[walk(w,r['walk_start'],3)[0],independently_turn(a,r['turn_start'],r['turn_end'],r['axis'])];assert np.allclose(base,old['Walk3Turn'][oi],atol=2e-5)
   windows=r['phase_windows'];extra=sum(z[1]-z[0] for z in windows.values());cw=r['context_window'];assert abs(cw[1]-cw[0]-(r['turn_end']-r['turn_start'])-extra)<1e-6;assert abs(r['observed_seconds']-3-(cw[1]-cw[0]))<1e-6
   assert windows['rise'][1]<cw[0]+1e-6 and cw[0]<r['turn_start']<r['turn_end']<cw[1]
   if 'sit' in windows:assert cw[1]<windows['sit'][0]+1e-6
   ff=[]
   for k,interval in windows.items():
    assert interval[0]>=a['time'][0] and interval[1]<=a['time'][-1]+1e-6
    if cohort.startswith('fixed') or cohort=='rise_strict':assert abs(interval[1]-interval[0]-2)<1e-6
    b=a[(a['time']>=interval[0]-1e-7)&(a['time']<=interval[1]+1e-7)];peak=float(max(abs(np.diff(b['pelvis_ty'])/np.diff(b['time']))));assert abs(peak-r['phase_max_vertical_speed'][k])<1e-8
    if cohort!='fixed_flagged':assert peak<=3
    ff.append(independent_height(a,interval))
   for interval in [r['context_pre'],r['context_post']]:
    b=a[(a['time']>=interval[0]-1e-7)&(a['time']<=interval[1]+1e-7)];assert max(abs(np.diff(b['pelvis_ty'])/np.diff(b['time'])))<=3
   cf=np.r_[independent_height(a,r['context_pre'])[1:],independent_height(a,r['context_post'])[1:],independently_turn(a,cw[0],cw[1],r['axis'])[1:]]
   expected={'TurnBase':base,'TurnContext':np.r_[base,cf]}
   if 'sit' in windows:expected.update({'ChairTime':np.r_[base,[f[0] for f in ff]],'ChairMotion':np.r_[base,*[f[1:] for f in ff]],'ChairAll':np.r_[base,*ff],'PrepTime':np.r_[base,[f[0] for f in ff]],'PrepAll':np.r_[base,*ff]})
   else:expected['RiseMotion']=np.r_[base,ff[0][1:]]
   if cohort=='prep_event':
    t=a['time'];width=max(3,round(.25/np.median(np.diff(t))));raw=np.array([a['pelvis_tilt'],a['lumbar_extension'],(a['hip_flexion_l']+a['hip_flexion_r'])/2]);smooth=np.stack([np.convolve(np.pad(x,(width//2,width-1-width//2),mode='edge'),np.ones(width)/width,mode='valid') for x in raw]);baseline=np.median(smooth[:,t<=t[0]+.75],axis=1);mask=(np.sum(abs(smooth-baseline[:,None])>=8,axis=0)>=2)&(t>=t[0]+.75)&(t<r['core_rise'][0]);hold=max(1,round(.25/np.median(np.diff(t))));starts=[k for k in range(len(t)-hold+1) if all(mask[k:k+hold])];left=r['core_rise'][0]
    if starts:left=min(left,max(float(np.ceil(t[0]*30-1e-6)/30),float(np.floor((t[starts[0]]-.25)*30+1e-6)/30)))
    assert abs(left-windows['rise'][0])<1e-8 and r['observed_seconds']<=30
   for name in info['views']:
    diff=float(max(abs(expected[name]-d[name][i])));worst=max(worst,diff);assert np.allclose(expected[name],d[name][i],atol=2e-5,rtol=2e-6),(cohort,r['id'],name,diff)
  plan=json.loads((out/'splits.json').read_text());mintrain=999;mintest=999
  for ri,rep in enumerate(plan):
   tested=[]
   for fi,f in enumerate(rep['splits']):
    tr=set(f['train']);te=set(f['test']);assert not tr&te and tr|te==set(range(n));mintrain=min(mintrain,len(tr));mintest=min(mintest,len(te));tested+=f['test'];oldf=priorplan[ri]['splits'][fi]
    assert set(mapping[list(te)])==set(mapping)&set(oldf['test']);vv=[]
    for ii,inn in enumerate(f['inner']):
     it=set(inn['train']);iv=set(inn['validation']);assert not it&iv and it|iv==tr;assert set(mapping[list(iv)])==set(mapping)&set(oldf['inner'][ii]['validation']);vv+=inn['validation']
    assert sorted(vv)==sorted(tr)
   assert sorted(tested)==list(range(n))
  assert mintrain>10 and mintest>0
  result[cohort]={'people':n,'long_people':int(sum(d['y']>=20)),'largest_feature_difference':worst,'min_outer_train':mintrain,'min_outer_test':mintest,'same_visit_labels':True,'same_prior_person_folds':True,'equal_disjoint_observation_time':True}
 save('input-verification.json',{'passed':True,'cohorts':result,'code_sha256':sha(STUDY/'check_inputs.py')});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
