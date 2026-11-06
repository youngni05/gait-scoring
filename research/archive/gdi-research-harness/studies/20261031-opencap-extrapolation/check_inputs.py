from study import *
from threadpoolctl import threadpool_limits

def main():
 protocol=json.loads((HERE/'protocol.json').read_text());results={};demo={}
 for cohort,info in protocol['cohorts'].items():
  src=source(info)
  for n,h in info['hashes'].items():assert sha(src/n)==h
  d=dict(np.load(src/'input.npz'));m=json.loads((src/'manifest.json').read_text());n=len(m);assert len(d['y'])==n and len(set(d['ids']))==n
  for i,r in enumerate(m):assert r['id']==d['ids'][i] and r['target']==d['y'][i]
  for rep in json.loads((src/'splits.json').read_text()):
   tested=[]
   for f in rep['splits']:
    tr=set(f['train']);te=set(f['test']);assert not tr&te and tr|te==set(range(n));tested+=f['test'];vv=[]
    for inn in f['inner']:
     it=set(inn['train']);iv=set(inn['validation']);assert not it&iv and it|iv==tr;vv+=inn['validation']
    assert sorted(vv)==sorted(tr)
   assert sorted(tested)==list(range(n))
  for r in m:
   if cohort=='geometry':assert abs(r['walk_end']-r['walk_start']-3-r['turn_observed_seconds'])<1e-6
   else:assert abs((r['context_window'][1]-r['context_window'][0])-(r['turn_end']-r['turn_start'])-sum(w[1]-w[0] for w in r['phase_windows'].values()))<1e-6
  results[cohort]={'people':n,'long_people':int(sum(d['y']>=20)),'unchanged_input_and_targets_and_folds':True,'equal_observation_time':True}
 # A capability check only, never clinical evidence.
 with threadpool_limits(limits=1):
  x=np.arange(11,dtype=float)[:,None];y=3+2*x[:,0]
  for family in GRIDS:
   pred,raw,_=fit(x,y,np.arange(6),np.array([10]),family,GRIDS[family][0]);assert pred[0]>max(y[:6]);demo[family]={'training_max':float(max(y[:6])),'outside_range_prediction':float(pred[0])}
 save('input-verification.json',{'passed':True,'cohorts':results,'synthetic_capability_only':demo,'code_sha256':sha(HERE/'check_inputs.py')});print(json.dumps({'cohorts':results,'synthetic':demo}),flush=True)
if __name__=='__main__':main()
