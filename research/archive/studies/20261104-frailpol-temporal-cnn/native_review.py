"""All native/phase saved checkpoints: source/normalization/epoch/CPU64 audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'native-motion';O=B/'native-motion';sys.path.insert(0,str(H));from native_motion import Model
CONDS=['phase_tail4','native_tail4','native_tail16'];GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8')
def metrics(y,p):
    e=np.abs(p-y);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def audit_fit(rec,d,tr,ev,c,seed,epochs,grid):
    ctx=read(rec['context']);assert ctx['condition']==c and ctx['seed']==seed and rec['seed']==seed and ctx['epochs']==rec['epochs']==epochs;assert ctx['train']==tr.tolist() and ctx['eval']==ev.tolist() and not set(tr)&set(ev);assert ctx['train_ids']==d['ids'][tr].tolist() and ctx['eval_ids']==d['ids'][ev].tolist();assert ctx['protocol_sha256']==sha(R/'protocol.json')
    raw=d['phase' if c.startswith('phase') else 'native'];mask=raw[:,-1:];mu=[];sd=[]
    for j in range(24):
        observed=raw[tr,j][mask[tr,0]>0];mu.append(observed.mean());sd.append(max(observed.std(),1e-8))
    sd=np.array(sd);sd[sd<=1e-8]=1.;p=ctx['prep'];assert np.allclose(mu,p['channel_mean'],atol=1e-12,rtol=1e-12) and np.allclose(sd,p['channel_std'],atol=1e-12,rtol=1e-12);fm=d['F'][tr].mean(0);fs=d['F'][tr].std(0);fs[fs<1e-8]=1.;assert np.array_equal(fm,p['manual_mean']) and np.array_equal(fs,p['manual_std']);ym=float(d['y'][tr].mean());ys=max(float(d['y'][tr].std()),1e-8);assert p['target_mean']==ym and p['target_std']==ys and p['condition']==c
    x=raw[ev].copy();x[:,:24]=np.where(mask[ev]>0,(raw[ev,:24]-np.array(p['channel_mean'])[None,:,None])/np.array(p['channel_std'])[None,:,None],0);x=x.astype(np.float32).astype(float);f=((d['F'][ev]-p['manual_mean'])/p['manual_std']).astype(np.float32).astype(float);assert [r['epoch'] for r in rec['curve']]==grid;maxdelta=0
    for row in rec['curve']:
        assert sha(row['model'])==row['sha256'];model=Model().double();model.load_state_dict(torch.load(row['model'],map_location='cpu',weights_only=True));model.eval();out=[]
        with torch.no_grad():
            for lo in range(0,len(ev),64):out.extend(model(torch.from_numpy(x[lo:lo+64]),torch.from_numpy(f[lo:lo+64])).numpy()*ys+ym)
        pr=np.array(out);original=np.array(row['prediction']);assert np.allclose(pr,original,atol=1e-6,rtol=1e-6);maxdelta=max(maxdelta,float(np.max(np.abs(pr-original))));m=metrics(d['y'][ev],original)
        for key,v in m.items():assert np.isclose(v,row['metrics'][key],atol=1e-12,rtol=1e-12)
    return maxdelta
def main():
    torch.set_num_threads(2);proto=read(R/'protocol.json');assert proto['code_sha256']==sha(H/'native_motion.py') and proto['block_library_sha256']==sha(H/'bundle/train.py') and proto['data_sha256']==sha(O/'data.npz');receipt=read(R/'receipt.json');assert receipt['cases']==15 and receipt['protocol_sha256']==sha(R/'protocol.json')
    for p,v in receipt['hashes'].items():assert sha(p)==v
    d=np.load(O/'data.npz');orig=np.load(B/'data.npz');assert all(np.array_equal(d[key],orig[key]) for key in ['ids','y','fold']) and np.array_equal(d['F'][:,0],orig['F'][:,0]);windows={r['id']:r['window'] for r in read('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103/round05/features.json')};assert np.array_equal(d['F'][:,1],[(windows[int(i)]['core_end']-windows[int(i)]['core_start'])/100 for i in d['ids']]);assert len(d['y'])==594 and sum(d['y']>=20)==9;assert np.all((d['native'][:,-1]==0)|(d['native'][:,-1]==1)) and np.all(d['native'][:,:24][np.repeat(d['native'][:,-1:]==0,24,axis=1)]==0)
    checks=[];preds={};n=0
    for ci,c in enumerate(CONDS):
      p=np.full(594,np.nan)
      for fold in range(5):
        rec=read(O/c/f'fold{fold}/fold.json');ss=proto['splits'][str(fold)];tr=np.array(ss['train']);te=np.array(ss['test']);assert np.array_equal(tr,np.flatnonzero(d['fold']!=fold)) and np.array_equal(te,np.flatnonzero(d['fold']==fold));assert rec['test_ids']==d['ids'][te].tolist();inner=np.zeros((4,len(tr)));seen=[];delta=0
        for k,(r,s) in enumerate(zip(rec['inner'],ss['inner'])):
            it=np.array(s['train']);iv=np.array(s['valid']);assert set(it)|set(iv)==set(tr) and not set(it)&set(iv);delta=max(delta,audit_fit(r,d,it,iv,c,20261105+fold*100+ci+k*1000,100,GRID));positions=[np.flatnonzero(tr==i)[0] for i in iv]
            for j,row in enumerate(r['curve']):inner[j,positions]=row['prediction']
            seen.extend(iv);n+=4
        assert len(seen)==len(tr) and set(seen)==set(tr);scores=[metrics(d['y'][tr],pr) for pr in inner]
        for a,b in zip(scores,rec['inner_scores']):assert all(np.isclose(a[key],b[key],atol=1e-12,rtol=1e-12) for key in a)
        epoch=GRID[int(np.argmin([m['score'] for m in scores]))];assert epoch==rec['epoch'];delta=max(delta,audit_fit(rec['final'],d,tr,te,c,20261105+fold*100+ci+5000,epoch,[epoch]));n+=1;assert rec['prediction']==rec['final']['curve'][0]['prediction'];p[te]=rec['prediction'];checks.append(dict(condition=c,fold=fold,epoch=epoch,cpu64_max_delta=delta,checkpoint_replays=13,people_and_training_only_normalization_verified=True));print('verified',c,fold,flush=True)
      assert np.isfinite(p).all();preds[c]=p
    assert n==195;y=d['y'];slow=np.flatnonzero(y>=20);rng=np.random.default_rng(20261105);boot=rng.integers(0,594,(5000,594));sb=slow[rng.integers(0,9,(5000,9))];out={};unc={};held=[]
    for c,p in preds.items():
        e=np.abs(p-y);out[c]=metrics(y,p);out[c]['slow_bias']=float((p-y)[slow].mean());out[c]['goal_met']=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3);diff=e-np.abs(preds['phase_tail4']-y);unc[c]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),difference_phase_tail4=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((p-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=c,actual=float(a),prediction=float(pr)) for i,a,pr in zip(d['ids'],y,p))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap;slow9separately;reuseddevelopmentfolds,notclinical/external. Native4vsphase4isolatesinput;native16alsochangestraininglabelweights.',conditions=unc));save(R/'verification.json',dict(status='passed',fits=60,inner_checkpoint_replays=180,final_checkpoint_replays=15,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
