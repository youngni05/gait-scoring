"""Independent manual-only NumPy + CPU64 replay and exact training-person audit."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'manual-controls';O=B/'manual-controls';sys.path.insert(0,str(H));from manual_controls import Model
CONDS=['speed_tail4','speed_tail16','speed_aux_tail4','speed_duration_tail4','speed_duration_aux_tail4'];GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def audit(rec,F,y,ids,tr,ev,c,seed,epochs,grid):
    ctx=read(rec['context']);assert ctx['condition']==c and ctx['train']==tr.tolist() and ctx['eval']==ev.tolist() and not set(tr)&set(ev) and ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[ev].tolist() and ctx['seed']==rec['seed']==seed and ctx['epochs']==rec['epochs']==epochs and ctx['protocol_sha256']==sha(R/'protocol.json');raw=F.copy()
    if 'duration' not in c:raw[:,1]=0
    mu=raw[tr].mean(0);sd=raw[tr].std(0);sd[sd<1e-8]=1;ym=float(y[tr].mean());ys=max(float(y[tr].std()),1e-8);positive=(y[tr,None]>=np.array([12,20])[None,:]).sum(0);p=ctx['prep'];assert np.array_equal(mu,p['mean']) and np.array_equal(sd,p['std']) and p['ym']==ym and p['ys']==ys and np.array_equal((len(tr)-positive)/positive,p['positive_weight']);f=((raw[ev]-mu)/sd).astype(np.float32).astype(float)
    if 'duration' not in c:assert np.all(f[:,1]==0)
    assert [r['epoch'] for r in rec['curve']]==grid;torch.manual_seed(seed);initial=Model('_aux_' in c).state_dict();delta=0;adelta=0
    def silu(a):return a/(1+np.exp(-a))
    for row in rec['curve']:
        assert sha(row['model'])==row['sha256'];state=torch.load(row['model'],map_location='cpu',weights_only=True);assert all(torch.equal(v,initial[k]) for k,v in state.items() if k.startswith('reference.core.conv.'))
        net=Model('_aux_' in c).double();net.load_state_dict(state);net.eval()
        with torch.no_grad():r,a=net.forward_all(torch.from_numpy(f));pr=r.numpy()*ys+ym
        original=np.array(row['prediction']);assert np.allclose(pr,original,atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(pr-original))));w={k:v.numpy().astype(float) for k,v in state.items()};manual=silu(f@w['reference.core.manual.0.weight'].T+w['reference.core.manual.0.bias']);latent=np.concatenate([manual,np.zeros((len(ev),16)),np.full((len(ev),16),1e-4)],1);hidden=silu(latent@w['reference.core.head.0.weight'].T+w['reference.core.head.0.bias']);independent=(hidden@w['reference.core.head.3.weight'].T+w['reference.core.head.3.bias']).ravel()*ys+ym;assert np.allclose(independent,pr,atol=1e-12,rtol=1e-12)
        if '_aux_' in c:
            logits=a.numpy();numpy_logits=latent@w['reference.aux.weight'].T+w['reference.aux.bias'];assert np.allclose(logits,numpy_logits,atol=1e-12,rtol=1e-12) and np.allclose(logits,row['aux_logits'],atol=1e-6,rtol=1e-6);adelta=max(adelta,float(np.max(np.abs(logits-row['aux_logits']))))
        else:assert row['aux_logits'] is None
        for key,v in metric(y[ev],original).items():assert np.isclose(v,row['metrics'][key],atol=1e-12,rtol=1e-12)
    return delta,adelta
def main():
    torch.set_num_threads(2);p=read(R/'protocol.json');receipt=read(R/'receipt.json');assert receipt['cases']==25 and receipt['protocol_sha256']==sha(R/'protocol.json');assert p['code_sha256']==sha(H/'manual_controls.py') and p['wave_library_sha256']==sha(H/'norm_aux.py') and p['native_library_sha256']==sha(H/'native_motion.py') and p['block_library_sha256']==sha(H/'bundle/train.py') and p['wave_protocol_sha256']==sha(H/'norm-aux/protocol.json');assert p['original_data_sha256']==sha(B/'data.npz') and p['native_data_sha256']==sha(B/'native-motion/data.npz')
    for path,value in receipt['hashes'].items():assert sha(path)==value
    orig=np.load(B/'data.npz');native=np.load(B/'native-motion/data.npz');assert all(np.array_equal(orig[k],native[k]) for k in ['ids','y','fold']);ids=orig['ids'];y=orig['y'];fold=orig['fold'];F=native['F'];assert len(ids)==594 and sum(y>=20)==9 and np.array_equal(F[:,0],orig['F'][:,0]);windows={r['id']:r['window'] for r in read('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103/round05/features.json')};assert np.array_equal(F[:,1],[(windows[int(i)]['core_end']-windows[int(i)]['core_start'])/100 for i in ids]);assert p['splits']==read(H/'norm-aux/protocol.json')['splits'];preds={};checks=[];count=0
    for c in CONDS:
      pred=np.full(594,np.nan)
      for f in range(5):
        rec=read(O/c/f'fold{f}/fold.json');ss=p['splits'][str(f)];tr=np.array(ss['train']);te=np.array(ss['test']);assert np.array_equal(tr,np.flatnonzero(fold!=f)) and np.array_equal(te,np.flatnonzero(fold==f)) and rec['test_ids']==ids[te].tolist();inner=np.zeros((4,len(tr)));seen=[];delta=0;adelta=0
        for k,(r,s) in enumerate(zip(rec['inner'],ss['inner'])):
            it=np.array(s['train']);iv=np.array(s['valid']);assert not set(it)&set(iv) and set(it)|set(iv)==set(tr);dd,aa=audit(r,F,y,ids,it,iv,c,20261106+f*100+k*1000,100,GRID);delta=max(delta,dd);adelta=max(adelta,aa);positions=[np.flatnonzero(tr==i)[0] for i in iv]
            for j,row in enumerate(r['curve']):inner[j,positions]=row['prediction']
            count+=4;seen.extend(iv)
        assert len(seen)==len(tr) and set(seen)==set(tr);scores=[metric(y[tr],a) for a in inner]
        for a,b in zip(scores,rec['inner_scores']):assert all(np.isclose(v,b[key],atol=1e-12,rtol=1e-12) for key,v in a.items())
        epoch=GRID[int(np.argmin([m['score'] for m in scores]))];assert epoch==rec['epoch'];dd,aa=audit(rec['final'],F,y,ids,tr,te,c,20261106+f*100+5000,epoch,[epoch]);delta=max(delta,dd);adelta=max(adelta,aa);count+=1;assert rec['prediction']==rec['final']['curve'][0]['prediction'];pred[te]=rec['prediction'];checks.append(dict(condition=c,fold=f,epoch=epoch,checkpoint_replays=13,max_cpu64_TUG_delta_seconds=delta,max_aux_logit_delta=adelta,numpy_head_replay_passed=True,encoder_weights_unchanged_from_initialization=True,source_speed_window_duration_and_training_scalers_verified=True));print('verified',c,f,flush=True)
      assert np.isfinite(pred).all();preds[c]=pred
    assert count==325;out={};unc={};held=[];rng=np.random.default_rng(20261106);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))]
    for c,pred in preds.items():
        e=np.abs(pred-y);out[c]=metric(y,pred);out[c]['slow_bias']=float((pred-y)[slow].mean());out[c]['goal_met']=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3);unc[c]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),p95_ci95=np.quantile(np.quantile(e[boot],.95,axis=1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((pred-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=c,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,pred))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000personbootstrap;slow9separately;reuseddevelopmentcohortnotexternalclinical/selectionadjusted.',conditions=unc));save(R/'verification.json',dict(status='passed',fits=100,inner_checkpoint_replays=300,final_checkpoint_replays=25,numpy_and_CPU64_replay=True,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
