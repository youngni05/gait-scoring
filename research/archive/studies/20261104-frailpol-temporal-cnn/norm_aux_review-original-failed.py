"""All 325 saved states: raw/norm alignment, train-only scaling/weights and CPU64 replay."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'norm-aux';O=B/'norm-aux';sys.path.insert(0,str(H));from norm_aux import Model
CONDS=['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16','symnorm_aux_tail4'];GRID=[10,25,50,100]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def kind(c):return 'symnorm' if c.startswith('symnorm') else 'norm' if c.startswith('norm') else 'full'
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),score=max(float(e.mean())/.9,float(np.quantile(e,.95))/3,float(e[y>=20].mean())/3))
def audit(rec,d,tr,ev,c,seed,epochs,grid):
    ctx=read(rec['context']);assert ctx['condition']==c and ctx['train']==tr.tolist() and ctx['eval']==ev.tolist() and not set(tr)&set(ev) and ctx['train_ids']==d['ids'][tr].tolist() and ctx['eval_ids']==d['ids'][ev].tolist() and ctx['seed']==rec['seed']==seed and ctx['epochs']==rec['epochs']==epochs and ctx['protocol_sha256']==sha(R/'protocol.json');raw=d[kind(c)];mask=d['valid'];mu=[];sd=[]
    for j in range(raw.shape[1]):
        observed=raw[tr,j][mask[tr,0]];mean=observed.mean();std=observed.std();mu.append(mean);sd.append(std if std>=1e-8 else 1.)
    p=ctx['prep'];assert p['kind']==kind(c) and p['condition']==c and np.allclose(mu,p['channel_mean'],atol=1e-12,rtol=1e-12) and np.allclose(sd,p['channel_std'],atol=1e-12,rtol=1e-12);fm=d['F'][tr].mean(0);fs=d['F'][tr].std(0);fs[fs<1e-8]=1;assert np.array_equal(fm,p['manual_mean']) and np.array_equal(fs,p['manual_std']);ym=float(d['y'][tr].mean());ys=max(float(d['y'][tr].std()),1e-8);assert p['ym']==ym and p['ys']==ys;labels=d['y'][tr,None]>=np.array([12,20])[None,:];positive=labels.sum(0);assert np.array_equal((len(tr)-positive)/positive,p['aux_positive_weight'])
    values=np.where(mask[ev],(raw[ev]-np.array(p['channel_mean'])[None,:,None])/np.array(p['channel_std'])[None,:,None],0);x=np.concatenate([values,mask[ev].astype(float)],axis=1).astype(np.float32).astype(float);f=((d['F'][ev]-p['manual_mean'])/p['manual_std']).astype(np.float32).astype(float);assert [r['epoch'] for r in rec['curve']]==grid;delta=0;adelta=0
    for row in rec['curve']:
        assert sha(row['model'])==row['sha256'];net=Model(raw.shape[1],'_aux_' in c).double();net.load_state_dict(torch.load(row['model'],map_location='cpu',weights_only=True));net.eval();out=[];aux=[]
        with torch.no_grad():
            for lo in range(0,len(ev),64):
                r,a=net.forward_all(torch.from_numpy(x[lo:lo+64]),torch.from_numpy(f[lo:lo+64]));out.extend(r.numpy()*ys+ym)
                if a is not None:aux.extend(a.numpy())
        pred=np.array(out);original=np.array(row['prediction']);assert np.isfinite(pred).all() and np.allclose(pred,original,atol=1e-6,rtol=1e-6);delta=max(delta,float(np.max(np.abs(pred-original))))
        if '_aux_' in c:
            aa=np.array(aux);assert np.allclose(aa,row['aux_logits'],atol=1e-6,rtol=1e-6);adelta=max(adelta,float(np.max(np.abs(aa-row['aux_logits']))))
        else:assert row['aux_logits'] is None
        for key,v in metric(d['y'][ev],original).items():assert np.isclose(v,row['metrics'][key],atol=1e-12,rtol=1e-12)
    return delta,adelta
def main():
    torch.set_num_threads(2);p=read(R/'protocol.json');receipt=read(R/'receipt.json');assert p['code_sha256']==sha(H/'norm_aux.py') and p['native_library_sha256']==sha(H/'native_motion.py') and p['block_library_sha256']==sha(H/'bundle/train.py');assert p['source_original_sha256']==sha(B/'data.npz') and p['source_native_sha256']==sha(B/'native-motion/data.npz') and p['source_norm_sha256']==sha(B/'norm-input/inputs.npz');assert receipt['cases']==25 and receipt['protocol_sha256']==sha(R/'protocol.json')
    for path,value in receipt['hashes'].items():assert sha(path)==value
    orig=np.load(B/'data.npz');native=np.load(B/'native-motion/data.npz');norm=np.load(B/'norm-input/inputs.npz');ids=orig['ids'];y=orig['y'];fold=orig['fold'];assert len(ids)==594 and sum(y>=20)==9 and all(np.array_equal(native[k],orig[k]) for k in ['ids','y','fold']) and all(np.array_equal(norm[k],native[k]) for k in ['ids','fold','F']);full=native['native'][:,:24];n=full[:,[3,7,11,15,19,23]];sym=np.stack([(n[:,0]+n[:,2])/2,(n[:,1]+n[:,3])/2,np.abs(n[:,0]-n[:,2]),np.abs(n[:,1]-n[:,3]),n[:,4],n[:,5]],1);assert np.array_equal(n,norm['norm']) and np.array_equal(sym,norm['symmetric_norm']);d=dict(ids=ids,y=y,fold=fold,F=norm['F'],full=full,norm=n,symnorm=sym,valid=native['native'][:,-1:]>0);checks=[];preds={};count=0
    for c in CONDS:
      pred=np.full(594,np.nan)
      for f in range(5):
        rec=read(O/c/f'fold{f}/fold.json');ss=p['splits'][str(f)];tr=np.array(ss['train']);te=np.array(ss['test']);assert np.array_equal(tr,np.flatnonzero(fold!=f)) and np.array_equal(te,np.flatnonzero(fold==f)) and rec['test_ids']==ids[te].tolist();inner=np.zeros((4,len(tr)));seen=[];delta=0;adelta=0
        for k,(r,s) in enumerate(zip(rec['inner'],ss['inner'])):
            it=np.array(s['train']);iv=np.array(s['valid']);assert set(it)|set(iv)==set(tr) and not set(it)&set(iv);dd,aa=audit(r,d,it,iv,c,20261106+f*100+k*1000,100,GRID);delta=max(delta,dd);adelta=max(adelta,aa);positions=[np.flatnonzero(tr==i)[0] for i in iv]
            for j,row in enumerate(r['curve']):inner[j,positions]=row['prediction']
            seen.extend(iv);count+=4
        assert len(seen)==len(tr) and set(seen)==set(tr);scores=[metric(y[tr],a) for a in inner]
        for a,b in zip(scores,rec['inner_scores']):assert all(np.isclose(v,b[key],atol=1e-12,rtol=1e-12) for key,v in a.items())
        epoch=GRID[int(np.argmin([s['score'] for s in scores]))];assert rec['epoch']==epoch;dd,aa=audit(rec['final'],d,tr,te,c,20261106+f*100+5000,epoch,[epoch]);delta=max(delta,dd);adelta=max(adelta,aa);count+=1;assert rec['prediction']==rec['final']['curve'][0]['prediction'];pred[te]=rec['prediction'];checks.append(dict(condition=c,fold=f,epoch=epoch,checkpoints=13,max_cpu64_TUG_delta_seconds=delta,max_cpu64_aux_logit_delta=adelta,person_normalization_aux_weights_and_selection_verified=True));print('verified',c,f,flush=True)
      assert np.isfinite(pred).all();preds[c]=pred
    assert count==325;out={};unc={};held=[];rng=np.random.default_rng(20261106);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))]
    for c,pred in preds.items():
        e=np.abs(pred-y);out[c]=metric(y,pred);out[c]['slow_bias']=float((pred-y)[slow].mean());out[c]['goal_met']=bool(e.mean()<=.9 and np.quantile(e,.95)<=3 and e[slow].mean()<=3);control='symnorm_tail4' if c in ['symnorm_tail16','symnorm_aux_tail4'] else 'full_tail4';diff=e-np.abs(preds[control]-y);unc[c]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),matching_control=control,difference_control=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={label:dict(n=int(q.sum()),mae=float(e[q].mean()),bias=float((pred-y)[q].mean()),max_error=float(e[q].max())) for label,q in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]});held.extend(dict(id=int(i),condition=c,actual=float(a),prediction=float(pr)) for i,a,pr in zip(ids,y,pred))
    save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrap;slow9separately;reuseddevelopmentcohortnotexternalclinicalorselectionadjusted. Fullvsnormfirstprojectiondiffers;normvssymnormmatchedcount/init;auxvssymnormmatchedcoreinit/dropoutrng.',conditions=unc));save(R/'verification.json',dict(status='passed',fits=100,inner_checkpoint_replays=300,final_checkpoint_replays=25,all_auxiliary_logits_replayed=True,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
