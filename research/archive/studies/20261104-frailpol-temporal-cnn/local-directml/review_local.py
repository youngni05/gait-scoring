"""All-checkpoint CPU64 replay and frozen-person/epoch/normalization audit."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,torch
H=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');B=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');LOCAL=H/'local-directml';OUT=B/'local-main';sys.path.insert(0,str(H/'bundle'))
from train import Model,CONDS
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()))
def audit_fit(rec,data,tr,ev,c,seed,epochs,grid,role,protocol):
    ctx=read(rec['context']);assert ctx['condition']==rec['condition']==c and ctx['seed']==rec['seed']==seed and ctx['epochs']==rec['epochs']==epochs and ctx['role']==role
    assert ctx['train_indices']==tr.tolist() and ctx['eval_indices']==ev.tolist() and ctx['train_ids']==data['ids'][tr].tolist() and ctx['eval_ids']==data['ids'][ev].tolist();assert not set(tr)&set(ev);assert ctx['protocol_sha256']==sha(LOCAL/'main-protocol.json') and ctx['data_sha256']==protocol['data_sha256']
    nf=1 if c.endswith('speed') or c=='speed_mlp' else data['F'].shape[1];F=data['F'][:,:nf];raw=F[tr];med=np.array([np.median(q[np.isfinite(q)]) if np.any(np.isfinite(q)) else 0. for q in raw.T]);a=np.where(np.isfinite(raw),raw,med);mean=a.mean(axis=0);std=a.std(axis=0);std[std<1e-8]=1.;ym=float(data['y'][tr].mean());ys=max(float(data['y'][tr].std()),1e-8);prep=ctx['preprocessing'];assert prep['nfeatures']==nf and prep['condition']==c;assert np.array_equal(med,prep['median']);assert np.allclose(mean,prep['mean'],atol=1e-12,rtol=1e-12) and np.allclose(std,prep['std'],atol=1e-12,rtol=1e-12);assert np.isclose(ym,prep['target_mean'],atol=1e-12,rtol=1e-12) and np.isclose(ys,prep['target_std'],atol=1e-12,rtol=1e-12)
    a=(np.where(np.isfinite(F[ev]),F[ev],med)-mean)/std;x=data['X'][ev]
    # The actual GPU/CPU model receives float32 tensors. Replay those exact inputs in CPU64.
    x64=torch.from_numpy(x.astype(np.float32).astype(np.float64));f64=torch.from_numpy(a.astype(np.float32).astype(np.float64));curve=read(rec['curve']);assert [r['epoch'] for r in curve]==grid;maxdelta=0.
    for row in curve:
        assert sha(row['checkpoint'])==row['checkpoint_sha256'];model=Model(c,nf).double();model.load_state_dict(torch.load(row['checkpoint'],map_location='cpu',weights_only=True));model.eval();pp=[]
        with torch.no_grad():
            for lo in range(0,len(ev),64):pp.extend(model(x64[lo:lo+64],f64[lo:lo+64]).numpy()*ys+ym)
        p=np.array(pp,float);original=np.array(row['prediction']);assert np.isfinite(p).all() and np.allclose(p,original,atol=1e-6,rtol=1e-6);maxdelta=max(maxdelta,float(np.max(np.abs(p-original))));m=metric(data['y'][ev],original)
        for key,val in m.items():assert np.isclose(val,row['metrics'][key],atol=1e-12,rtol=1e-12)
        assert np.isclose(row['metrics']['score'],max(m['mae']/.9,m['p95']/3,m['slow_mae']/3),atol=1e-12,rtol=1e-12)
    assert curve[-1]['prediction']==rec['prediction'] and curve[-1]['checkpoint']==rec['checkpoint'] and sha(rec['checkpoint'])==rec['checkpoint_sha256'];return curve,maxdelta
def main():
    torch.set_num_threads(2);proto=read(LOCAL/'main-protocol.json');assert proto['code_sha256']==sha(LOCAL/'train_local.py') and proto['model_library_sha256']==sha(H/'bundle/train.py') and proto['base_protocol_sha256']==sha(LOCAL/'base-protocol.json') and proto['data_sha256']==sha(B/'data.npz');data=np.load(B/'data.npz');ids=data['ids'];y=data['y'];fold=data['fold'];assert len(y)==594 and sum(y>=20)==9
    source=read('D:/gait-research-data/D_P/frailpol-short-walk-20261103/cohorts.json');source=[r for r in source if r['status']=='ok' and r['tug'] is not None];assert [r['id'] for r in source]==ids.tolist() and np.array_equal([r['tug'] for r in source],y) and np.array_equal([r['fold'] for r in source],fold)
    for worker in ['controls','motion']:
        receipt=read(LOCAL/(worker+'-receipt.json'));assert receipt['cases']==10 and receipt['protocol_sha256']==sha(LOCAL/'main-protocol.json')
        for p,v in receipt['hashes'].items():assert sha(p)==v
    out={};checks=[];oof=[];saved_checkpoints=0
    for c in CONDS:
      pred=np.zeros(len(y));seen=[];ci=CONDS.index(c)
      for f in range(5):
        split=proto['splits'][str(f)];tr=np.array(split['train']);te=np.array(split['test']);assert np.array_equal(tr,np.flatnonzero(fold!=f)) and np.array_equal(te,np.flatnonzero(fold==f));rec=read(OUT/c/f'fold{f}/complete.json');assert rec['test_ids']==ids[te].tolist() and rec['protocol_sha256']==sha(LOCAL/'main-protocol.json');inner=[];maxdelta=0.
        for k,ss in enumerate(split['inner']):
            it=np.array(ss['train']);iv=np.array(ss['valid']);assert not set(it)&set(iv) and set(it)|set(iv)==set(tr);curve,delta=audit_fit(rec['inner'][k],data,it,iv,c,20261104+f*100+ci+k*1000,100,[10,25,50,100],'inner_validation',proto);inner.append(curve);maxdelta=max(maxdelta,delta);saved_checkpoints+=4
        scores=[]
        for j,epoch in enumerate([10,25,50,100]):
            ip=np.zeros(len(tr));seeninner=[]
            for ss,curve in zip(split['inner'],inner):
                ix=[int(np.flatnonzero(tr==i)[0]) for i in ss['valid']];ip[ix]=curve[j]['prediction'];seeninner+=ss['valid']
            assert len(seeninner)==len(tr) and set(seeninner)==set(tr);m=metric(y[tr],ip);s=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);assert np.isclose(s,rec['inner_scores'][j]['score'],atol=1e-12,rtol=1e-12);scores.append(s)
        selected=[10,25,50,100][int(np.argmin(scores))];assert selected==rec['selected_epoch'];curve,delta=audit_fit(rec['final'],data,tr,te,c,20261104+f*100+ci+5000,selected,[selected],'outer_test',proto);saved_checkpoints+=1;assert rec['prediction']==curve[-1]['prediction'];pred[te]=rec['prediction'];seen.extend(te);checks.append(dict(condition=c,fold=f,selected_epoch=selected,training_only_normalization_verified=True,inner_epoch_selection_verified=True,saved_checkpoints_replayed=13,max_cpu64_replay_delta_seconds=max(maxdelta,delta)));print('verified',c,f,flush=True)
      assert len(seen)==594 and set(seen)==set(range(594));m=metric(y,pred);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[c]=m;oof.extend(dict(id=int(i),condition=c,actual=float(actual),prediction=float(pr)) for i,actual,pr in zip(ids,y,pred))
    assert saved_checkpoints==260
    save(OUT/'heldout.json',oof);save(LOCAL/'comparison.json',out);rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];uncertainty={}
    for c in CONDS:
        rows=sorted([r for r in oof if r['condition']==c],key=lambda r:r['id']);lookup={r['id']:r['prediction'] for r in rows};p=np.array([lookup[int(i)] for i in ids]);e=np.abs(p-y);control='speed_mlp' if c=='motion_speed' else 'features_mlp' if c=='motion_features' else c;clook={r['id']:r['prediction'] for r in oof if r['condition']==control};ce=np.abs(np.array([clook[int(i)] for i in ids])-y);d=e-ce
        uncertainty[c]=dict(mae_ci95=np.quantile(e[boot].mean(axis=1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(axis=1),[.025,.975]).tolist(),mae_difference_matching_control=float(d.mean()),mae_difference_ci95=np.quantile(d[boot].mean(axis=1),[.025,.975]).tolist(),bands={label:dict(n=int(mask.sum()),mae=float(e[mask].mean()),bias=float((p-y)[mask].mean()),max_error=float(e[mask].max())) for label,mask in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]})
    save(LOCAL/'uncertainty.json',dict(method='5000pairedpersonbootstrap;slow9separately;conditionalreuseddevelopmentfolds,notselection-adjustedorclinicalexternalCI.',conditions=uncertainty));save(LOCAL/'verification.json',dict(status='passed',fits=80,inner_checkpoint_replays=240,final_checkpoint_replays=20,checks=checks,comparison_sha256=sha(LOCAL/'comparison.json'),heldout_sha256=sha(OUT/'heldout.json'),review_code_sha256=sha(Path(__file__))));print(out)
if __name__=='__main__':main()
