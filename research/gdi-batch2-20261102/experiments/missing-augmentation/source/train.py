import csv,json,hashlib,sys,time,copy
from pathlib import Path
import numpy as np
S=Path(__file__).resolve().parent;H=S.parents[1];R=H.parent;PREV=H/'studies/20261028-rhythm-versus-missing-pooling';CV=H/'control/desktop-sessions/20261026T041807Z-f050df';LABELS=H/'control/desktop-sessions/20261026T130506Z-e876e7/length_internal_oof.csv'
for p in (R/'outputs/motionbert-smoke-20261015/deps',R/'outputs/gdi-ridge-experiment-20261016/deps'):sys.path.insert(0,str(p))
import torch
torch.set_num_threads(4);torch.use_deterministic_algorithms(True);sys.stdout.reconfigure(encoding='utf-8')
SEEDS=(20261028,20261029,20261030);ARMS=('baseline','extra_original','extra_missing');POSITIONS=('front','middle','last','stress')
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
    with (S/n).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def state_hash(model):
    hh=hashlib.sha256()
    for k,v in model.state_dict().items():hh.update(k.encode());hh.update(v.detach().numpy().tobytes())
    return hh.hexdigest()
def new_head(seed):
    torch.manual_seed(seed);m=torch.nn.Sequential(torch.nn.Linear(512,16),torch.nn.ReLU(),torch.nn.Linear(16,2)).double()
    torch.nn.init.zeros_(m[2].weight);torch.nn.init.zeros_(m[2].bias);return m

def train(m,z,base,y,steps=100):
    z=torch.from_numpy(z);base=torch.from_numpy(base);y=torch.from_numpy(y);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    history=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True);pred=base+m(z);loss=torch.mean((pred-y)**2);assert torch.isfinite(loss).item()
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters());opt.step()
        if step==0 or (step+1)%10==0:history.append(dict(step=step+1,loss_before_update=float(loss.detach())))
    with torch.inference_mode():final=float(torch.mean((base+m(z)-y)**2))
    return history,final

def identity_test():
    rng=np.random.default_rng(39);x=rng.normal(size=(24,512));b=rng.normal(size=(24,2));y=rng.normal(size=(24,2));m=new_head(123)
    with torch.inference_mode():assert torch.count_nonzero(m(torch.from_numpy(x)))==0
    first,second=copy.deepcopy(m),copy.deepcopy(m);train(first,x,b,y,8);train(second,x.copy(),b.copy(),y.copy(),8);assert state_hash(first)==state_hash(second)
    return dict(zero_initial_residual=True,identity_augmentation_same_parameters=True,steps=8,synthetic_only=True)
def persons(v,ids):return np.array([v[ids==p].mean() for p in sorted(set(ids))])
def comparison(a,b):
    # Each a/b is seed x person; average paired errors across seeds, not predictions.
    d=(a-b).mean(0);rng=np.random.default_rng(20261026);ci=np.percentile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[2.5,97.5])
    return dict(reduction=float(d.mean()),paired_bootstrap95=ci.tolist(),seed_reductions=(a-b).mean(1).tolist(),people_improved=int((d>0).sum()),people_worsened=int((d<0).sum()))
def main():
    start=time.perf_counter();assert not (S/'result.json').exists();assert load(S/'preparation.json')['verified'];assert sha(S/'features.npz')==load(S/'preparation.json')['features_sha256']
    save('identity_training_check.json',identity_test())
    receipt={a['path']:a['sha256'] for a in load(PREV/'study_receipt.json')['artifacts']}
    for n in ('fold_models.npz','features.npz','predictions.csv'):
        p=PREV/n;assert sha(p)==receipt[p.relative_to(H).as_posix()]
    old_receipt={a['path']:a['sha256'] for a in load(LABELS.parent/'receipt.json')['artifacts']};assert sha(LABELS)==old_receipt[LABELS.relative_to(H).as_posix()]
    protocol=load(CV/'internal_cv_protocol.json');assert sha(CV/'internal_cv_protocol.json')=={a['path']:a['sha256'] for a in load(CV/'receipt.json')['artifacts']}[(CV/'internal_cv_protocol.json').relative_to(H).as_posix()]
    records=list(csv.DictReader(LABELS.open(encoding='utf-8')));lab={r['video_id']:r for r in records};assert len(records)==len(lab)==203
    with np.load(S/'features.npz',allow_pickle=False) as z:
        ids,vid=(z[k].copy() for k in ('patient_ids','video_ids'));x={p:z[p].copy() for p in POSITIONS};aug=z['training_aug'].copy()
    assert len(vid)==203 and len(set(ids))==175 and set(map(str,vid))==set(lab)
    y=np.array([[float(lab[str(v)]['actual_L']),float(lab[str(v)]['actual_R'])] for v in vid]);folds=np.array([int(lab[str(v)]['fold']) for v in vid])
    for i,v in enumerate(vid):assert str(ids[i])==lab[str(v)]['patient_id']
    for k,pp in enumerate(protocol['fold_people']):assert set(map(str,ids[folds==k]))==set(pp)
    with np.load(PREV/'fold_models.npz',allow_pickle=False) as z:old={k:z[k].copy() for k in z.files if k.startswith('baseline_')}
    preds={arm:{pos:np.full((3,203,2),np.nan) for pos in POSITIONS} for arm in ARMS};model_params={};logs=[]
    for fold in range(5):
        tr=folds!=fold;te=~tr;assert len(set(ids[tr]))==140 and len(set(ids[te]))==35 and not set(ids[tr])&set(ids[te])
        base={n:old[f'baseline_fold{fold}__{n}'] for n in ('scaler_mean','scaler_scale','coef','intercept')}
        mu,scale=base['scaler_mean'],base['scaler_scale'];np.testing.assert_allclose(mu,x['front'][tr].mean(0),rtol=0,atol=1e-12)
        ym=y[tr].mean(0);ys=y[tr].std(0);assert (ys>0).all()
        def zs(xx):return (xx-mu)/scale
        def bp(zz):return zz@base['coef'].T+base['intercept']
        zorig=zs(x['front'][tr]);zaug=zs(aug[tr]);target=np.tile((y[tr]-ym)/ys,(2,1))
        model_params.update({f'fold{fold}__{k}':v for k,v in {**base,'y_mean':ym,'y_scale':ys}.items()})
        for si,seed in enumerate(SEEDS):
            init=new_head(seed);ih=state_hash(init)
            with torch.inference_mode():assert torch.count_nonzero(init(torch.from_numpy(zorig)))==0
            for pos in POSITIONS:preds['baseline'][pos][si,te]=bp(zs(x[pos][te]))
            pair=[]
            for arm,second in (('extra_original',zorig),('extra_missing',zaug)):
                model=copy.deepcopy(init);assert state_hash(model)==ih
                data=np.concatenate([zorig,second]);base_norm=(bp(data)-ym)/ys
                assert data.shape==(2*tr.sum(),512) and target.shape==(2*tr.sum(),2)
                history,final=train(model,data,base_norm,target)
                with torch.inference_mode():
                    for pos in POSITIONS:
                        zz=zs(x[pos][te]);preds[arm][pos][si,te]=bp(zz)+model(torch.from_numpy(zz)).numpy()*ys
                params={'w1':model[0].weight.detach().numpy().copy(),'b1':model[0].bias.detach().numpy().copy(),'w2':model[2].weight.detach().numpy().copy(),'b2':model[2].bias.detach().numpy().copy()}
                model_params.update({f'{arm}_seed{seed}_fold{fold}__{k}':v for k,v in params.items()})
                pair.append(dict(arm=arm,seed=seed,fold=fold,initial_state_sha256=ih,updates=100,batch_rows=int(len(data)),targets_sha256=hashlib.sha256(target.tobytes()).hexdigest(),data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),first_half_sha256=hashlib.sha256(data[:tr.sum()].tobytes()).hexdigest(),final_loss=final,history=history))
            assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256'] and pair[0]['targets_sha256']==pair[1]['targets_sha256'] and pair[0]['first_half_sha256']==pair[1]['first_half_sha256'];logs.extend(pair)
        print(json.dumps(dict(fold_finished=fold,seconds=time.perf_counter()-start)),flush=True)
    oldpred=np.array([[float(lab[str(v)]['short124_L']),float(lab[str(v)]['short124_R'])] for v in vid]);replay=float(np.max(np.abs(preds['baseline']['front'][0]-oldpred)));assert replay<1e-7
    np.savez_compressed(S/'models.npz',**model_params);save('training_logs.json',logs)
    for arm in ARMS:
        for pos in POSITIONS:assert np.isfinite(preds[arm][pos]).all()
        preds[arm]['mean3']=np.stack([preds[arm][p] for p in ('front','middle','last')]).mean(0)
    errors={};ranges={};shifts={};stats={}
    for arm,pp in preds.items():
        errors[arm]={pos:np.stack([persons(np.abs(v[s]-y).mean(1),ids) for s in range(3)]) for pos,v in pp.items()}
        ranges[arm]=np.stack([persons(np.ptp(np.stack([pp[p][s] for p in ('front','middle','last')]),axis=0).mean(1),ids) for s in range(3)])
        shifts[arm]=np.stack([persons(np.abs(pp['stress'][s]-pp['front'][s]).mean(1),ids) for s in range(3)])
        stats[arm]=dict(person_mae_seedmean={pos:float(v.mean()) for pos,v in errors[arm].items()},person_mae_by_seed={pos:v.mean(1).tolist() for pos,v in errors[arm].items()},range_seedmean=float(ranges[arm].mean()),stress_shift_seedmean=float(shifts[arm].mean()),range_by_seed=ranges[arm].mean(1).tolist(),stress_shift_by_seed=shifts[arm].mean(1).tolist(),front_prediction_sd_by_seed=[float(np.std(persons(pp['front'][s].mean(1),ids),ddof=1)) for s in range(3)],front_fold_mae_seedmean={str(k):float(np.mean([persons(np.abs(pp['front'][s,folds==k]-y[folds==k]).mean(1),ids[folds==k]).mean() for s in range(3)])) for k in range(5)})
    comparisons={}
    for ref,can in (('baseline','extra_original'),('extra_original','extra_missing'),('baseline','extra_missing')):
        comparisons[can+'_vs_'+ref]=dict(mae={pos:comparison(errors[ref][pos],errors[can][pos]) for pos in errors[ref]},range=comparison(ranges[ref],ranges[can]),stress_shift=comparison(shifts[ref],shifts[can]),front_folds_improved=sum(stats[can]['front_fold_mae_seedmean'][str(k)]<stats[ref]['front_fold_mae_seedmean'][str(k)] for k in range(5)))
    with (S/'predictions.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['patient_id','video_id','fold','seed','arm','position','actual_L','actual_R','prediction_L','prediction_R'])
        for arm,pp in preds.items():
            for pos,values in pp.items():
                for si,seed in enumerate(SEEDS):
                    for i in range(203):w.writerow([ids[i],vid[i],folds[i],seed,arm,pos,*y[i],*values[si,i]])
    tracked={p.relative_to(R).as_posix():sha(p) for p in (Path(__file__),S/'plan.md',S/'features.npz',PREV/'fold_models.npz',CV/'internal_cv_protocol.json',LABELS)};save('analysis_sources.json',tracked)
    result=dict(people=175,records=203,folds=5,seeds=list(SEEDS),updates_per_model=100,hidden_width=16,residual_parameter_count=8242,trained_residual_models=30,encoder_trainable=False,hyperparameter_search=False,validation37_used=False,protected38_or223_used=False,baseline_replay_max=replay,metrics=stats,comparisons=comparisons,seconds=time.perf_counter()-start,seed_average_is_not_ensemble=True,independent_final_performance=False)
    save('result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
