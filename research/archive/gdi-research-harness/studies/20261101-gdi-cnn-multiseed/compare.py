"""Predeclared equal initialization average; no weight or seed selection."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
H=Path(__file__).resolve().parent;OLD=H.parent/'20261030-gpu-motionbert-study'
DATA=Path('D:/gait-research-data/D_P/gdi-research-harness/studies/20261101-gdi-cnn-multiseed')
sys.path.insert(0,str(H/'seed20261101'))
import torch
import run as r
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def combine(cnn,stt,bilateral,w,t):
    base=w*cnn+(1-w)*stt;mean=base.mean(1);diff=(1-t)*(base[:,0]-base[:,1])+t*(bilateral[:,0]-bilateral[:,1])
    return np.column_stack([mean+diff/2,mean-diff/2])
def main():
    save(H/'watch/comparison-progress.json',dict(phase='running',pid=os.getpid(),member_count=0))
    protocol=read(H/'protocol.json');assert protocol['approved_initialization_seeds']==[20261101,20261102] and not protocol['outer_selection']
    receipt=read(H/'checks/full-receipt.json');assert receipt['passed'] and len(receipt['seeds'])==2
    oldreport=read(OLD/'inner-bagging692/analysis/comparison.json');assert sha(oldreport['predictions_path'])==oldreport['predictions_sha256']
    old=arrays(oldreport['predictions_path']);md=arrays(OLD/'mean-difference692/analysis/predictions.npz');selection=read(OLD/'mean-difference692/analysis/selection.json')['folds']
    y=old['y'];ids=old['person_id'];fold=old['fold'];people=np.unique(ids)
    def errors(p):return np.array([abs(p[ids==i]-y[ids==i]).mean() for i in people])
    assert abs(errors(old['bagged']).mean()-6.515816569178735)<1e-10
    for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(old[k],md[k])
    reproduced=np.full_like(y,np.nan)
    for sf in selection:
        te=fold==sf['outer'];reproduced[te]=combine(old['CNN_bag'][te],old['STT_bag'][te],md['bilateral_STT'][te],sf['cnn_weight'],sf['fraction'])
    assert abs(reproduced-old['bagged']).max()<1e-10
    r.setup(7,'cpu');seed_predictions=[];members=[];sources={};replay_errors=[]
    for entry in receipt['seeds']:
        root=Path(entry['extracted']);r.ROOT=root;assert sha(entry['review'])==entry['review_sha256'] and read(entry['review'])['passed']
        d=r.data();c=r.config()
        for k in ['y','person_id','video_id','fold']:np.testing.assert_array_equal(d[k],old[k])
        assert c['seed']==20261028 and c['initialization_seed']==entry['seed']
        result=np.full_like(y,np.nan)
        for f in range(5):
            tr,_,te=r.split(d,f);parts=[]
            for j in [None]+[j for j in range(5) if j!=f]:
                folder=root/('runs/main/fold%d/%s'%(f,'refit' if j is None else 'inner%d'%j));done=r.load(folder/'fit_complete.json');ident=done['identity'];epochs=done['epochs']
                assert ident==r.identity_for(d,c,f,j,epochs,'outer_refit' if j is None else 'inner_epoch_selection')
                assert not set(ident['train_people'])&set(ids[te]) and not set(ident['validation_people'])&set(ids[te])
                ckpath=folder/'latest.pt';assert sha(ckpath)==done['checkpoint_sha256'];sources[str(ckpath)]=sha(ckpath)
                rawpath=folder/'raw_predictions'/('epoch%04d.npz'%epochs);history=r.load(folder/'curve.json');assert sha(rawpath)==history[-1]['raw_predictions_sha256']
                raw=arrays(rawpath);excluded=j is not None and bool((raw['train_raw'].std(0)<0.01).any())
                members.append(dict(seed=entry['seed'],outer=f,inner=j,excluded=excluded))
                save(H/'watch/comparison-progress.json',dict(phase='running',pid=os.getpid(),member_count=len(members),active_member=members[-1],completed_model_replays=len(replay_errors)))
                if excluded:continue
                ck=torch.load(ckpath,map_location='cpu',weights_only=True);model=r.GdiCNN(c['dropout']).double();model.load_state_dict(ck['model']);model.eval();coef=np.array(history[-1]['calibration'])
                def infer(mask):
                    x=d['x'][mask];chunks=[]
                    with torch.inference_mode():
                        for start in range(0,len(x.values),128):
                            t=r.channels(torch.from_numpy(x.values[start:start+128])).reshape(-1,12,124).double()
                            v=model(t)[:,0]*float(ck['span'])+float(ck['lo']);chunks.append(v.numpy().reshape(-1,2))
                    return x.aggregate(np.concatenate(chunks))
                raw_test=infer(te);pred=r.corrected(raw_test,coef)
                if j is None:
                    q=arrays(folder.parent/'predictions.npz');error=max(float(abs(pred-q['predictions'][:,0]).max()),float(abs(raw_test-q['predictions_raw'][:,0]).max()))
                else:
                    _,va,_=r.split(d,f,j);v=infer(va);error=max(float(abs(v-raw['validation_raw']).max()),float(abs(r.corrected(v,coef)-r.corrected(raw['validation_raw'],coef)).max()))
                assert error<1e-4,error;replay_errors.append(error);parts.append(pred)
            assert parts;result[te]=np.mean(parts,axis=0)
        seed_predictions.append(result)
    cnn=np.mean([old['CNN_bag']]+seed_predictions,axis=0);output=np.full_like(y,np.nan)
    for sf in selection:
        te=fold==sf['outer'];output[te]=combine(cnn[te],old['STT_bag'][te],md['bilateral_STT'][te],sf['cnn_weight'],sf['fraction'])
    variants=dict(baseline=old['bagged'],three_initialization_blend=output,old_CNN_bag=old['CNN_bag'],CNN_seed20261101=seed_predictions[0],CNN_seed20261102=seed_predictions[1],three_initialization_CNN=cnn)
    high=np.array([y[ids==i].mean()>=90 for i in people]);assert high.sum()==68
    def stats(p):
        e=errors(p);bias=np.array([(p[ids==i]-y[ids==i]).mean() for i in people])
        return dict(mae=float(e.mean()),bias=float(bias.mean()),high90=dict(people=int(high.sum()),mae=float(e[high].mean()),bias=float(bias[high].mean())),fold_mae=[float(np.mean([abs(p[(ids==i)]-y[ids==i]).mean() for i in np.unique(ids[fold==f])])) for f in range(5)])
    delta=errors(output)-errors(old['bagged']);rng=np.random.default_rng(20261101);boots=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(10000)])
    cache=DATA/'comparison';cache.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache/'predictions.npz',**variants,y=y,person_id=ids,video_id=old['video_id'],fold=fold)
    report=dict(passed=True,people=692,records=1061,metrics={k:stats(v) for k,v in variants.items()},new_minus_baseline=float(delta.mean()),paired_person_bootstrap95=np.quantile(boots,[.025,.975]).tolist(),target6_3_met=bool(errors(output).mean()<=6.3),members=members,maximum_replay_error=max(replay_errors),baseline_reproduction_max=float(abs(reproduced-old['bagged']).max()),protocol_sha256=sha(H/'protocol.json'),full_receipt_sha256=sha(H/'checks/full-receipt.json'),sources=sources,predictions_path=str(cache/'predictions.npz'),predictions_sha256=sha(cache/'predictions.npz'),independent_final_evaluation=False,seed_or_weight_selection_on_outer=False)
    save(H/'comparison.json',report);print(json.dumps({k:report[k] for k in ['metrics','new_minus_baseline','paired_person_bootstrap95','target6_3_met']}))
    save(H/'watch/comparison-progress.json',dict(phase='complete',pid=os.getpid(),member_count=len(members),completed_model_replays=len(replay_errors),report_sha256=sha(H/'comparison.json')))
if __name__=='__main__':main()
