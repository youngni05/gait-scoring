"""CPU64 replay of all supervised and masked-pretraining states with exact person splits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from torch import nn
H=Path('D:/Projects/D_P/studies/20261104-mediapipe-functional-loop');B=Path('D:/gait-research-data/D_P/mediapipe-functional-loop-20261104');R=H/'motion-pretrain';O=B/'motion-pretrain';sys.path.insert(0,str(H));from mp_compare14 import Net
VAR=['pelvis_stable_xy','pair_mean_absdiff'];MODES=['supervised','masked_pretrain'];GRID=[5,20,50]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def audit(rec,x,m,y,ids,tr,ev,seed,epochs,grid,mode):
    ctx=read(rec['context']);assert ctx['train']==tr.tolist() and ctx['eval']==ev.tolist() and not set(tr)&set(ev) and ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[ev].tolist();assert ctx['seed']==seed and ctx['epochs']==epochs and ctx['mode']==mode and ctx['protocol_sha256']==sha(R/'protocol.json');count=np.maximum(m[tr].sum((0,1)),1);mu=np.where(m[tr],x[tr],np.float32(0)).sum((0,1))/count;std=np.sqrt(np.where(m[tr],(x[tr]-mu)**2,0).sum((0,1))/count);std[std<1e-8]=1;p=ctx['prep'];assert np.array_equal(mu,p['mean']) and np.array_equal(std,p['std']) and p['ym']==y[tr].mean() and p['ys']==max(float(y[tr].std()),1e-8)
    a=np.concatenate([np.where(m,(x-mu)/std,0),m.astype(float)],axis=-1).transpose(0,2,1).astype(np.float32).astype(float);C=x.shape[-1];maxdelta=0;precount=0;precheck=None
    if mode=='masked_pretrain':
        pre=rec['pretrain'];assert pre['epochs']==100 and sha(pre['model'])==pre['sha256'];state=torch.load(pre['model'],map_location='cpu',weights_only=True);model=Net(2*C).double();model.conv.load_state_dict(state['encoder']);decoder=nn.Conv1d(16,C,1).double();decoder.load_state_dict(state['decoder']);corruption=np.load(pre['probe'])['corruption'];assert corruption.shape==(len(tr),C,120) and corruption.any() and np.all(~corruption|(a[tr,C:]>0));assert np.array_equal(corruption[:,::2],corruption[:,1::2]) or np.any(a[tr,C::2]!=a[tr,C+1::2]);noisy=a[tr].copy();noisy[:,:C]*=~corruption;noisy[:,C:]*=~corruption;model.eval();decoder.eval()
        with torch.no_grad():r=decoder(model.conv(torch.from_numpy(noisy))).numpy()
        original=np.load(pre['replay'])['prediction'];assert np.isfinite(r).all();error=np.abs(r-original);failed=error>1e-6+1e-6*np.abs(original);precheck=dict(context=rec['context'],cpu64_strict_passed=bool(not failed.any()),failed_points=int(failed.sum()),total_points=int(failed.size),max_cpu64_delta=float(error.max()))
        # Keep the failed CPU64 check as failed, without loosening its tolerance.
        # Same-backend CPU32 exact replay is additional diagnostic evidence only.
        model32=Net(2*C);model32.conv.load_state_dict(state['encoder']);decoder32=nn.Conv1d(16,C,1);decoder32.load_state_dict(state['decoder']);model32.eval();decoder32.eval()
        with torch.no_grad():same=decoder32(model32.conv(torch.from_numpy(noisy.astype(np.float32)))).numpy()
        assert np.array_equal(same,original);precheck['same_cpu32_exact_replay']=True;precount=1
    else:assert rec['pretrain'] is None
    assert [r['epoch'] for r in rec['curve']]==grid
    for r in rec['curve']:
        assert sha(r['model'])==r['sha256'];model=Net(2*C).double();model.load_state_dict(torch.load(r['model'],map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():pr=model(torch.from_numpy(a[ev])).numpy()*p['ys']+p['ym']
        assert np.allclose(pr,r['prediction'],atol=1e-6,rtol=1e-6);maxdelta=max(maxdelta,float(np.max(np.abs(pr-r['prediction']))))
    return maxdelta,precount,precheck
def main():
    torch.set_num_threads(2);proto=read(R/'protocol.json');receipt=read(R/'receipt.json');assert proto['code_sha256']==sha(H/'mp_pretrain14.py') and proto['base_library_sha256']==sha(H/'mp_compare14.py') and proto['input_sha256']==sha(B/'bottom14-inputs.npz') and proto['label_sha256']==sha(H/'clinical-label-audit.json');assert receipt['cases']==56 and receipt['protocol_sha256']==sha(R/'protocol.json')
    for p,v in receipt['hashes'].items():assert sha(p)==v
    raw=np.load(B/'bottom14-inputs.npz');ids=np.array(sorted(set(raw['person_ids'])));ix=np.array([np.flatnonzero(raw['person_ids']==pid)[0] for pid in ids]);assert proto['ids']==ids.tolist() and proto['window_indices']==ix.tolist();labels={r['person_id']:r['tug'] for r in read(H/'clinical-label-audit.json')['labels']};y=np.array([labels[pid] for pid in ids]);preds={};checks=[];prechecks=[];n=0;pn=0
    for v in VAR:
      x=raw[v][ix];m=np.repeat(raw['pelvis_mask' if v=='pelvis_stable_xy' else 'pair_mask'][ix],2,-1);vi=['image_xy','pelvis_stable_xy','relative_mp_xyz','bones_velocity','pair_mean_absdiff'].index(v)
      for mode in MODES:
        p=[]
        for outer in range(14):
            rec=read(O/v/mode/f'fold{outer}/fold.json');tr=np.array([i for i in range(14) if i!=outer]);ev=np.array([outer]);groups=np.array_split(np.random.default_rng(20261104+outer).permutation(tr),3);inner=np.zeros((3,14));delta=0
            for k,(r,iv) in enumerate(zip(rec['inner'],groups)):
                it=np.array([i for i in tr if i not in iv]);dd,pp,pc=audit(r,x,m,y,ids,it,iv,20261104+outer*100+vi+k*1000,50,GRID,mode);delta=max(delta,dd);pn+=pp;n+=3
                if pc is not None:prechecks.append(pc)
                for j,row in enumerate(r['curve']):inner[j,iv]=row['prediction']
            scores=np.abs(inner[:,tr]-y[tr]).mean(1);assert np.allclose(scores,rec['inner_mae'],atol=1e-12,rtol=1e-12);epoch=GRID[int(np.argmin(scores))];assert rec['epoch']==epoch;dd,pp,pc=audit(rec['final'],x,m,y,ids,tr,ev,20261104+outer*100+vi+5000,epoch,[epoch],mode);delta=max(delta,dd);pn+=pp;n+=1;assert rec['prediction']==rec['final']['curve'][0]['prediction'][0];p.append(rec['prediction']);checks.append(dict(variant=v,mode=mode,outer=outer,epoch=epoch,checkpoints=10,pretrained_states=4 if mode=='masked_pretrain' else 0,cpu64_max_TUG_delta_seconds=delta,unsupervised_and_supervised_person_separation_verified=True))
            if pc is not None:prechecks.append(pc)
        preds[v+'/'+mode]=np.array(p);print('verified',v,mode,flush=True)
    assert n==560 and pn==112;rng=np.random.default_rng(20261104);boot=rng.integers(0,14,(5000,14));out={};unc={};held=[]
    for key,p in preds.items():
        e=np.abs(p-y);out[key]=dict(n=14,mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_n=1,slow_error=float(e[y>=20][0]),bias=float((p-y).mean()));control=key.replace('/masked_pretrain','/supervised');diff=e-np.abs(preds[control]-y);unc[key]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),difference_matched_supervised=float(diff.mean()),difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist());held.extend(dict(id=str(pid),condition=key,actual=float(a),prediction=float(pr)) for pid,a,pr in zip(ids,y,p))
    failed=[r for r in prechecks if not r['cpu64_strict_passed']];save(O/'heldout.json',held);save(R/'comparison.json',out);save(R/'uncertainty.json',dict(method='5000pairedpersonbootstrapon14;1severe;conditionalexploratorydevelopmentpipeline,notclinical/external/selectionadjusted. FullpretrainingCPU64verificationfailed;do notpromoteasfullyverifiedpipeline.',conditions=unc));save(R/'verification.json',dict(status='failed_pretraining_reconstruction_cpu64_tolerance' if failed else 'passed',people=14,cnn_fits=224,TUG_CPU64_replay_status='passed',inner_checkpoint_replays=504,final_checkpoint_replays=56,pretrained_states_and_reconstruction_probes_replayed=112,pretrained_CPU64_failed_states=len(failed),pretrain_checks=prechecks,checks=checks,comparison_sha256=sha(R/'comparison.json'),heldout_sha256=sha(O/'heldout.json'),review_code_sha256=sha(Path(__file__)),limits='CPU64reconstructiontolerance1e-6absolute+relative unchanged;all112CPU32reconstructionsrequiredexactbitwise. FailedCPU64statesretained,notpromotedasfullverifiedpretraining. All560TUGcheckpointsstillrequireoriginalCPU64tolerance.'));print(out);print('pretraining CPU64 failures',len(failed));return bool(failed)
if __name__=='__main__':raise SystemExit(1 if main() else 0)
