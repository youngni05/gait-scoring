"""Original CUDA artifacts: independent persons, train transforms, epochs and CPU64."""
from school_gpu_common import *
import sys,importlib.util,datetime
import numpy as np,torch
CONDS=['speed_mlp','features_mlp','motion_speed','motion_features'];GRID=[10,25,50,100]
def metric(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()))
def library(root):
    assert sha(root/'train.py')==sha(H/'bundle/train.py');assert sha(root/'protocol.json')==sha(S/'protocol.json')
    spec=importlib.util.spec_from_file_location('original_cuda_train',root/'train.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def audit_fit(module,root,rec,data,tr,ev,c,seed,epochs,grid):
    ctx=read(root/rec['context']);assert ctx['condition']==c and ctx['seed']==seed and ctx['train_indices']==tr.tolist() and ctx['eval_indices']==ev.tolist();assert not set(tr)&set(ev);assert ctx['train_ids']==data['ids'][tr].tolist() and ctx['eval_ids']==data['ids'][ev].tolist();assert ctx['code_sha256']==sha(root/'train.py') and ctx['protocol_sha256']==sha(root/'protocol.json');assert rec['epochs']==epochs
    nf=1 if c in ['speed_mlp','motion_speed'] else 213;F=data['F'][:,:nf];raw=F[tr];med=np.array([np.median(a[np.isfinite(a)]) if np.isfinite(a).any() else 0. for a in raw.T]);filled=np.where(np.isfinite(raw),raw,med);mu=filled.mean(0);sd=filled.std(0);sd[sd<1e-8]=1.;ym=float(data['y'][tr].mean());ys=max(float(data['y'][tr].std()),1e-8);p=ctx['preprocessing'];assert p['condition']==c and p['nfeatures']==nf and np.array_equal(med,p['median']);assert np.allclose(mu,p['mean'],atol=1e-12,rtol=1e-12) and np.allclose(sd,p['std'],atol=1e-12,rtol=1e-12);assert np.isclose(ym,p['target_mean'],atol=1e-12,rtol=1e-12) and np.isclose(ys,p['target_std'],atol=1e-12,rtol=1e-12)
    x=data['X'][ev].astype(np.float32).astype(float);a=((np.where(np.isfinite(F[ev]),F[ev],med)-mu)/sd).astype(np.float32).astype(float);curve=read(root/rec['curve']);assert [r['epoch'] for r in curve]==grid;delta=0.
    for row in curve:
        pt=root/row['checkpoint'];assert sha(pt)==row['checkpoint_sha256'];net=module.Model(c,nf).double();net.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True));net.eval();values=[]
        with torch.no_grad():
            for lo in range(0,len(ev),64):values.extend(net(torch.from_numpy(x[lo:lo+64]),torch.from_numpy(a[lo:lo+64])).numpy()*ys+ym)
        pr=np.array(values);saved=np.array(row['prediction']);assert np.isfinite(pr).all() and len(pr)==len(ev) and np.allclose(pr,saved,atol=1e-6,rtol=1e-6),('CPU64 strict replay',c,float(np.max(np.abs(pr-saved))));delta=max(delta,float(np.max(np.abs(pr-saved))));m=metric(data['y'][ev],saved)
        for k,v in m.items():assert np.isclose(v,row['metrics'][k],atol=1e-12,rtol=1e-12)
        assert np.isclose(max(m['mae']/.9,m['p95']/3,m['slow_mae']/3),row['metrics']['score'],atol=1e-12,rtol=1e-12)
    assert curve[-1]['prediction']==rec['prediction'] and curve[-1]['checkpoint']==rec['checkpoint'] and curve[-1]['checkpoint_sha256']==rec['checkpoint_sha256'];return curve,delta
def remote_file(relative,dest):
    expected=remote_py('import hashlib\nfrom pathlib import Path\nprint(hashlib.sha256(Path('+repr(REMOTE+'/'+relative)+').read_bytes()).hexdigest())');scp('RESEARCH_USER@PRIVATE_HOST:'+REMOTE+'/'+relative,dest);assert sha(dest)==expected;return expected
def main():
    mode=sys.argv[1];assert mode in ['pilot','main'];torch.set_num_threads(4);proto=read(S/'protocol.json');data=np.load(B/'data.npz');assert sha(B/'data.npz')==proto['data_sha256'] and len(data['y'])==594 and sum(data['y']>=20)==9;assert sha(H/'bundle/train.py')==proto['training_code_sha256'];assert sha(H/'protocol.json')==proto['source_protocol_sha256'];submission=read(S/'submission.json')
    if mode=='pilot':
        root,digest=recover('runs/pilot-review.tar.gz',B/'received/pilot');module=library(root);summary=read(root/'runs/pilot/summary.json');assert summary['status']=='passed_pending_local_review' and set(summary['conditions'])==set(CONDS);assert summary['protocol_sha256']==sha(S/'protocol.json') and summary['data_sha256']==sha(B/'data.npz') and summary['code_sha256']==sha(root/'train.py');env=summary['environment'];assert env['hostname'] in ['DIS01','DIS02','DIS03'] and env['gpu_uuid'].startswith('GPU-') and 'A6000' in env['gpu'];ss=proto['splits']['0']['inner'][0];tr=np.array(ss['train']);ev=np.array(ss['valid']);checks=[];timings={}
        for ci,c in enumerate(CONDS):
            rec=summary['conditions'][c]['fit'];curve,delta=audit_fit(module,root,rec,data,tr,ev,c,20261104+ci,10,[10]);timings[c]=rec['elapsed_seconds'];checks.append(dict(condition=c,epochs=10,training_people=len(tr),validation_people=len(ev),cpu64_max_delta=delta,checkpoint_sha256=rec['checkpoint_sha256']))
        log_hash=remote_file('logs/frailpol-motion-pilot-'+submission['pilot_job']+'.log',B/'received/pilot/slurm-original.log')
        result=dict(status='passed',job=submission['pilot_job'],people_in_frozen_cohort=594,pilot_checkpoints_replayed=4,archive_server_local_sha256=digest,slurm_log_sha256=log_hash,checks=checks,environment=env,fit_seconds_per10epochs=timings,train_batches_per_pilot_epoch=int(np.ceil(len(tr)/64)),protocol_sha256=sha(S/'protocol.json'),review_code_sha256=sha(Path(__file__)),limit='Four10epochinnertrainingpilots only;no full cohort performance claim. Source trainer/protocol/modelunmodified. Pointtolerance1e-6absolute+relative unchanged.')
        save(S/'pilot-verification.json',result);print(json.dumps(result,indent=2));return
    assert submission['main_submitted'];roots={};artifacts={};gpu=[]
    for worker in submission['main_workers']:
        job=worker['job'];dest=B/'received/main'/job;root,digest=recover('runs/worker-'+job+'-review.tar.gz',dest);module=library(root);artifacts[job]=dict(archive_sha256=digest)
        for relative,name in [('control/worker-'+job+'.json','worker-original.json'),('logs/'+worker['job_name']+'-'+job+'.log','slurm-original.log')]:artifacts[job][name]=remote_file(relative,dest/name)
        info=read(dest/'worker-original.json');assert info['job_id']==job and info['phase']=='assigned_folds_complete' and info['folds']==worker['folds'];gpu.append(info['environment'])
        for f in worker['folds']:assert f not in roots;roots[f]=root
    assert set(roots)==set(range(5));out={};held=[];checks=[];count=0;y=data['y'];ids=data['ids']
    for ci,c in enumerate(CONDS):
        pred=np.full(594,np.nan)
        for f in range(5):
            root=roots[f];module=library(root);split=proto['splits'][str(f)];tr=np.array(split['train']);te=np.array(split['test']);assert np.array_equal(tr,np.flatnonzero(data['fold']!=f)) and np.array_equal(te,np.flatnonzero(data['fold']==f));assert read(root/f'runs/main/fold{f}/complete.json')['conditions']==CONDS;rec=read(root/f'runs/main/fold{f}/{c}/complete.json');assert rec['test_ids']==ids[te].tolist() and rec['condition']==c and rec['fold']==f;inner=[];delta=0.
            for k,ss in enumerate(split['inner']):
                it=np.array(ss['train']);iv=np.array(ss['valid']);assert not set(it)&set(iv) and set(it)|set(iv)==set(tr);curve,dd=audit_fit(module,root,rec['inner'][k],data,it,iv,c,20261104+f*100+ci+k*1000,100,GRID);inner.append(curve);delta=max(delta,dd);count+=4
            scores=[]
            for j,epoch in enumerate(GRID):
                ip=np.full(len(tr),np.nan);seen=[]
                for ss,curve in zip(split['inner'],inner):
                    ix=[int(np.flatnonzero(tr==i)[0]) for i in ss['valid']];ip[ix]=curve[j]['prediction'];seen+=ss['valid']
                assert len(seen)==len(tr) and set(seen)==set(tr) and np.isfinite(ip).all();m=metric(y[tr],ip);score=max(m['mae']/.9,m['p95']/3,m['slow_mae']/3);assert np.isclose(score,rec['inner_scores'][j]['score'],atol=1e-12,rtol=1e-12);scores.append(score)
            epoch=GRID[int(np.argmin(scores))];assert epoch==rec['selected_epoch'];curve,dd=audit_fit(module,root,rec['final'],data,tr,te,c,20261104+f*100+ci+5000,epoch,[epoch]);count+=1;assert curve[-1]['prediction']==rec['prediction'];pred[te]=rec['prediction'];checks.append(dict(condition=c,fold=f,selected_epoch=epoch,saved_checkpoint_replays=13,training_only_normalization_and_inner_epoch_choice_verified=True,cpu64_max_delta=max(delta,dd)));print('verified',c,f,flush=True)
        assert np.isfinite(pred).all();m=metric(y,pred);m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);out[c]=m;held.extend(dict(id=int(i),condition=c,actual=float(t),prediction=float(p)) for i,t,p in zip(ids,y,pred))
    assert count==260;save(B/'received/main/heldout.json',held);save(S/'comparison.json',out);rng=np.random.default_rng(20261104);boot=rng.integers(0,594,(5000,594));slow=np.flatnonzero(y>=20);sb=slow[rng.integers(0,9,(5000,9))];uncertainty={}
    for c in CONDS:
        lookup={r['id']:r['prediction'] for r in held if r['condition']==c};p=np.array([lookup[int(i)] for i in ids]);e=np.abs(p-y);ctrl='speed_mlp' if c=='motion_speed' else 'features_mlp' if c=='motion_features' else c;clook={r['id']:r['prediction'] for r in held if r['condition']==ctrl};diff=e-np.abs(np.array([clook[int(i)] for i in ids])-y);uncertainty[c]=dict(mae_ci95=np.quantile(e[boot].mean(1),[.025,.975]).tolist(),slow_mae_ci95=np.quantile(e[sb].mean(1),[.025,.975]).tolist(),matching_control=ctrl,mae_difference=float(diff.mean()),mae_difference_ci95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist(),bands={name:dict(n=int(m.sum()),mae=float(e[m].mean()),bias=float((p-y)[m].mean()),max_error=float(e[m].max())) for name,m in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',y>=20)]})
    save(S/'uncertainty.json',dict(method='5000pairedpersonbootstrap;9slowseparately. Conditionalreuseddevelopmentcohort,notclinical/externaloradjustedforsearch. GPUbackendrunpreservesrecipebutCUDA/DirectMLrandomnumberstreamsarenotidentical;notpurehardwarecausalcomparison.',conditions=uncertainty));result=dict(status='passed',people=594,slow_people=9,fits=80,inner_checkpoint_replays=240,final_checkpoint_replays=20,checks=checks,recovery_artifacts=artifacts,gpu_environments=gpu,comparison_sha256=sha(S/'comparison.json'),heldout_sha256=sha(B/'received/main/heldout.json'),uncertainty_sha256=sha(S/'uncertainty.json'),protocol_sha256=sha(S/'protocol.json'),review_code_sha256=sha(Path(__file__)));save(S/'verification.json',result);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
