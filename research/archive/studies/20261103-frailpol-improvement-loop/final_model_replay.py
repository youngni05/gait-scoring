"""Fresh saved-input -> every final base model -> fusion -> gate replay for all594."""
from pathlib import Path
import sys,json,hashlib,time,subprocess
H=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');B=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');G=Path('D:/Projects/D_P/studies/20261104-frailpol-temporal-cnn');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104');R=H/'final-model-replay';O=B/'final-model-replay';sys.path.insert(0,str(H));from round3 import predict2,np,joblib;from review2 import check_context_model
# The original library sets BLAS/OpenMP limits before importing NumPy/PyTorch.
# Preserve that import order, not just the later torch intra-op thread count.
import torch
from scipy.special import expit
sys.path.insert(0,str(G/'bundle'));from train import Model as OldModel
sys.path.insert(0,str(G));from norm_aux import Model as WaveModel;from manual_controls import Model as ManualModel
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def assert_points(p,q):
    assert np.isfinite(p).all() and np.allclose(p,q,atol=1e-6,rtol=1e-6), ('point replay failed',float(np.max(np.abs(p-q))));return float(np.max(np.abs(p-q)))
def scalar_metric(y,p):
    e=np.abs(y-p);m=dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()));m['goal_met']=bool(m['mae']<=.9 and m['p95']<=3 and m['slow_mae']<=3);return m
def main():
    R.mkdir(exist_ok=True);O.mkdir(exist_ok=True);torch.set_num_threads(2);d=np.load(GB/'data.npz');native=np.load(GB/'native-motion/data.npz');ids=d['ids'];y=d['y'];fold=d['fold'];assert len(ids)==594 and sum(y>=20)==9;bridge=read(H/'current-label-bridge.json');assert bridge['status']=='passed' and bridge['current_data_sha256']==sha(GB/'data.npz') and bridge['native_data_sha256']==sha(GB/'native-motion/data.npz');assert all(read(H/s/'verification.json')['status']=='passed' for s in ['round05','round07','round08','round11','round12']);assert read(G/'local-directml/verification.json')['status']=='passed' and read(G/'manual-controls/verification.json')['status']=='passed';wavever=read(G/'norm-aux/verification.json');assert set(['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16'])<=set(wavever['passed_conditions']);assert 'symnorm_aux_tail4' not in wavever['passed_conditions'];raw24=native['native'][:,:24];mask=native['native'][:,-1:]>0;norm=raw24[:,[3,7,11,15,19,23]];sym=np.stack([(norm[:,0]+norm[:,2])/2,(norm[:,1]+norm[:,3])/2,np.abs(norm[:,0]-norm[:,2]),np.abs(norm[:,1]-norm[:,3]),norm[:,4],norm[:,5]],1);wave_raw=dict(full=raw24,norm=norm,symnorm=sym);proof={};checks=[];preds={};originals={};base_count=0;gate_count=0;start=time.perf_counter()
    def track(path,expected=None):
        value=sha(path);assert expected is None or value==expected;proof[str(path)]=value
    def base_prediction(model,path,X,wrapped,tag):
        base=model['base'] if wrapped else model
        if base['cfg']['kind']!='TabPFNv2':return predict2(base,X)
        # Keep the original saved model/context and point tolerance. Isolate the
        # numerical runtime from other neural libraries loaded in this process.
        request=O/(tag+'-input.npy');output=O/(tag+'-output.npy');worker=H/'foundation_replay_worker.py'
        np.save(request,X);track(worker)
        run=subprocess.run([sys.executable,str(worker),str(path),str(request),str(output),'1' if wrapped else '0'],capture_output=True,text=True)
        assert run.returncode==0,(run.stdout,run.stderr)
        save(O/(tag+'-worker-receipt.json'),dict(source_model=str(path),source_sha256=sha(path),input_sha256=sha(request),output_sha256=sha(output),worker_sha256=sha(worker),stdout=run.stdout,stderr=run.stderr))
        return np.load(output)
    for f in range(5):
        tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f);allbase={};maxdelta=0
        torch.set_num_threads(4) # Original fixed foundation-model factory execution setting.
        for family in ['speed','bilateral']:
            X=d['F'][:,:1] if family=='speed' else d['F'];r5=read(B/'round05'/f'{family}-{f}.json');r8=read(B/'round08'/f'{family}-{f}.json');ref5=np.load(r5['meta_context'])['test_matrix'];ref8=np.load(r8['context'])['test_matrix'];p5=[];p8=[]
            for r in sorted(r5['models'],key=lambda a:a['candidate']):
                print('replay source',f,family,'round05',r['candidate'],flush=True)
                track(r['path'],r['sha256']);m=joblib.load(r['path']);check_context_model(m,X[tr],y[tr],m['cfg']);p=base_prediction(m,r['path'],X[te],False,f'round05-{family}-{f}-{r["candidate"]}');maxdelta=max(maxdelta,assert_points(p,ref5[:,r['candidate']]));p5.append(p);base_count+=1
            for r in sorted(r8['models'],key=lambda a:a['candidate']):
                print('replay source',f,family,'round08',r['candidate'],flush=True)
                track(r['path'],r['sha256']);m=joblib.load(r['path']);cfg=m['base']['cfg'];speed=X[tr,0]
                if m['transform']=='speed_product':target=y[tr]*speed;assert m['coefficient'] is None
                else:
                    A=np.column_stack([np.ones(len(tr)),1/speed]);coef=np.linalg.lstsq(A,y[tr],rcond=None)[0];assert np.array_equal(coef,m['coefficient']);target=y[tr]-A@coef
                check_context_model(m['base'],X[tr],target,cfg);p=base_prediction(m,r['path'],X[te],True,f'round08-{family}-{f}-{r["candidate"]}');p=p/X[te,0] if m['transform']=='speed_product' else p+np.column_stack([np.ones(len(te)),1/X[te,0]])@m['coefficient'];maxdelta=max(maxdelta,assert_points(p,ref8[:,r['candidate']]));p8.append(p);base_count+=1
            allbase['raw_'+family]=np.column_stack(p5);allbase['target_'+family]=np.column_stack(p8)
        torch.set_num_threads(2) # Original CPU64 neural-model review setting.
        old={}
        for c in ['speed_mlp','features_mlp','motion_speed','motion_features']:
            rec=read(GB/'local-main'/c/f'fold{f}/complete.json');r=rec['final'];ctx=read(r['context']);assert ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[te].tolist();p=ctx['preprocessing'];F=d['F'][:,:1] if p['nfeatures']==1 else d['F'];a=(np.where(np.isfinite(F[te]),F[te],p['median'])-p['mean'])/p['std'];a=a.astype(np.float32).astype(float);x=d['X'][te].astype(np.float32).astype(float);assert p['target_mean']==y[tr].mean() and np.isclose(p['target_std'],y[tr].std(),atol=1e-12,rtol=1e-12);track(r['checkpoint'],r['checkpoint_sha256']);net=OldModel(c,p['nfeatures']).double();net.load_state_dict(torch.load(r['checkpoint'],map_location='cpu',weights_only=True));net.eval();values=[]
            with torch.no_grad():
                for lo in range(0,len(te),64):values.extend(net(torch.from_numpy(x[lo:lo+64]),torch.from_numpy(a[lo:lo+64])).numpy()*p['target_std']+p['target_mean'])
            old[c]=np.array(values);maxdelta=max(maxdelta,assert_points(old[c],rec['prediction']));base_count+=1
        manual={};wave={}
        for c in ['speed_tail4','speed_tail16','speed_aux_tail4','speed_duration_tail4','speed_duration_aux_tail4']:
            rec=read(GB/'manual-controls'/c/f'fold{f}/fold.json');r=rec['final'];ctx=read(r['context']);assert ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[te].tolist();p=ctx['prep'];raw=native['F'][te].copy()
            if 'duration' not in c:raw[:,1]=0
            a=((raw-p['mean'])/p['std']).astype(np.float32).astype(float);assert p['ym']==y[tr].mean() and np.isclose(p['ys'],y[tr].std(),atol=1e-12,rtol=1e-12);row=r['curve'][0];track(row['model'],row['sha256']);net=ManualModel('_aux_' in c).double();net.load_state_dict(torch.load(row['model'],map_location='cpu',weights_only=True));net.eval();values=[]
            with torch.no_grad():
                for lo in range(0,len(te),64):values.extend(net(torch.from_numpy(a[lo:lo+64])).numpy()*p['ys']+p['ym'])
            manual[c]=np.array(values);maxdelta=max(maxdelta,assert_points(manual[c],rec['prediction']));base_count+=1
        for c in ['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16']:
            rec=read(GB/'norm-aux'/c/f'fold{f}/fold.json');r=rec['final'];ctx=read(r['context']);assert ctx['train_ids']==ids[tr].tolist() and ctx['eval_ids']==ids[te].tolist();p=ctx['prep'];raw=wave_raw[p['kind']];xval=np.where(mask[te],(raw[te]-np.array(p['channel_mean'])[None,:,None])/np.array(p['channel_std'])[None,:,None],0);x=np.concatenate([xval,mask[te].astype(float)],1).astype(np.float32).astype(float);a=((native['F'][te]-p['manual_mean'])/p['manual_std']).astype(np.float32).astype(float);assert p['ym']==y[tr].mean() and np.isclose(p['ys'],y[tr].std(),atol=1e-12,rtol=1e-12);row=r['curve'][0];track(row['model'],row['sha256']);net=WaveModel(raw.shape[1],False).double();net.load_state_dict(torch.load(row['model'],map_location='cpu',weights_only=True));net.eval();values=[]
            with torch.no_grad():
                for lo in range(0,len(te),64):values.extend(net(torch.from_numpy(x[lo:lo+64]),torch.from_numpy(a[lo:lo+64])).numpy()*p['ys']+p['ym'])
            wave[c]=np.array(values);maxdelta=max(maxdelta,assert_points(wave[c],rec['prediction']));base_count+=1
        speed=np.column_stack([allbase['raw_speed'],old['speed_mlp'],allbase['target_speed'],manual['speed_tail4'],manual['speed_tail16'],manual['speed_aux_tail4']]);duration=np.column_stack([speed,manual['speed_duration_tail4'],manual['speed_duration_aux_tail4']]);combined=np.column_stack([allbase['raw_bilateral']]+[old[c] for c in ['speed_mlp','features_mlp','motion_speed','motion_features']]+[allbase['target_bilateral']]+[manual[c] for c in ['speed_tail4','speed_tail16','speed_aux_tail4','speed_duration_tail4','speed_duration_aux_tail4']]+[wave[c] for c in ['full_tail4','norm_tail4','symnorm_tail4','symnorm_tail16']]);matrices=dict(speed=speed,speed_duration=duration,combined=combined);fold_pipeline_delta=0
        for c,Z in matrices.items():
            r11=read(B/'round11'/f'{c}-{f}.json');ctx11=np.load(r11['context']);assert Z.shape==ctx11['T'].shape and np.array_equal(ctx11['test_ids'],ids[te]);assert_points(Z,ctx11['T']);expert=[]
            for tail in [1,16]:
                row=next(r for r in r11['models'] if r['tail']==tail);track(row['path'],row['sha256']);m=np.load(row['path']);expert.append(((Z-m['mean'])/m['std'])@m['coef']+m['intercept'])
            names=r11['feature_names'];first='wave/norm_tail4' if c=='combined' else 'manual/speed_duration_tail4' if c=='speed_duration' else 'manual/speed_tail4';second='round07base15' if c=='combined' else 'round07base7';phi=np.column_stack([Z[:,names.index(first)],Z[:,names.index(second)],np.log1p(Z.std(1)),np.log(d['F'][te,0]),native['F'][te,1] if c!='speed' else np.zeros(len(te))]);flags=(d['F'][te,0]<d['F'][tr,0].min())|(d['F'][te,0]>d['F'][tr,0].max());fallback=wave['norm_tail4'] if c=='combined' else manual['speed_duration_tail4'] if c=='speed_duration' else manual['speed_tail4'];r12=read(B/'round12'/f'{c}-{f}.json')
            for row in r12['models']:
                track(row['path'],row['sha256']);m=np.load(row['path']);g=expit(((phi-m['mean'])/m['std'])@m['gate_weight']+m['gate_bias']);pred=expert[0]+g*(expert[1]-expert[0]);guarded=np.where(flags,fallback,pred)
                for policy,fresh,key in [('plain',pred,'prediction'),('guarded',guarded,'guarded_prediction')]:
                    reference=np.array(row[key]);delta=assert_points(fresh,reference);fold_pipeline_delta=max(fold_pipeline_delta,delta);name=c+f'/config{row["index"]}/'+policy;preds.setdefault(name,np.full(594,np.nan))[te]=fresh;originals.setdefault(name,np.full(594,np.nan))[te]=reference
                gate_count+=1
        np.savez_compressed(O/f'fold{f}-fresh-base-predictions.npz',ids=ids[te],**matrices);checks.append(dict(fold=f,base_models_replayed=49,gates_replayed=12,maximum_base_delta_seconds=maxdelta,maximum_final_pipeline_delta_seconds=fold_pipeline_delta));save(R/'progress.json',dict(status='replaying',folds_done=f+1,expected_folds=5,base_models_replayed=base_count,gates_replayed=gate_count,elapsed_seconds=time.perf_counter()-start));print('fresh final-model replay fold',f,'base delta',maxdelta,'pipeline delta',fold_pipeline_delta,flush=True)
    assert base_count==245 and gate_count==60;summary={};held=[];report=read(H/'round12/comparison.json')
    for name,pred in preds.items():
        assert np.isfinite(pred).all();reference=originals[name];stored=scalar_metric(y,reference);fresh=scalar_metric(y,pred);e_delta=np.abs(pred-reference);assert np.isclose(stored['mae'],report[name]['mae'],atol=1e-12,rtol=1e-12) and np.isclose(stored['p95'],report[name]['p95'],atol=1e-12,rtol=1e-12) and np.isclose(stored['slow_mae'],report[name]['slow_mae'],atol=1e-12,rtol=1e-12);assert abs(fresh['mae']-stored['mae'])<=e_delta.mean()+1e-12 and abs(fresh['p95']-stored['p95'])<=e_delta.max()+1e-12 and abs(fresh['slow_mae']-stored['slow_mae'])<=e_delta[y>=20].mean()+1e-12;assert fresh['goal_met']==stored['goal_met'];summary[name]=dict(fresh_metrics=fresh,original_metrics=stored,max_point_delta_seconds=float(e_delta.max()),metrics_difference_bounded_by_actual_point_replay_error=True);held.extend(dict(id=int(i),condition=name,actual=float(a),prediction=float(v)) for i,a,v in zip(ids,y,pred))
    for path,value in proof.items():assert sha(path)==value
    save(O/'fresh-heldout.json',held);save(R/'source-model-hashes.json',proof);save(R/'verification.json',dict(status='passed',people=594,slow_people=9,final_base_models_replayed=245,gate_models_replayed=60,conditions_rebuilt=24,checks=checks,goal_meeting_configurations=[k for k,v in summary.items() if v['fresh_metrics']['goal_met']],conditions=summary,source_hashes_sha256=sha(R/'source-model-hashes.json'),fresh_heldout_sha256=sha(O/'fresh-heldout.json'),code_sha256=sha(Path(__file__)),environment=dict(torch=torch.__version__,numpy=np.__version__,threads=2),limits='Freshreplaybeginsatpreviouslyraw-verifiedsavedinputs;doesnotredoallrawfeatureextractionhere. Originalraw594inputdouble-replays/outside-coretestsandproviderlabelbridgeareseparatehashedreceipts. Originalpointtolerance1e-6absolute+relative unchanged;metricdriftboundisderivedfromactualpointdeltas,notaloosenedmetricthreshold. All24policiesreplayed,bothgoal-meetingconfigsreported,noproductionmodelselectionorclinical/externalvalidation.'));print('fresh entire saved-model pipeline replay passed',base_count,gate_count)
if __name__=='__main__':main()
