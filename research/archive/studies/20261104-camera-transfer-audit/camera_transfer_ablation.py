"""Source-family ablation: refit every fusion and gate, remove all predictor routes."""
from pathlib import Path
import sys,json,hashlib,time
OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-improvement-loop');sys.path.insert(0,str(OLD))
from round12 import torch,np,tensors_loss
from scipy.optimize import nnls
H=Path('D:/Projects/D_P/studies/20261104-camera-transfer-audit');B=Path('D:/gait-research-data/D_P/camera-transfer-audit-20261104');SOURCE=Path('D:/gait-research-data/D_P/frailpol-improvement-loop-20261103');GB=Path('D:/gait-research-data/D_P/frailpol-temporal-cnn-20261104')
GROUPS={'full':[],'without_learned_waveforms':[15,16,30,31,32,33],'without_213feature_experts':list(range(13))+[14,16]+list(range(17,25)),'without_native_norm_family':list(range(30,34)),'without_target_transform_family':list(range(17,25))}
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def metrics(y,p):
    e=np.abs(y-p);return dict(mae=float(e.mean()),p95=float(np.quantile(e,.95)),slow_mae=float(e[y>=20].mean()),slow_bias=float((p-y)[y>=20].mean()))
def expert(X,y,T,tail):
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1.;A=(X-mu)/sd;Z=(T-mu)/sd;w=np.where(y>=20,tail,np.where(y>=12,np.sqrt(tail),1.));am=np.average(A,axis=0,weights=w);ym=np.average(y,weights=w);coef,_=nnls(np.vstack([(A-am)*np.sqrt(w[:,None]),np.sqrt(10)*np.eye(A.shape[1])]),np.r_[(y-ym)*np.sqrt(w),np.zeros(A.shape[1])],maxiter=10000);intercept=ym-am@coef;return dict(mean=mu,std=sd,coef=coef,intercept=intercept,weight=w),A@coef+intercept,Z@coef+intercept
def phi(X,keep,speed,duration):
    # Removed experts cannot survive in direct gate inputs, dispersion or fallback.
    return np.column_stack([X[:,31] if keep[31] else np.zeros(len(X)),X[:,15] if keep[15] else np.zeros(len(X)),np.log1p(X[:,keep].std(1)),np.log(speed),duration])
def main():
    H.mkdir(exist_ok=True);B.mkdir(exist_ok=True);torch.set_num_threads(2);d=np.load(GB/'data.npz');native=np.load(GB/'native-motion/data.npz');assert len(d['y'])==594 and sum(d['y']>=20)==9;proto=dict(created_unix=time.time(),code_sha256=sha(Path(__file__)),source_verifications={str(OLD/f'round{r:02d}/verification.json'):sha(OLD/f'round{r:02d}/verification.json') for r in [11,12]},data_sha256=sha(GB/'data.npz'),native_data_sha256=sha(GB/'native-motion/data.npz'),source_receipt_sha256=sha(OLD/'round11/run-receipt.json'),gate_library_sha256=sha(OLD/'round12.py'),groups=GROUPS,design='Same594/outer5. Predeclaredwhole-familyremovalfromall34expert-predictionroutes. Setremovedtrain/testcolumnszero;refitNNLSexpertsandgateonoutertrainonly. Allgroupsuseoriginalgateconfig1(tail1,aux.1,L2.01)andexperttails1/16,alpha10. Gatefirstnorm/secondoldmotionpredictionzeroifremoved;dispersiononlyretainedcolumns. Fallbacknorm_tail4ifallowed,elsemanualspeed_duration_tail4. Samearchitecture/initialization/loss/optimizer,noouterbasedselection. Reportplainandguarded. Fullarmmustreproduceoriginal0.8805399withinunchanged1e-6pointtolerance.',group_semantics={'without_learned_waveforms':'Remove6CNNpredictionexpertsincludingmixedwave+213inputmodel;otherhandcraftedIMUstatisticsremain. Thisislearned-waveformbranchremoval,notallaccelerationinformationremoval.','without_213feature_experts':'RemoveeveryexpertwhoseinputcontainsF213:13R5,8targettransforms,featuresMLP,motion_features. Keepwaveform+speed/durationfamilies.','without_native_norm_family':'Remove4physical-time/magnitudeCNNexpertsandtheirnormgate/fallbackroutes;oldphaseCNNremains.','without_target_transform_family':'Remove8same-inputtargettransformexperts;thistestslearningstrategy,notnewphysicalinformation.'},limits='Wholeprediction-familydependence,notindividualanatomicalfeaturecausality. Familiesoverlap;parameter/countchangesarepartofablation. StoredinnerOOFbasecontextsfromverifiedoriginalstudiesreusedwithcurrenthashes;no rawbaseCNNretraining. Masks/selectedclipdurationandextractoravailabilitycanremaininallowedinputs;notvalidatedcamerameasurements. Reuseddevelopmentfolds,singleinitialization,9slow. No claimphoneMAE. Allmodelsoriginalpreserved.')
    if (H/'ablation-protocol.json').exists():
        p=read(H/'ablation-protocol.json');proto['created_unix']=p['created_unix'];assert proto==p
    else:save(H/'ablation-protocol.json',proto)
    receipt=read(OLD/'round11/run-receipt.json')
    for group,removed in GROUPS.items():
        for f in range(5):
            dest=B/f'{group}-{f}.json'
            if dest.exists():continue
            record=SOURCE/'round11'/f'combined-{f}.json';assert sha(record)==receipt['hashes'][str(record)];r=read(record);ctx=np.load(r['context']);assert sha(r['context'])==r['context_sha256'];tr=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f);assert np.array_equal(ctx['train_ids'],d['ids'][tr]) and np.array_equal(ctx['test_ids'],d['ids'][te]) and np.array_equal(ctx['y'],d['y'][tr]);X=ctx['X'].copy();T=ctx['T'].copy();keep=np.ones(34,dtype=bool);keep[removed]=False;X[:,~keep]=0;T[:,~keep]=0;low,lt,lv=expert(X,d['y'][tr],T,1);high,ht,hv=expert(X,d['y'][tr],T,16);experts=[]
            for tail,m in [(1,low),(16,high)]:
                path=B/f'{group}-{f}-expert{tail}.npz';np.savez_compressed(path,**m);experts.append(dict(tail=tail,path=str(path),sha256=sha(path)))
            raw=phi(X,keep,ctx['train_speed'],native['F'][tr,1]);test=phi(T,keep,ctx['test_speed'],native['F'][te,1]);mu=raw.mean(0);sd=raw.std(0);sd[sd<1e-8]=1.;P=(raw-mu)/sd;Q=(test-mu)/sd;w=torch.nn.Parameter(torch.zeros(5,dtype=torch.float64));bias=torch.nn.Parameter(torch.tensor(-2.,dtype=torch.float64));opt=torch.optim.LBFGS([w,bias],lr=1,max_iter=200,max_eval=300,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe');tp,tlo,thi,ty=[torch.from_numpy(a) for a in [P,lt,ht,d['y'][tr]]]
            def closure():
                opt.zero_grad();loss=tensors_loss(w,bias,tp,tlo,thi,ty,1,.1,.01);assert torch.isfinite(loss);loss.backward();return loss
            initial=float(closure().detach());opt.step(closure);final=float(closure().detach());assert final<=initial+1e-10
            with torch.no_grad():pred=(torch.from_numpy(lv)+torch.sigmoid(torch.from_numpy(Q)@w+bias)*(torch.from_numpy(hv)-torch.from_numpy(lv))).numpy()
            fallback_index=31 if keep[31] else r['feature_names'].index('manual/speed_duration_tail4');assert keep[fallback_index];fallback=T[:,fallback_index];guarded=np.where(ctx['guard_flags'],fallback,pred);path=B/f'{group}-{f}-gate.npz';np.savez_compressed(path,mean=mu,std=sd,weight=w.detach().numpy(),bias=bias.detach().numpy(),gradient=np.r_[w.grad.numpy(),float(bias.grad)]);context=B/f'{group}-{f}-context.npz';np.savez_compressed(context,X=X,T=T,raw_phi=raw,test_phi=test,y=d['y'][tr],train_ids=d['ids'][tr],test_ids=d['ids'][te],keep=keep,low_train=lt,high_train=ht,low_test=lv,high_test=hv,fallback=fallback,guard_flags=ctx['guard_flags']);save(dest,dict(group=group,fold=f,experts=experts,gate=str(path),gate_sha256=sha(path),context=str(context),context_sha256=sha(context),source_record=str(record),source_record_sha256=sha(record),source_context=r['context'],source_context_sha256=sha(r['context']),plain=pred.tolist(),guarded=guarded.tolist(),fallback_index=fallback_index,initial_objective=initial,final_objective=final,protocol_sha256=sha(H/'ablation-protocol.json')));print('ablation fitted',group,f,flush=True)
    save(H/'ablation-run-receipt.json',dict(status='complete_pending_review',cases=25,expert_models=50,gates=25,protocol_sha256=sha(H/'ablation-protocol.json'),hashes={str(p):sha(p) for p in B.glob('*') if p.is_file()}))
if __name__=='__main__':main()
