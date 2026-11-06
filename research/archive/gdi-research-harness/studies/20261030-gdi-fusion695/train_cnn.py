"""Reuse prepared CNN training with the actual CPU inner folds; write only here."""
from common import HERE, ROOT, MB, data, load, save, sha, split, summary
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import sys
import time
import numpy as np

sys.path.insert(0,str(HERE/'cnn_source'))
spec=importlib.util.spec_from_file_location('prepared_cnn_runner',HERE/'cnn_source/run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
import torch

original_setup=runner.setup
def single_thread_setup(seed,device):
    environment=original_setup(seed,device)
    torch.set_num_threads(1)
    environment['intraop_threads']=1
    return environment
runner.setup=single_thread_setup


def config():
    return {**load(HERE/'cnn_source/protocol.json'),**load(HERE/'protocol.json')['CNN']}


def identity(d,outer,inner,epochs,purpose):
    tr,te=split(d,outer)
    if inner is None: va=np.array([],dtype=int)
    else: tr,va=split(d,outer,inner)
    return {'protocol_sha256':sha(HERE/'protocol.json'),'local_data_sha256':sha(HERE/'data.npz'),
            'outer_fold':outer,'inner_fold':inner,'epochs':epochs,'purpose':purpose,
            'fit_seed':20261028+outer*100+(inner if inner is not None else 50),
            'train_people':np.unique(d['person_id'][tr]).tolist(),
            'validation_people':np.unique(d['person_id'][va]).tolist(),
            'outer_evaluation_people':np.unique(d['person_id'][te]).tolist()}


def frame_features(d):
    path=HERE/'frame_features.npy'
    proof=HERE/'frame_features.json'
    if proof.exists():
        assert sha(path)==load(proof)['sha256']
        return np.load(path,mmap_mode='r',allow_pickle=False)
    assert not path.exists(),'Partial frame cache requires inspection, not silent reuse'
    sys.path.insert(0,str(MB/'vendor'))
    from lib.model.DSTformer import DSTformer
    assert sha(MB/'assets/MB_lite.bin')=='6a6ad0055c7ad50da083af0549a24c52ec1c21f89e440912645054d74be0a461'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    model=DSTformer(dim_in=3,dim_out=3,dim_feat=256,dim_rep=512,depth=5,num_heads=8,
                    mlp_ratio=4,num_joints=17,maxlen=243,qkv_bias=True,att_fuse=True)
    state=torch.load(MB/'assets/MB_lite.bin',map_location='cpu',weights_only=True)['model_pos']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    model.requires_grad_(False);model.eval()
    arr=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(1064,124,512))
    max_delta=0.;tick=time.monotonic()
    with torch.inference_mode():
        for i,x in enumerate(d['x']):
            rep=model.get_representation(torch.from_numpy(x[None]))
            pooled=rep.mean((1,2)).numpy()[0]
            np.testing.assert_allclose(pooled,d['motionbert'][i],rtol=1e-5,atol=2e-6)
            frame=rep.mean(2).numpy()[0]
            assert frame.shape==(124,512) and np.isfinite(frame).all()
            arr[i]=frame;max_delta=max(max_delta,float(abs(pooled-d['motionbert'][i]).max()))
            if (i+1)%100==0:
                arr.flush();print(json.dumps({'frame_features':i+1,'total':1064,'seconds':time.monotonic()-tick}),flush=True)
    arr.flush();del arr
    save('frame_features.json',{'shape':[1064,124,512],'sha256':sha(path),'pooled_max_abs_difference':max_delta,
                                'encoder_trainable_parameters':0,'elapsed_seconds':time.monotonic()-tick})
    return np.load(path,mmap_mode='r',allow_pickle=False)


def use_temporal_model():
    from model import GdiCNN,SameConv
    from torch.nn import functional as F
    class TemporalCNN(GdiCNN):
        def __init__(self,dropout=.5):
            super().__init__(dropout)
            self.convs[0]=SameConv(512,32,8)
            torch.nn.init.xavier_uniform_(self.convs[0].weight);torch.nn.init.zeros_(self.convs[0].bias)
        def forward(self,x):
            if x.ndim!=3 or tuple(x.shape[1:])!=(512,124):raise ValueError('Expected B,512,124')
            for i,(conv,norm) in enumerate(zip(self.convs,self.norms)):
                x=norm(F.relu(conv(x)))
                if i%2:x=self.dropout(F.max_pool1d(x,3 if i==5 else 2))
            x=x.transpose(1,2).reshape(x.shape[0],-1)
            return self.output(F.relu(self.hidden(x)))
    runner.GdiCNN=TemporalCNN
    def channels(x):
        assert x.ndim==3 and tuple(x.shape[1:])==(124,512)
        return x.transpose(1,2).contiguous()
    runner.channels=channels


def fit_task(mode,outer,inner,epochs):
    d=data();c=config();x=d['x']
    if mode=='temporal':
        x=np.load(HERE/'frame_features.npy',mmap_mode='r',allow_pickle=False)
        use_temporal_model()
    tr,te=split(d,outer)
    if inner is None:
        va=None;valid_x=valid_y=valid_people=None
    else:
        tr,va=split(d,outer,inner)
        valid_x,valid_y,valid_people=x[va],d['y'][va],d['person_id'][va]
    purpose=mode+('_outer_refit' if inner is None else '_inner_epoch_selection')
    ident=identity(d,outer,inner,epochs,purpose)
    dest=HERE/'cnn'/mode/f'fold{outer}'/('refit' if inner is None else f'inner{inner}')
    model,lo,span,hist=runner.fit(x[tr],d['y'][tr],d['person_id'][tr],valid_x,valid_y,valid_people,
                               c,epochs,dest,ident,'cpu')
    if inner is None:
        value=runner.predict(model,runner.channels(torch.from_numpy(np.asarray(x[te]))),lo,span,'cpu')
        np.savez_compressed(dest.parent/'predictions.npz',predictions=value,indices=te)
        runner.save(dest.parent/'complete.json',{'identity':ident,'selected_epoch':epochs,
            'checkpoint_sha256':sha(dest/'latest.pt'),'prediction_sha256':sha(dest.parent/'predictions.npz')})
    return {'outer':outer,'inner':inner,'epochs':epochs}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pilot','raw','temporal','extract']);a=parser.parse_args()
    d=data();c=config();tick=time.monotonic()
    if a.mode=='extract':frame_features(d);return
    x=d['x']
    if a.mode=='temporal':x=frame_features(d);use_temporal_model()
    out=HERE/'cnn'/a.mode;out.mkdir(parents=True,exist_ok=True)
    save(out.relative_to(HERE)/'environment.json',runner.setup(c['seed'],'cpu'))
    if a.mode=='pilot':
        tr,va=split(d,0,0)
        ident=identity(d,0,0,3,'resource_pilot_no_outer_evaluation')
        _,_,_,hist=runner.fit(x[tr],d['y'][tr],d['person_id'][tr],x[va],d['y'][va],d['person_id'][va],
                             c,3,out/'fit',ident,'cpu')
        estimate=float(np.mean([h['seconds'] for h in hist]))*100*(20+5*556/417)
        save('cnn/pilot/resource.json',{'mean_epoch_seconds':float(np.mean([h['seconds'] for h in hist])),
                                      'rough_max_total_seconds':estimate,'not_a_performance_result':True})
        print(json.dumps({'pilot_estimated_total_seconds':estimate}),flush=True);return
    # Independent fits use separate processes/RNG/checkpoints; four single-thread
    # workers replace sequential four-thread execution, without changing methods.
    execution=HERE/'cnn_execution.json'
    plan={'workers':4,'threads_per_fit':1,'source_default_threads':4,
          'reason':'Training-only resource pilot projected 34 minutes sequentially; parallel independent fits on an 8-logical-CPU host.',
          'statistical_protocol_changed':False,'parallel_rng_shared':False}
    if execution.exists():assert load(execution)==plan
    else:save('cnn_execution.json',plan)
    pred=np.full((1064,2),np.nan)
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        jobs=[pool.submit(fit_task,a.mode,o,i,c['max_epochs']) for o in range(5) for i in range(4)]
        for job in as_completed(jobs):
            print(json.dumps({'inner_fit_finished':job.result(),'seconds':time.monotonic()-tick}),flush=True)
        jobs=[]
        for outer in range(5):
            dest=out/f'fold{outer}'
            histories=[load(dest/f'inner{i}/curve.json') for i in range(4)]
            epoch,scores=runner.select_epoch(histories)
            runner.save(dest/'selection.json',{'selected_epoch':epoch,'inner_person_MAE_curve':scores,'outer_evaluation_used':False})
            jobs.append(pool.submit(fit_task,a.mode,outer,None,epoch))
        for job in as_completed(jobs):
            print(json.dumps({'outer_fit_finished':job.result(),'seconds':time.monotonic()-tick}),flush=True)
    for outer in range(5):
        dest=out/f'fold{outer}'
        if (dest/'complete.json').exists():
            with np.load(dest/'predictions.npz',allow_pickle=False) as z:pred[z['indices']]=z['predictions']
            continue
        raise AssertionError('Parallel outer fit did not complete')
    assert np.isfinite(pred).all()
    methods={a.mode+'_cnn':pred,'pls':d['pls'],a.mode+'_cnn_pls_equal':(pred+d['pls'])/2}
    if a.mode=='raw':
        with np.load(HERE/'rf_predictions.npz',allow_pickle=False) as z:methods['raw_cnn_rf_coordinate_equal']=(pred+z['rf_coordinate'])/2
    np.savez_compressed(out/'oof_predictions.npz',**methods)
    result={**summary(d,methods),'elapsed_seconds':time.monotonic()-tick,'finished_utc':datetime.now(timezone.utc).isoformat(),
            'independent_final_evaluation':False,'protected_numeric_data_read':False,
            'sha256':{'predictions':sha(out/'oof_predictions.npz'),'training_code':sha(HERE/'train_cnn.py'),'protocol':sha(HERE/'protocol.json')}}
    runner.save(out/'summary.json',result)
    print(json.dumps({'completed':a.mode,'MAE':{k:v['MAE'] for k,v in result['metrics'].items()},'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
