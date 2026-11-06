"""Retrieve four actual CUDA pilot checkpoints and replay calibration/selection on CPU."""
import json
import numpy as np
import torch
import manage
from model import TransferCNN
from bn_verify import verify_bn
from train import partition, predict, mae


def main():
    torch.set_num_threads(2)
    specifications=[('source/select','source')]+[('target-'+a,'target') for a in ('scratch','gdi','health')]
    results=[]
    for folder,name in specifications:
        remote=manage.REMOTE+'/runs/pilot/'+folder
        dest=manage.DATA/'received/pilot'/folder
        dest.mkdir(parents=True,exist_ok=True)
        meta=json.loads(manage.remote_python("import json,pathlib;r=pathlib.Path(%r);print(json.dumps({n:json.loads((r/n).read_text()) for n in ['identity.json','fit_complete.json','curve.json','bn_calibration.json']}))"%remote))
        for n,v in meta.items():manage.save(dest/n,v)
        manage.run(manage.SCP+['RESEARCH_USER@PRIVATE_HOST:'+remote+'/model.pt',str(dest/'model.pt')])
        assert manage.sha(dest/'model.pt')==meta['fit_complete.json']['checkpoint_sha256']
        with np.load(manage.BUNDLE/(name+'.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        if name=='source':tr,va=d['folds']!=0,d['folds']==0
        else:tr,va,_=partition(d,0,0)
        bn_error=verify_bn(dest,d,tr,manage.sha(manage.BUNDLE/(name+'.npz')))
        identity=meta['identity.json']
        assert sorted(set(d['ids'][va]))==identity['validation_people']
        best=min(meta['curve.json'],key=lambda r:(r['validation_MAE'],r['epoch']))
        assert best['epoch']==meta['fit_complete.json']['best_epoch']
        ck=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
        m=TransferCNN(d['y'].shape[1],identity['config']['dropout']);m.load_state_dict(ck['model'])
        pp=predict(m,d,'cpu',ck['mu'].numpy(),ck['sd'].numpy())
        value=mae(d,pp,va,ck['sd'].numpy() if name=='source' else None)
        error=abs(value-best['validation_MAE']);assert error<1e-4
        probe=torch.from_numpy(d['x'][:3]);m.eval()
        with torch.no_grad():
            one=torch.cat([m(probe[i:i+1]) for i in range(3)]);batch=m(probe)
        batch_error=float((one-batch).abs().max());assert batch_error<1e-5
        results.append(dict(fit=folder,max_bn_buffer_error=bn_error,validation_replay_error=error,batch_size_error=batch_error))
    result=dict(passed=True,fits=results,received=str(manage.DATA/'received/pilot'))
    manage.save(manage.HERE/'pilot-cpu-verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
