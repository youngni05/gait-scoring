"""Retrieve and independently replay source validation before target transfer."""
from pathlib import Path
import json
import numpy as np
import torch
import manage
from model import TransferCNN


def main():
    torch.set_num_threads(2)
    jobs=manage.submissions()['stages']['source']['jobs']
    assert manage.completed(jobs)
    remote=manage.REMOTE
    text=manage.remote_python("import json,pathlib; r=pathlib.Path(%r); names=['summary.json']+[p+'/'+n for p in ['select','refit'] for n in ['identity.json','fit_complete.json','curve.json']]; print(json.dumps({n:json.loads((r/'runs/source'/n).read_text()) for n in names}))"%remote)
    meta=json.loads(text)
    dest=manage.DATA/'received/source';dest.mkdir(parents=True,exist_ok=True)
    manage.save(dest/'metadata.json',meta)
    for name in ('select','refit'):
        path=dest/(name+'.pt')
        manage.run(manage.SCP+['RESEARCH_USER@PRIVATE_HOST:'+remote+'/runs/source/'+name+'/model.pt',str(path)])
        assert manage.sha(path)==meta[name+'/fit_complete.json']['checkpoint_sha256']
    with np.load(manage.BUNDLE/'source.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    train=d['folds']!=0;valid=~train
    expected=[sorted(set(d['ids'][m])) for m in (train,valid)]
    identity=meta['select/identity.json'];assert identity['train_people']==expected[0] and identity['validation_people']==expected[1]
    assert identity['input_sha256']==manage.sha(manage.BUNDLE/'source.npz')
    assert meta['refit/identity.json']['input_sha256']==identity['input_sha256']
    assert not(set(expected[0])&set(expected[1])) and not identity['test_people']
    curve=meta['select/curve.json'];best=min(curve,key=lambda v:(v['validation_MAE'],v['epoch']))
    assert best['epoch']==meta['select/fit_complete.json']['best_epoch']
    assert meta['refit/fit_complete.json']['epochs_run']==best['epoch']
    assert meta['refit/identity.json']['train_people']==sorted(set(d['ids'])) and not meta['refit/identity.json']['validation_people']
    for name,mask in (('select',train),('refit',np.ones(len(train),bool))):
        ck=torch.load(dest/(name+'.pt'),map_location='cpu',weights_only=True)
        people=sorted(set(d['ids'][mask]));means=[d['y'][(d['ids']==p)&mask].astype(np.float64).mean(0) for p in people]
        mu=np.mean(means,0)
        sd=np.sqrt(np.mean([((d['y'][(d['ids']==p)&mask].astype(np.float64)-mu)**2).mean(0) for p in people],0))
        np.testing.assert_allclose(ck['mu'].numpy(),mu,atol=1e-5,rtol=0)
        np.testing.assert_allclose(ck['sd'].numpy(),sd,atol=1e-5,rtol=0)
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
    ck=torch.load(dest/'select.pt',map_location='cpu',weights_only=True)
    model=TransferCNN(4);model.load_state_dict(ck['model']);model.eval()
    wi=np.flatnonzero(valid[d['owner']]);pred=[]
    with torch.no_grad():
        for a in range(0,len(wi),128):
            pred.append(model(torch.from_numpy(d['x'][wi[a:a+128]])).numpy()*ck['sd'].numpy()+ck['mu'].numpy())
    pp=np.concatenate(pred);owner=d['owner'][wi];record_pred={r:pp[owner==r].astype(np.float64).mean(0) for r in np.flatnonzero(valid)}
    errors=[]
    for p in expected[1]:
        rs=np.flatnonzero((d['ids']==p)&valid)
        errors.append(np.mean([np.abs((record_pred[r]-d['y'][r])/ck['sd'].numpy()).mean() for r in rs]))
    score=float(np.mean(errors));difference=abs(score-best['validation_MAE']);assert difference<1e-4
    result=dict(passed=True,jobs=jobs,source_people=len(set(d['ids'])),selection_train_people=len(expected[0]),selection_validation_people=len(expected[1]),selected_epochs=best['epoch'],source_metric='person-averaged standardized four-parameter MAE, not SPPB',CPU_validation_MAE=score,recorded_validation_MAE=best['validation_MAE'],CPU_replay_difference=difference,all_source_refit_people=len(set(d['ids'])),refit_checkpoint_sha256=manage.sha(dest/'refit.pt'),received=str(dest))
    manage.save(manage.HERE/'source-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
