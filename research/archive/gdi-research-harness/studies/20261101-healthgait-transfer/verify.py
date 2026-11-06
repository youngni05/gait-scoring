"""Independently verify returned splits, selected fits, predictions and every model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from model import TransferCNN


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))


def verify(root,round_no,out):
    torch.set_num_threads(2)
    for n,h in load(root/'manifest.json').items():assert sha(root/n)==h,n
    with np.load(root/'target.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    people=sorted(set(d['ids']));assert len(people)==44
    protocol=load(root/'protocol.json');seeds=protocol['target']['seeds']
    rows=[];models=0;fits=0;replay_error=0.;epoch_rows=[]
    for fold in range(5):
        fr=root/('runs/round%d/fold%d'%(round_no,fold));complete=load(fr/'complete.json')
        test=d['folds']==fold;train=~test
        expected_test=sorted(set(d['ids'][test]));assert complete['people']==expected_test
        foldrows=list(csv.DictReader((fr/'person-predictions.csv').open(encoding='utf8')))
        assert len(foldrows)==len(expected_test)*9
        rows.extend(foldrows)
        for arm in ('scratch','gdi','health'):
            for seed in seeds:
                ar=fr/('%s-s%d'%(arm,seed));selection=load(ar/'selection.json');cs=selection['candidates']
                for item in cs:
                    inner=[];ns=[]
                    for j in range(3):
                        p=ar/('candidate%d/inner%d'%(item['candidate'],j))
                        identity=load(p/'identity.json');r=load(p/'fit_complete.json');curve=load(p/'curve.json')
                        tr=set(identity['train_people']);va=set(identity['validation_people']);te=set(identity['test_people'])
                        assert te==set(expected_test) and not(tr&va or tr&te or va&te)
                        assert tr|va==set(d['ids'][train])
                        assert identity['input_sha256']==sha(root/'target.npz')
                        assert r['checkpoint_sha256']==sha(p/'model.pt')
                        earliest=min(curve,key=lambda v:(v['validation_MAE'],v['epoch']))
                        assert earliest['epoch']==r['best_epoch']
                        assert abs(earliest['validation_MAE']-r['best_validation_MAE'])<1e-10
                        # Independently reproduce the selected inner checkpoint's
                        # person-aggregated validation metric and training-only scale.
                        ck_inner=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
                        yy_inner=np.stack([d['y'][d['ids']==pid].mean(0) for pid in sorted(tr)])
                        np.testing.assert_allclose(ck_inner['mu'].numpy(),yy_inner.mean(0),atol=1e-6)
                        np.testing.assert_allclose(ck_inner['sd'].numpy(),np.maximum(yy_inner.std(0),1e-4),atol=1e-6)
                        inner_model=TransferCNN(1,item['config']['dropout']);inner_model.load_state_dict(ck_inner['model']);inner_model.eval()
                        iw=np.flatnonzero(np.isin(d['ids'][d['owner']],list(va)));inner_pred=[]
                        with torch.no_grad():
                            for st in range(0,len(iw),128):
                                inner_pred.append(inner_model(torch.from_numpy(d['x'][iw[st:st+128]])).numpy()*ck_inner['sd'].numpy()+ck_inner['mu'].numpy())
                        ip=np.concatenate(inner_pred)[:,0];iv=[]
                        for pid in sorted(va):
                            records=np.flatnonzero(d['ids']==pid)
                            prediction=np.mean([ip[d['owner'][iw]==idx].mean() for idx in records])
                            iv.append(abs(prediction-d['y'][d['ids']==pid].mean()))
                        assert abs(np.mean(iv)-r['best_validation_MAE'])<1e-4
                        inner.append(r);ns.append(len(va));fits+=1
                    score=np.average([r['best_validation_MAE'] for r in inner],weights=ns)
                    assert abs(score-item['score'])<1e-10
                    assert int(np.median([r['best_epoch'] for r in inner]))==item['epochs']
                best=min(cs,key=lambda c:(c['score'],c['candidate']));assert best==selection['selected']
                p=ar/'refit';identity=load(p/'identity.json');r=load(p/'fit_complete.json')
                assert set(identity['train_people'])==set(d['ids'][train]) and not identity['validation_people']
                assert identity['test_people']==expected_test and r['best_epoch']==best['epochs']
                assert identity['config']==best['config'] and r['checkpoint_sha256']==sha(p/'model.pt')
                ck=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
                train_ids=sorted(set(d['ids'][train]));yy=np.stack([d['y'][(d['ids']==v)&train].mean(0) for v in train_ids])
                mu=yy.mean(0);sd=np.maximum(yy.std(0),1e-4)
                np.testing.assert_allclose(ck['mu'].numpy(),mu,atol=1e-6)
                np.testing.assert_allclose(ck['sd'].numpy(),sd,atol=1e-6)
                m=TransferCNN(1,best['config']['dropout']);m.load_state_dict(ck['model']);m.eval()
                keep=test[d['owner']];xx=d['x'][keep];pred=[]
                with torch.no_grad():
                    for a in range(0,len(xx),128):pred.append(m(torch.from_numpy(xx[a:a+128])).numpy()*ck['sd'].numpy()+ck['mu'].numpy())
                wp=np.concatenate(pred)
                with np.load(ar/'predictions.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['window_indices'],np.flatnonzero(keep))
                    np.testing.assert_allclose(wp,z['window_predictions'],atol=1e-4,rtol=0)
                    replay_error=max(replay_error,float(np.max(np.abs(wp-z['window_predictions']))))
                owners=d['owner'][keep]
                rp={i:wp[owners==i].mean() for i in np.flatnonzero(test)}
                for pid in expected_test:
                    truth=float(d['y'][(d['ids']==pid)&test].mean())
                    pp=float(np.mean([rp[i] for i in np.flatnonzero((d['ids']==pid)&test)]))
                    row=[v for v in foldrows if v['arm']==arm and int(v['seed'])==seed and v['id']==pid]
                    assert len(row)==1 and float(row[0]['y'])==truth
                    assert abs(float(row[0]['prediction'])-pp)<1e-4
                epoch_rows.append(dict(fold=fold,arm=arm,seed=seed,epochs=best['epochs'],config=best['config']))
                models+=1;fits+=1
    summary={};arrays={}
    truth=np.array([float(d['y'][d['ids']==p].mean()) for p in people])
    for arm in ('scratch','gdi','health'):
        per_seed={};allpred=[]
        for seed in seeds:
            pred=np.array([float(next(r['prediction'] for r in rows if r['id']==p and r['arm']==arm and int(r['seed'])==seed)) for p in people])
            allpred.append(pred);per_seed[str(seed)]=dict(MAE=float(np.abs(pred-truth).mean()),RMSE=float(np.sqrt(np.mean((pred-truth)**2))))
        pred=np.mean(allpred,axis=0);arrays[arm]=pred;err=pred-truth
        summary[arm]=dict(seeds=per_seed,ensemble_MAE=float(np.abs(err).mean()),ensemble_RMSE=float(np.sqrt(np.mean(err**2))),bias=float(err.mean()),prediction_range=[float(pred.min()),float(pred.max())],low_score_5_8_MAE=float(np.abs(err[truth<=8]).mean()),seed_prediction_std_mean=float(np.std(allpred,axis=0).mean()))
    baseline={}
    for name in ('mean','median'):
        pred=np.empty(len(people))
        for fold in range(5):
            train_people=sorted(set(d['ids'][d['folds']!=fold]));yy=np.array([d['y'][d['ids']==p].mean() for p in train_people])
            value=np.mean(yy) if name=='mean' else np.median(yy)
            for i,p in enumerate(people):
                if d['folds'][np.flatnonzero(d['ids']==p)[0]]==fold:pred[i]=value
        baseline[name]=dict(MAE=float(np.abs(pred-truth).mean()),RMSE=float(np.sqrt(np.mean((pred-truth)**2))))
    rng=np.random.RandomState(20261101);boot=rng.randint(0,len(people),(10000,len(people)));differences={}
    for arm in ('gdi','health'):
        diff=np.abs(arrays[arm]-truth)-np.abs(arrays['scratch']-truth)
        differences[arm]=dict(MAE_minus_scratch=float(diff.mean()),conditional_person_bootstrap95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist())
    out.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,round=round_no,people=44,records=220,final_models=models,all_fits=fits,max_cpu_replay_error=replay_error,metrics=summary,baselines=baseline,paired_differences=differences,scope='Exploratory development CV; bootstrap conditional on fitted predictions; no independent clinical/video test')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (out/'selected-configs.json').write_text(json.dumps(epoch_rows,indent=2)+'\n',encoding='utf8')
    with (out/'person-predictions.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.writer(f);w.writerow(['id','SPPB','scratch','gdi','health'])
        for i,p in enumerate(people):w.writerow([p,truth[i],*[arrays[k][i] for k in ('scratch','gdi','health')]])
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--round',type=int,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();verify(a.root,a.round,a.out)
