"""Independent label/split/replay/metric audit after artifact retrieval."""
from common import *
import csv
import pandas as pd
import scipy.io
import openpyxl
import joblib
from datetime import datetime, timezone
sys.path.append(str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
import torch
from gpu import Net

def run():
    d=load();x,y,ids,folds=(d[k] for k in ['x','y','ids','folds'])
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
    wb=openpyxl.load_workbook(protocol['sources']['clinical_table'],data_only=True)
    raw_labels={str(r[0]):float(r[9]) for r in list(wb.active.values)[1:] if isinstance(r[0],int)}
    wb.close()
    assert sha(Path(protocol['sources']['clinical_table']))==protocol['sources']['clinical_sha256']
    assert sha(Path(protocol['sources']['input']))==protocol['sources']['input_sha256']
    original=np.load(protocol['sources']['input'],allow_pickle=False)
    np.testing.assert_array_equal(x,original['input'].reshape(59,124,51).transpose(0,2,1))
    np.testing.assert_array_equal(ids,original['person'].astype(str))
    np.testing.assert_allclose(y,[raw_labels[p] for p in ids],rtol=0,atol=1e-6)
    splits=json.loads((HERE/'splits.json').read_text())
    covered=[]
    for s in splits:
        tr=np.array(s['train']);te=np.array(s['test']);covered.extend(te)
        assert set(tr)|set(te)==set(range(len(y)))
        assert not(set(ids[tr])&set(ids[te]))
        assert np.all(folds[te]==s['fold']) and np.all(folds[tr]!=s['fold'])
        vc=[]
        for q in s['inner']:
            a=np.array(q['train']);b=np.array(q['validation']);vc.extend(b)
            assert set(a)|set(b)==set(tr) and not(set(ids[a])&set(ids[b]))
        assert sorted(vc)==sorted(tr)
    assert sorted(covered)==list(range(len(y)))
    cpu=pd.read_csv(HERE/'cpu-predictions.csv',dtype={'id':str})
    np.testing.assert_allclose(cpu.target,y,rtol=0,atol=1e-6)
    f=features(x,d['frames'])
    for s in splits:
        tr=np.array(s['train']);te=np.array(s['test'])
        weighted_mean=np.average(y[tr].astype(float),weights=weights(ids[tr]))
        np.testing.assert_allclose(cpu.Mean[te],weighted_mean,rtol=0,atol=1e-6)
        order=np.argsort(y[tr]);ww=weights(ids[tr])[order]
        med=y[tr][order[np.flatnonzero(np.cumsum(ww)>=ww.sum()/2)[0]]]
        np.testing.assert_allclose(cpu.Median[te],med,rtol=0,atol=1e-6)
        for name in ['Ridge','RF']:
            scaler,model=joblib.load(HERE/'cpu-models'/f'{name}-fold{s["fold"]}.joblib')
            np.testing.assert_allclose(scaler.mean_,np.average(f[tr],axis=0,weights=weights(ids[tr])),rtol=0,atol=1e-6)
            np.testing.assert_allclose(model.predict(scaler.transform(f[te])),cpu[name][te],rtol=0,atol=1e-6)
    ret=HERE/'gpu-return'
    gp=pd.read_csv(ret/'gpu-predictions.csv')
    merged=cpu.copy()
    for name in ['CNN','TCN']:
        selected=gp[gp.model==name].sort_values('sample')
        np.testing.assert_array_equal(selected['sample'],np.arange(len(y)))
        np.testing.assert_allclose(selected.target,y,rtol=0,atol=1e-6)
        merged[name]=selected.prediction.to_numpy()
        for s in splits:
            te=np.array(s['test']);tr=np.array(s['train'])
            ck=torch.load(ret/f'runs/main/{name}/fold{s["fold"]}-refit/model.pt',map_location='cpu',weights_only=False)
            assert ck['input_sha256']==sha(HERE/'input.npz')
            assert ck['train_indices']==tr.tolist() and ck['validation_indices'] is None
            n=ck['normalization'];w=weights(ids[tr]);w/=w.sum()
            np.testing.assert_allclose(n['xmean'],np.sum(x[tr].astype(float).mean(-1)*w[:,None],axis=0)[None,:,None],rtol=0,atol=1e-10)
            net=Net(name);net.load_state_dict(ck['state']);net.eval()
            xx=torch.tensor((x[te]-n['xmean'])/n['xstd'],dtype=torch.float32)
            ff=torch.tensor((np.log1p(d['frames'][te]).astype(float)[:,None]-n['fmean'])/n['fstd'],dtype=torch.float32)
            with torch.no_grad(): pp=net(xx,ff).numpy()*n['ystd']+n['ymean']
            np.testing.assert_allclose(pp,merged[name][te],rtol=0,atol=1e-4)
    models=['Mean','Median','Ridge','RF','CNN','TCN']
    people=[]
    for p,g in merged.groupby('id',sort=True):
        assert len(g.target.unique())==1
        people.append({'id':p,'fold':int(g.fold.iloc[0]),'clips':len(g),'target':float(g.target.iloc[0]),**{m:float(g[m].mean()) for m in models}})
    people=pd.DataFrame(people);people.to_csv(HERE/'person-predictions.csv',index=False)
    summary={}
    rng=np.random.default_rng(20261030);boot=rng.integers(0,len(people),(10000,len(people)))
    base=np.abs(people.Median-people.target).to_numpy()
    for m in models:
        e=(people[m]-people.target).to_numpy();diff=np.abs(e)-base
        summary[m]={'person_MAE':float(np.abs(e).mean()),'person_RMSE':float(np.sqrt((e**2).mean())),
                    'MAE_difference_from_median':float(diff.mean()),'paired_person_bootstrap95_difference':np.quantile(diff[boot].mean(1),[.025,.975]).tolist()}
    gs=json.loads((ret/'gpu-summary.json').read_text());assert gs['folds_finished']==5 and gs['models_finished']==2
    for m in ['CNN','TCN']:
        assert abs(summary[m]['person_MAE']-gs['metrics'][m]['person_MAE'])<1e-5
    save('summary.json',{'people':len(people),'clips':len(x),'target':'clinical TUG seconds','metrics':summary,'scope':'Exploratory12-person repeated development dataset; bootstrap conditional on fitted OOF predictions; no independent clinical validation'})
    save('verification.json',{'passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'actual_TUG_table_linkage':True,'input_equal_to_audited_source':True,'all_person_splits_disjoint':True,'outer_prediction_coverage':True,'cpu_model_replay':True,'GPU_checkpoint_CPU_reinference_tolerance':.0001,'train_only_normalization_recomputed':True,'metrics_independently_recomputed':True,'checkpoints_local':True,'code_sha256':{n:sha(HERE/n) for n in ['common.py','cpu.py','gpu.py','protocol.json','verify.py']}})
    print((HERE/'summary.json').read_text())

if __name__=='__main__':run()
