"""Independent raw-window reconstruction, person split, model replay and metrics."""
from common import *
import pandas as pd
import openpyxl
import joblib
from datetime import datetime, timezone
sys.path.append(str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
import torch
from gpu import Net

def verify_raw(d):
    protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf8'))
    audit=json.loads((HERE/'input-audit.json').read_text(encoding='utf8'))
    wb=openpyxl.load_workbook(protocol['labels'],data_only=True)
    import re
    labels={}
    for r in wb.active.values:
        m=re.fullmatch(r'Subject\s*(\d+)',str(r[0] or '').strip())
        if m:labels[str(int(m.group(1)))]=int(r[7])
    wb.close();assert len(labels)==44
    assert sha(Path(protocol['labels']))==protocol['labels_sha256']
    manifest=pd.read_csv(HERE/'manifest.csv',dtype={'id':str})
    raw=HERE.parent/'20261030-sppb-timeseries/raw'
    expected=[];hashes=[]
    for i,r in manifest.iterrows():
        p=raw/r.file
        assert sha(p)==audit['raw_files_sha256'][r.file]
        text=p.read_text(encoding='utf-8-sig').splitlines()
        start=text.index('Trajectories');assert text[start+1]=='100'
        names=text[start+2].split('\t')
        index=[]
        for marker in protocol['markers']:
            col=names.index(f'Subject{int(r.id):02d}:{marker}');index.extend([col,col+1,col+2])
        tab=pd.read_csv(p,sep='\t',skiprows=start+5,header=None,usecols=[0]+index)
        sel=tab[(tab[0]>=r.run_start_frame)&(tab[0]<=r.run_start_frame+300)]
        np.testing.assert_array_equal(sel[0],r.run_start_frame+np.arange(301))
        points=sel[index].to_numpy(dtype=float).reshape(301,12,3)/1000
        assert np.isfinite(points).all()
        pelvis=points[:,:4].mean(1)
        forward=pelvis[-1]-pelvis[0];forward[2]=0;forward/=np.linalg.norm(forward)
        axes=np.array([[forward[1],-forward[0],0],forward,[0,0,1]]).T
        length=np.median(np.linalg.norm(points[:,6:8]-pelvis[:,None],axis=2))
        body=((points-pelvis[:,None])@axes/length).reshape(301,36)
        velocity=np.empty_like(pelvis)
        velocity[1:-1]=(pelvis[2:]-pelvis[:-2])*50
        velocity[0]=(pelvis[1]-pelvis[0])*100
        velocity[-1]=(pelvis[-1]-pelvis[-2])*100
        channels=np.column_stack([body,velocity@axes])
        location=np.arange(91)*(100/30)
        left=np.floor(location).astype(int);right=np.minimum(left+1,300);alpha=location-left
        sampled=(channels[left]*(1-alpha[:,None])+channels[right]*alpha[:,None]).T
        np.testing.assert_allclose(d['x'][i],sampled,rtol=0,atol=1e-6)
        assert d['ids'][i]==r.id and d['y'][i]==labels[r.id]==r.target and d['folds'][i]==r.fold
        expected.append(labels[r.id]);hashes.append(sha(p))
    assert len(manifest)==220 and manifest.groupby('id').size().eq(5).all()
    assert len(set(hashes))==220
    motion_hashes={}
    for i,z in enumerate(d['x']):
        h=hashlib.sha256(z.tobytes()).hexdigest()
        if h in motion_hashes:assert motion_hashes[h]==d['ids'][i], 'Identical cross-person input'
        motion_hashes[h]=d['ids'][i]
    np.testing.assert_array_equal(d['y'],expected)
    return manifest

def run():
    d=load();x,y,ids,folds=(d[k] for k in ['x','y','ids','folds'])
    manifest=verify_raw(d)
    splits=json.loads((HERE/'splits.json').read_text())
    covered=[]
    for s in splits:
        tr=np.array(s['train']);te=np.array(s['test']);covered.extend(te)
        assert set(tr)|set(te)==set(range(len(y))) and not(set(ids[tr])&set(ids[te]))
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
        np.testing.assert_allclose(cpu.Mean[te],np.average(y[tr].astype(float),weights=weights(ids[tr])),rtol=0,atol=1e-6)
        order=np.argsort(y[tr]);ww=weights(ids[tr])[order]
        med=y[tr][order[np.flatnonzero(np.cumsum(ww)>=ww.sum()/2)[0]]]
        np.testing.assert_allclose(cpu.Median[te],med,rtol=0,atol=1e-6)
        for name in ['Ridge','RF']:
            scaler,model=joblib.load(HERE/'cpu-models'/f'{name}-fold{s["fold"]}.joblib')
            np.testing.assert_allclose(scaler.mean_,np.average(f[tr],axis=0,weights=weights(ids[tr])),rtol=0,atol=1e-6)
            np.testing.assert_allclose(model.predict(scaler.transform(f[te])),cpu[name][te],rtol=0,atol=1e-6)
    ret=HERE/'gpu-return';gp=pd.read_csv(ret/'gpu-predictions.csv');merged=cpu.copy()
    for name in ['CNN','TCN']:
        selected=gp[gp.model==name].sort_values('sample')
        np.testing.assert_array_equal(selected['sample'],np.arange(len(y)))
        np.testing.assert_allclose(selected.target,y,rtol=0,atol=1e-6)
        merged[name]=selected.prediction.to_numpy()
        for s in splits:
            te=np.array(s['test']);tr=np.array(s['train'])
            ck=torch.load(ret/f'runs/main/{name}/fold{s["fold"]}-refit/model.pt',map_location='cpu',weights_only=False)
            assert ck['input_sha256']==sha(HERE/'input.npz') and ck['train_indices']==tr.tolist() and ck['validation_indices'] is None
            n=ck['normalization'];w=weights(ids[tr]);w/=w.sum()
            np.testing.assert_allclose(n['xmean'],np.sum(x[tr].astype(float).mean(-1)*w[:,None],axis=0)[None,:,None],rtol=0,atol=1e-10)
            net=Net(name);net.load_state_dict(ck['state']);net.eval()
            xx=torch.tensor((x[te]-n['xmean'])/n['xstd'],dtype=torch.float32)
            ff=torch.tensor((np.log1p(d['frames'][te]).astype(float)[:,None]-n['fmean'])/n['fstd'],dtype=torch.float32)
            with torch.no_grad():pp=net(xx,ff).numpy()*n['ystd']+n['ymean']
            np.testing.assert_allclose(pp,merged[name][te],rtol=0,atol=1e-4)
    models=['Mean','Median','Ridge','RF','CNN','TCN']
    people=[]
    for p,g in merged.groupby('id',sort=True):
        assert len(g.target.unique())==1
        people.append({'id':p,'fold':int(g.fold.iloc[0]),'clips':len(g),'target':float(g.target.iloc[0]),**{m:float(g[m].mean()) for m in models}})
    people=pd.DataFrame(people);people.to_csv(HERE/'person-predictions.csv',index=False)
    summary={};rng=np.random.default_rng(20261030);boot=rng.integers(0,len(people),(10000,len(people)))
    base=np.abs(people.Median-people.target).to_numpy()
    for m in models:
        e=(people[m]-people.target).to_numpy();diff=np.abs(e)-base
        summary[m]={'person_MAE':float(np.abs(e).mean()),'person_RMSE':float(np.sqrt((e**2).mean())),
                    'MAE_difference_from_median':float(diff.mean()),'paired_person_bootstrap95_difference':np.quantile(diff[boot].mean(1),[.025,.975]).tolist()}
    bands=[]
    for score,g in people.groupby('target'):
        bands.append({'actual_SPPB':int(score),'people':len(g),**{m:float(np.mean(np.abs(g[m]-g.target))) for m in models}})
    pd.DataFrame(bands).to_csv(HERE/'errors-by-actual-score.csv',index=False)
    gs=json.loads((ret/'gpu-summary.json').read_text());assert gs['folds_finished']==5 and gs['models_finished']==2
    for m in ['CNN','TCN']:assert abs(summary[m]['person_MAE']-gs['metrics'][m]['person_MAE'])<1e-5
    save('summary.json',{'people':len(people),'clips':len(x),'target':'supplied actual SPPB total points','window_seconds':3,'metrics':summary,'scope':'Optical marker-based exploratory44person study, not video-based/clinical validation; bootstrap conditional on fittedOOFpredictions'})
    save('verification.json',{'passed':True,'checked_utc':datetime.now(timezone.utc).isoformat(),'all220_provider_hashes':True,'actual_SPPB_table_linkage':True,'independent_all220_raw_windows_reconstructed_atol':1e-6,'all_person_splits_disjoint':True,'outer_prediction_coverage':True,'cpu_model_replay':True,'GPU10checkpoint_CPU_reinference_atol':.0001,'train_only_normalization_recomputed':True,'metrics_independently_recomputed':True,'cross_person_exact_input_duplicate':False,'checkpoints_local':True,'code_sha256':{n:sha(HERE/n) for n in ['prepare.py','common.py','cpu.py','gpu.py','protocol.json','verify.py']}})
    print((HERE/'summary.json').read_text())

if __name__=='__main__':run()
