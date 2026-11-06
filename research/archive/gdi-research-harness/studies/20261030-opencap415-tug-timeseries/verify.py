from pathlib import Path
import sys,json,hashlib,io,importlib.util,zlib,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'outputs/gdi-ridge-experiment-20261016/deps'))
import numpy as np
import pandas as pd
import joblib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    labels=pd.read_csv(HERE.parent/'20261030-older44-sppb-timeseries/candidate-audit/nmd_opencap_participant_info.csv').set_index(['subid','visit'])
    idx={x['name']:x for x in json.loads((HERE/'archive-index.json').read_text())}
    for r in json.loads((HERE/'acquisition.json').read_text()):
        p=HERE/'raw'/r['name'];assert sha(p)==r['sha256'];assert zlib.crc32(p.read_bytes())==idx[r['name']]['crc']
    matched=[]
    for r in json.loads((HERE/'overlap-sha256.json').read_text()):
        p=HERE/'raw'/r['new_file'];assert sha(p)==r['sha256'];matched.append(r['old_id'])
    assert len(set(matched))==124
    summaries={};allpred=[];ref=None
    for condition,secs in [('1',1),('3',3),('6',6),('all3',3)]:
        p=HERE/f'{condition}s';d=dict(np.load(p/'input.npz'));a=json.loads((p/'input-audit.json').read_text());assert sha(p/'input.npz')==a['input_sha256'] and sha(p/'protocol.json')==a['protocol_sha256']
        man=pd.read_csv(p/'manifest.csv');assert len(set(d['ids']))==len(d['ids'])
        assert len({hashlib.sha256(x.tobytes()).hexdigest() for x in d['x']})==len(d['x'])
        if ref is None:ref=d
        elif condition!='all3':
            for n in ['ids','folds','y']:assert np.array_equal(ref[n],d[n])
        else:
            for n in ['ids','folds','y']:assert np.array_equal(ref[n],d[n][:len(ref[n])])
        cols=json.loads((p/'protocol.json').read_text())['channels'][:17]
        for k,r in man.iterrows():
            ident,visit=re.search(r'(sub-\d+)/visit-(\d+)',r.file).groups();assert ident==r.id==d['ids'][k] and int(visit)==r.visit
            assert np.isclose(r.target,labels.loc[(r.id,r.visit),'tft_tug'],atol=1e-10) and np.isclose(r.target,d['y'][k],atol=1e-10)
            f=HERE/'raw'/r.file;assert sha(f)==r.source_sha256
            text=f.read_text().splitlines();h=text.index('endheader');raw=pd.read_csv(io.StringIO('\n'.join(text[h+1:])),sep=r'\s+')
            t=raw.time.to_numpy();end=np.searchsorted(t,r.start_time+secs);b=raw.iloc[int(r.start_frame):end+1];tt=b.time.to_numpy();grid=r.start_time+np.arange(secs*30+1)/30
            # Independent linear interpolation and explicit finite differences.
            hi=np.searchsorted(tt,grid,side='right').clip(1,len(tt)-1);lo=hi-1;w=(grid-tt[lo])/(tt[hi]-tt[lo]);v=b[cols+['pelvis_tx','pelvis_ty','pelvis_tz']].to_numpy();interp=(v[lo]*(1-w[:,None])+v[hi]*w[:,None]).T
            pos=interp[-3:];vel=np.empty_like(pos);vel[:,1:-1]=(pos[:,2:]-pos[:,:-2])*15;vel[:,0]=(pos[:,1]-pos[:,0])*30;vel[:,-1]=(pos[:,-1]-pos[:,-2])*30
            expected=np.concatenate([interp[:17],np.hypot(vel[0],vel[2])[None],vel[1,None]])
            assert np.allclose(expected,d['x'][k],atol=1e-5,rtol=1e-6)
        x=d['x'];features=np.concatenate([x.mean(-1),x.std(-1),np.abs(np.diff(x,axis=-1)).mean(-1)],axis=1)
        pred=pd.read_csv(p/'cpu-predictions.csv');splits=json.loads((p/'splits.json').read_text());seen=[]
        for f in splits:
            tr,te=f['train'],f['test'];seen+=te;assert not set(d['ids'][tr])&set(d['ids'][te])
            for inner in f['inner']:
                assert set(inner['train'])|set(inner['validation'])==set(tr)
                assert not set(inner['train'])&set(inner['validation'])
            for m in ['Ridge','RF']:
                scale,model=joblib.load(p/'cpu-models'/f'{m}-fold{f["fold"]}.joblib')
                assert np.allclose(scale.mean_,features[tr].mean(0),atol=1e-5)
                assert np.allclose(model.predict(scale.transform(features[te])),pred.loc[te,m],atol=1e-8)
        assert sorted(seen)==list(range(len(x)))
        if (p/'gpu-predictions.csv').exists():
            sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'))
            import torch
            torch.set_num_threads(2)
            sys.path.insert(0,str(p));sys.modules.pop('common',None)
            spec=importlib.util.spec_from_file_location(f'gpu_{secs}',p/'gpu.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            gp=pd.read_csv(p/'gpu-predictions.csv')
            for name in ['CNN','TCN']:
                values=gp[gp.model==name].set_index('sample').prediction.reindex(range(len(x)));assert values.notna().all();pred[name]=values.to_numpy()
                for f in splits:
                    ck=torch.load(p/f'runs/main/{name}/fold{f["fold"]}-refit/model.pt',map_location='cpu',weights_only=False);n=ck['normalization'];tr=np.array(f['train']);te=np.array(f['test']);assert ck['train_indices']==tr.tolist();assert ck['input_sha256']==sha(p/'input.npz')
                    mu=x[tr].astype(float).mean((0,2))[None,:,None];sd=np.maximum(np.sqrt(((x[tr]-mu)**2).mean((0,2)))[None,:,None],.001)
                    assert np.allclose(mu,n['xmean']) and np.allclose(sd,n['xstd']) and np.isclose(n['ymean'],d['y'][tr].mean())
                    model=mod.Net(name);model.load_state_dict(ck['state']);model.eval()
                    with torch.no_grad():
                        yy=model(torch.tensor((x[te]-n['xmean'])/n['xstd'],dtype=torch.float32),torch.tensor((np.log1p(d['frames'][te,None])-n['fmean'])/n['fstd'],dtype=torch.float32)).numpy()*n['ystd']+n['ymean']
                    assert np.allclose(yy,values.iloc[te],atol=1e-4,rtol=0)
            sys.path.remove(str(p))
        models=[n for n in ['Mean','Median','Ridge','RF','CNN','TCN'] if n in pred]
        summaries[condition]={m:{'MAE_seconds':float(np.mean(abs(pred[m]-d['y']))),'RMSE_seconds':float(np.sqrt(np.mean((pred[m]-d['y'])**2)))} for m in models}
        pred['seconds']=secs;pred['condition']=condition;allpred.append(pred)
    pd.concat(allpred).to_csv(HERE/'comparison-predictions.csv',index=False)
    (HERE/'summary.json').write_text(json.dumps({'people':len(ref['ids']),'metrics':summaries},indent=2))
    (HERE/'verification.json').write_text(json.dumps({'passed':True,'original_source_CRC_SHA_verified':True,'overlap124_SHA_equal':True,'all_windows_independently_reconstructed':True,'primary_conditions_same_people_labels_folds':True,'expanded_cohort_preserves_primary_people_folds':True,'CPU_models_replayed':True,'GPU_models_replayed':all((HERE/f'{s}s/gpu-predictions.csv').exists() for s in ['1','3','6','all3']),'train_only_normalization_checked':True},indent=2))
    print(json.dumps({'people':len(ref['ids']),'metrics':summaries},indent=2))
if __name__=='__main__':main()
