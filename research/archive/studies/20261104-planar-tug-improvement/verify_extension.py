"""Verify short-window provenance, zero-adaptation test, and pooled person folds."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,csv,zlib
import numpy as np,joblib
import sensor_loop as s
import gstride_extension as g
from verify_batch import replay,metric,delta
HERE=s.HERE;HEAVY=s.HEAVY
def checkprep(m,X):
    med=np.array([np.nanmedian(X[:,j]) if np.isfinite(X[:,j]).any() else 0 for j in range(X.shape[1])])
    np.testing.assert_allclose(m['imputer'].statistics_,med,atol=1e-12,rtol=0)
    raw=np.where(np.isnan(X),med,X);raw=np.column_stack([raw,1/np.maximum(raw[:,0],.1)])
    np.testing.assert_allclose(m['scaler'].mean_,raw.mean(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(m['scaler'].var_,raw.var(0),atol=1e-10,rtol=1e-10)
def metapred(m,A,y,T):
    center=A.mean(0);scale=np.maximum(A.std(0),1e-6);ym=y.mean()
    np.testing.assert_allclose(center,m['center'],atol=1e-12,rtol=0);np.testing.assert_allclose(scale,m['scale'],atol=1e-12,rtol=0)
    assert abs(ym-m['ymean'])<1e-12
    z=(A-center)/scale;w=np.where(y>=20,m['tail'],np.where(y>=12,np.sqrt(m['tail']),1.))
    grad=z.T@(w*(z@m['coef']-(y-ym)))+10*m['coef'];active=m['coef']>1e-10
    assert np.max(abs(grad[active]),initial=0)<1e-6 and np.min(grad[~active],initial=0)>-1e-6
    return np.maximum(.1,(T-center)/scale@m['coef']+ym)
def sourcecheck():
    prot=json.loads((HERE/'extension-protocol.json').read_text());assert s.sha(HERE/'gstride_extension.py')==prot['code_sha256']
    assert s.sha(g.G/'GSTRIDE_DDBB.xlsx')==prot['workbook_sha256'] and s.sha(g.G/'workbook-cells.json')==prot['workbook_cells_sha256']
    unique=set()
    for name in ['gstride-parameters-acquisition.json','gstride-segmentation-acquisition.json']:
        r=json.loads((HERE/name).read_text())
        for item in r['members']:
            file=g.G/'processed'/item['member'];assert s.sha(file)==item['sha256'] and zlib.crc32(file.read_bytes())==item['crc32'];unique.add(str(file))
    audit=json.loads((HERE/'gstride-short-window-audit.json').read_text());assert s.sha(g.G/'short-window-input.npz')==audit['input_sha256']
    gd=np.load(g.G/'short-window-input.npz');assert len(gd['y'])==len(set(gd['ids']))==audit['people']
    for i,r in enumerate(audit['included']):
        assert r['id']==gd['ids'][i] and r['y']==gd['y'][i]
        path=g.G/'processed/_GAIT/_PARAMETERS'/f"{r['id']}_GAIT_PARAMETERS.csv"
        evpath=g.G/'processed/_GAIT/_SEGMENTATION'/f"{r['id']}_GAIT_SEGMENTATION.csv"
        pp={int(float(x['index'])):x for x in csv.DictReader(path.open(encoding='utf-8-sig'))}
        ee={int(float(x['index'])):x for x in csv.DictReader(evpath.open(encoding='utf-8-sig'))}
        chosen=[pp[j] for j in r['stride_indices']];events=[ee[j] for j in r['stride_indices']]
        duration=sum(float(x['CycleDuration[s]']) for x in chosen);assert duration<=4+1e-9 and abs(duration-r['duration_seconds'])<1e-12
        for j,x in enumerate(chosen):
            assert abs(float(x['TurningAngle[degrees]']))<20 and .6<=float(x['CycleDuration[s]'])<=3
            if j:assert events[j]['HS_ini[sample]']==events[j-1]['HS_fin[sample]']
        values=[np.mean([float(x[n]) for x in chosen]) for n in ['Stride_Velocity[m/s]','CycleDuration[s]','Per_Swing[%CD]']];values[2]/=100
        np.testing.assert_allclose(values,gd['X'][i],atol=1e-12,rtol=0)
    return gd,len(unique)
def main():
    gd,sourcefiles=sourcecheck();fd,FX,sets=s.load();nF=len(fd['y']);nG=len(gd['y'])
    assert json.loads((HERE/'sensor-verification.json').read_text())['passed']
    external=np.load(g.G/'external-zero-adaptation.npz');external_receipt=json.loads((HERE/'gstride-external-receipt.json').read_text());assert s.sha(g.G/'external-zero-adaptation.npz')==external_receipt['output_sha256']
    source=json.loads((HERE/'sensor-receipt.json').read_text());external_replays=0
    for view in ['speed','core3']:
        XX=gd['X'][:,:1] if view=='speed' else gd['X'];foldpred=[]
        for r in source:
            if r['view']!=view:continue
            T=[]
            for k in range(len(s.CONFIGS)):
                mr=next(z for z in r['models'] if z['candidate']==k and z['inner'] is None);path=HEAVY/'sensor-models'/mr['file'];assert s.sha(path)==mr['sha256']
                T.append(replay(joblib.load(path),XX));external_replays+=1
            ctx=np.load(HEAVY/'sensor-models'/r['context']);meta=joblib.load(HEAVY/'sensor-models'/r['meta'])['stacked'];tr=np.array(r['train'])
            foldpred.append(metapred(meta,ctx['A'],fd['y'][tr],np.column_stack(T)))
        np.testing.assert_allclose(np.mean(foldpred,0),external[view],atol=1e-9,rtol=0)
    recs=json.loads((HERE/'extension-receipt.json').read_text());assert len(recs)==20
    predictions={};nmodels=0;maxdiff=0.
    for r in recs:
        mode=r['mode'];view=r['view'];f=r['fold'];path=HEAVY/'extension-models'/r['context'];assert s.sha(path)==r['context_sha256'];ctx=np.load(path)
        GX=gd['X'][:,:1] if view=='speed' else gd['X']
        if mode=='local_GSTRIDE':X=GX;y=gd['y'];ids=gd['ids'];fold=gd['fold'];cohort=np.array(['G']*nG)
        else:
            X=np.concatenate([FX[view],GX]);y=np.r_[fd['y'],gd['y']];ids=np.r_[np.array(['F'+str(x) for x in fd['ids']]),gd['ids']];fold=np.r_[fd['fold'],gd['fold']];cohort=np.r_[np.array(['F']*nF),np.array(['G']*nG)]
        tr=np.array(r['train']);te=np.array(r['test']);assert np.array_equal(tr,np.flatnonzero(fold!=f)) and np.array_equal(te,np.flatnonzero(fold==f));assert not set(ids[tr])&set(ids[te])
        assert np.array_equal(ctx['ids'],ids) and np.array_equal(ctx['y'],y) and np.array_equal(ctx['cohort'],cohort)
        A=np.full_like(ctx['A'],np.nan);T=np.empty_like(ctx['T'])
        for mr in r['models']:
            file=HEAVY/'extension-models'/mr['file'];assert s.sha(file)==mr['sha256'];m=joblib.load(file)
            it=np.array(mr['train']);iv=np.array(mr['test']);assert not set(ids[it])&set(ids[iv]);checkprep(m,X[it]);pp=replay(m,X[iv]);k=mr['candidate']
            if mr['inner'] is None:assert np.array_equal(it,tr) and np.array_equal(iv,te);T[:,k]=pp
            else:assert set(it)|set(iv)==set(tr);A[np.searchsorted(tr,iv),k]=pp
            nmodels+=1
        for a,b in [(A,ctx['A']),(T,ctx['T'])]:dd=float(abs(a-b).max());assert dd<1e-9;maxdiff=max(maxdiff,dd)
        mp=HEAVY/'extension-models'/r['meta'];assert s.sha(mp)==r['meta_sha256'];meta=joblib.load(mp);pp=metapred(meta,A,y[tr],T)
        np.testing.assert_allclose(pp,ctx['prediction'],atol=1e-9,rtol=0)
        key=mode+'/'+view
        if key not in predictions:predictions[key]=np.full(len(y),np.nan)
        predictions[key][te]=pp
        print('EXTENSION_VERIFIED',mode,view,f,flush=True)
    assert nmodels==960 and all(np.isfinite(p).all() for p in predictions.values())
    sourcepred=np.load(HEAVY/'sensor-verified-predictions.npz');result={};comparisons={}
    for view in ['speed','core3']:
        result['external/'+view]=metric(gd['y'],external[view])
        local=predictions['local_GSTRIDE/'+view];pooled=predictions['pooled/'+view]
        result['local_GSTRIDE/'+view]=metric(gd['y'],local);result['pooled_GSTRIDE/'+view]=metric(gd['y'],pooled[nF:]);result['pooled_FRAILPOL/'+view]=metric(fd['y'],pooled[:nF])
        comparisons[view]={'pooled_vs_source_on_FRAILPOL':delta(abs(pooled[:nF]-fd['y']),abs(sourcepred[view+'__stacked']-fd['y'])),
          'pooled_vs_local_on_GSTRIDE':delta(abs(pooled[nF:]-gd['y']),abs(local-gd['y']))}
    result['paired_comparisons']=comparisons
    result['cohort_counts']={'FRAILPOL':nF,'GSTRIDE':nG,'pooled':nF+nG,'GSTRIDE_TUG20plus':int((gd['y']>=20).sum())}
    result['limits']='New sensor proxy cohort, not RGB/MediaPipe. Onefoot/bilateral and event-detector/task differences persist; external first-pass is separate from local/pooled development CV.'
    s.save(HERE/'extension-results.json',result)
    np.savez_compressed(HEAVY/'extension-verified-predictions.npz',**{k.replace('/','__'):p for k,p in predictions.items()})
    s.save(HERE/'extension-verification.json',{'passed':True,'source_CRC_SHA_files':sourcefiles,'short_window_people':nG,'models_replayed':nmodels,
       'source_models_replayed_for_external':external_replays,'meta_models_checked':20,'outer_person_disjoint_cases':20,'max_replay_abs':maxdiff,
       'results_sha256':s.sha(HERE/'extension-results.json'),'protocol_sha256':s.sha(HERE/'extension-protocol.json')})
    print('EXTENSION_VERIFICATION_PASSED',flush=True)
if __name__=='__main__':main()
