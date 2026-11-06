"""Replay every inner/outer bundle and independently rebuild selection/metrics."""
from pathlib import Path
import json, sys, collections, csv
import numpy as np, joblib
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
HERE=e.HERE;HEAVY=e.HEAVY
def infer(b,X,ix):
    xx=np.column_stack([X[:,b['cols']],6/np.maximum(X[:,0],.1)])[ix]
    filled=np.where(np.isnan(xx),b['imputer'].statistics_,xx)
    z=(filled-b['scaler'].mean_)/b['scaler'].scale_;m=b['models']
    p={n:np.maximum(.1,m[n].predict(z)) for n in ['base','lds05','lds1','extrap','extrap_lds','high','low','q90']}
    logit=m['logit'].predict_proba(z)[:,1];tree=m['tree'].predict_proba(z)[:,1]
    out={n:p[n] for n in ['base','lds05','lds1','extrap','extrap_lds']};flags={n:np.zeros(len(ix),bool) for n in out}
    for n,s,t in [('predgate10',p['base'],10),('predgate12',p['base'],12),('logitgate03',logit,.3),('logitgate05',logit,.5),('logitgate07',logit,.7),('treegate03',tree,.3),('treegate05',tree,.5)]:
        flags[n]=s>=t;out[n]=np.where(flags[n],p['high'],p['base'])
    out['softgate']=p['low']*(1-logit)+p['high']*logit;flags['softgate']=logit>=.5
    flags['quantilegate']=logit>=.5;out['quantilegate']=np.where(flags['quantilegate'],np.maximum(p['base'],p['q90']),p['base'])
    return out,flags
def score(y,p):
    ae=np.abs(y-p);gg=[y<12,(y>=12)&(y<20),y>=20]
    return {'MAE':float(ae.mean()),'low_MAE':float(ae[gg[0]].mean()),'tail_MAE':float(ae[gg[2]].mean()),'balanced_MAE':float(np.mean([ae[g].mean() for g in gg if g.any()]))}
def selected(scores,policy):
    if policy=='overall_selected':return min(scores,key=lambda k:scores[k]['MAE'])
    eligible=list(scores)
    if policy in ['tail_constrained','two_stage_constrained']:
        base=scores['all/base']
        eligible=[k for k in eligible if scores[k]['MAE']<=base['MAE']+.1 and scores[k]['low_MAE']<=base['low_MAE']+.1]
        if policy=='two_stage_constrained':eligible=[k for k in eligible if 'gate' in k or k=='all/base']
    return min(eligible,key=lambda k:scores[k]['balanced_MAE'])
def check_bundle(b,X,y,ids,tr):
    assert b['train']==tr.tolist() and b['ids']==ids[tr].tolist()
    xx=np.column_stack([X[:,b['cols']],6/np.maximum(X[:,0],.1)])[tr]
    med=np.array([np.nanmedian(xx[:,j]) if np.isfinite(xx[:,j]).any() else 0 for j in range(xx.shape[1])])
    np.testing.assert_allclose(b['imputer'].statistics_,med,atol=1e-12,rtol=0)
    filled=np.where(np.isnan(xx),med,xx)
    np.testing.assert_allclose(b['scaler'].mean_,filled.mean(0),atol=1e-12,rtol=0)
    np.testing.assert_allclose(b['scaler'].var_,filled.var(0),atol=1e-9,rtol=1e-10)
    yy=y[tr];binids=np.floor(yy).astype(int);hist=np.bincount(binids,minlength=int(max(yy))+4).astype(float)
    kernel=np.exp(-.5*(np.arange(-8,9)/2)**2);kernel/=kernel.sum();density=np.convolve(hist,kernel,mode='same')[binids]
    for power,key in [(.5,'weights05'),(1.,'weights1')]:
        w=np.maximum(density,1e-8)**(-power);w/=w.mean();np.testing.assert_allclose(b[key],np.minimum(w,10.),atol=1e-12,rtol=1e-12)
def ci(diff):
    rng=np.random.default_rng(20261104);a=np.asarray(diff)
    means=np.concatenate([a[rng.integers(0,len(a),(1000,len(a)))].mean(1) for _ in range(10)])
    return {'delta_MAE':float(a.mean()),'CI95':np.quantile(means,[.025,.975]).tolist()}
def main():
    prot=json.loads((HERE/'protocol.json').read_text())
    assert e.sha(HERE/'experiment.py')==prot['code_sha256']
    assert e.sha(e.DATA/'features.npz')==prot['source_features_sha256']
    assert e.sha(e.SRC/'effective-splits.json')==prot['source_splits_sha256']
    oldv=json.loads((e.SRC/'verification.json').read_text());assert oldv['passed'] and e.sha(e.SRC/'results.json')==oldv['results_sha256']==prot['source_result_sha256']
    d=np.load(e.DATA/'features.npz');X=d['X'];y=d['y'];ids=d['ids'];names=d['names'].tolist()
    assert len(ids)==len(set(ids))==297
    folds=json.loads((e.SRC/'effective-splits.json').read_text());sel=json.loads((HERE/'selections.json').read_text());hashes=json.loads((HERE/'bundle-hashes.json').read_text())
    assert len(sel)==25 and len(hashes)==200
    rows=json.loads((HEAVY/'predictions.json').read_text());assert len(rows)==32*5*297
    recs=collections.defaultdict(list)
    for r in rows:recs[r['repeat'],r['fold'],r['name']].append(r)
    predictions={n:np.full((5,len(y)),np.nan) for n in sorted({r['name'] for r in rows})}
    route={n:np.zeros((5,len(y)),bool) for n in predictions}
    nreplay=0;maxdelta=0.
    for rep in folds:
      rr=rep['repeat'];seen=[]
      for fold in rep['splits']:
        ff=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test']);seen.extend(te.tolist())
        assert not set(ids[tr])&set(ids[te])
        inn={};outer={};outerflag={}
        for view in ['all','core']:
          for ii,inner in enumerate(fold['inner']):
            it=np.array(inner['train']);iv=np.array(inner['validation'])
            assert set(it)|set(iv)==set(tr) and not set(it)&set(iv)
            path=HEAVY/'models'/f'r{rr}-f{ff}-{view}-inner{ii}.joblib';assert e.sha(path)==hashes[path.name]
            b=joblib.load(path);assert b['cols']==e.columns(names,view);check_bundle(b,X,y,ids,it);pp,flag=infer(b,X,iv);nreplay+=1
            for k,p in pp.items():
                key=view+'/'+k
                if key not in inn:inn[key]=np.full(len(y),np.nan)
                inn[key][iv]=p
          path=HEAVY/'models'/f'r{rr}-f{ff}-{view}-outer.joblib';assert e.sha(path)==hashes[path.name]
          b=joblib.load(path);check_bundle(b,X,y,ids,tr);pp,flags=infer(b,X,te);nreplay+=1
          for k,p in pp.items():outer[view+'/'+k]=p;outerflag[view+'/'+k]=flags[k]
        stored=np.load(HEAVY/'models'/f'r{rr}-f{ff}-inner-predictions.npz');scores={}
        s=next(z for z in sel if z['repeat']==rr and z['fold']==ff);assert s['train']==tr.tolist() and s['test']==te.tolist()
        for k,pp in inn.items():
            delta=float(np.max(abs(pp[tr]-stored[k.replace('/','__')])));assert delta<1e-9;maxdelta=max(maxdelta,delta)
            scores[k]=score(y[tr],pp[tr])
            for metric,value in scores[k].items():assert abs(value-s['scores'][k][metric])<1e-9
        for policy in e.POLICIES:
            key=selected(scores,policy);assert key==s['selected'][policy];outer[policy]=outer[key];outerflag[policy]=outerflag[key]
        for name,pp in outer.items():
            records=recs[rr,ff,name];assert [r['index'] for r in records]==te.tolist()
            for r,i in zip(records,te):assert r['id']==ids[i] and r['true']==y[i]
            delta=float(np.max(abs(pp-np.array([r['pred'] for r in records]))));assert delta<1e-9;maxdelta=max(maxdelta,delta)
            np.testing.assert_array_equal(outerflag[name],[r['flag'] for r in records])
            predictions[name][rr,te]=pp;route[name][rr,te]=outerflag[name]
      assert sorted(seen)==list(range(len(y)));print('TAIL_VERIFIED_REPEAT',rr,flush=True)
    assert nreplay==200 and all(np.isfinite(v).all() for v in predictions.values())
    previous=json.loads((e.DATA/'predictions.json').read_text())
    for view in ['speed_only','speed_plus_landmarks']:
        key='previous/'+view;predictions[key]=np.full((5,len(y)),np.nan);route[key]=np.zeros((5,len(y)),bool)
        for r in previous:
            if r['view']==view:predictions[key][r['repeat'],r['index']]=r['pred']
    age=np.array([r['age'] for r in json.loads((e.SRC/'metadata.json').read_text())])
    groups={'all':np.ones(len(y),bool),'under12':y<12,'12to20':(y>=12)&(y<20),'20plus':y>=20,'age65plus':age>=65}
    report_names=['previous/speed_plus_landmarks','previous/speed_only','all/base','all/lds05','all/lds1','all/predgate10','all/predgate12','all/logitgate05','all/quantilegate']+e.POLICIES
    summary={};comparison={};tail=y>=20;low=y<12
    for n in report_names:
        pp=predictions[n];err=pp-y;ae=abs(err);flag=route[n]
        summary[n]={'groups':{g:{'people':int(mask.sum()),'MAE_repeated':float(ae[:,mask].mean()),'bias_repeated':float(err[:,mask].mean()),
           'MAE_primary':float(ae[0,mask].mean()),'p95_AE_primary':float(np.quantile(ae[0,mask],.95))} for g,mask in groups.items()},
          'MAE_each_repeat':ae.mean(1).tolist(),'tail_MAE_each_repeat':ae[:,tail].mean(1).tolist(),
          'referral':{'people_each_repeat':flag.sum(1).tolist(),'severe_caught_each_repeat':(flag&tail).sum(1).tolist(),'low_TUG_referred_each_repeat':(flag&low).sum(1).tolist()},
          'selected_candidates':dict(collections.Counter(s['selected'][n] for s in sel)) if n in e.POLICIES else None}
        comparison[n]={}
        for base in ['previous/speed_plus_landmarks','all/base']:
            be=abs(predictions[base]-y).mean(0);me=ae.mean(0)
            comparison[n][base]={g:ci((me-be)[mask]) for g,mask in groups.items()}
    severe=[]
    for i in np.flatnonzero(tail):
        severe.append({'id':str(ids[i]),'true_TUG':float(y[i]),'speed':float(X[i,0]),
          'predictions_mean':{n:float(predictions[n][:,i].mean()) for n in report_names}})
    result={'people':297,'severe_people':int(tail.sum()),'summary':summary,'comparisons':comparison,'severe_cases':severe,
      'limits':['Exploratory development data repeatedly used; not an external test','Only9 severe people; bootstrap conditional on fitted models',
        'Gating uses held-out inputs only, never true held-out TUG','No added chair/turn measurements entered models; task pilot is linkage audit only',
        'Ideal projected multiview landmarks; actual phone accuracy unmeasured','Fixed candidate results are descriptive, no outer-result winner selection']}
    e.save(HERE/'results.json',result)
    with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['method','all_MAE','under12_MAE','12to20_MAE','20plus_MAE','65plus_MAE'])
        for n in report_names:w.writerow([n]+[summary[n]['groups'][g]['MAE_repeated'] for g in groups])
    np.savez_compressed(HEAVY/'verified-predictions.npz',y=y,ids=ids,**{n.replace('/','__'):p for n,p in predictions.items()})
    verification={'passed':True,'bundle_hashes_checked':200,'inner_outer_bundles_replayed':nreplay,'submodels_per_bundle':10,
      'all_saved_submodels_replayed':2000,'outer_folds_person_disjoint':25,'inner_folds_person_disjoint':75,
      'train_only_imputation_scaling_weights_checked':True,'candidate_inner_scores_recomputed':25*28,
      'nested_policy_choices_checked':25*len(e.POLICIES),'prediction_rows_checked':len(rows),'max_abs_replay_difference':maxdelta,
      'protocol_sha256':e.sha(HERE/'protocol.json'),'results_sha256':e.sha(HERE/'results.json'),'verification_code_sha256':e.sha(__file__)}
    e.save(HERE/'verification.json',verification);print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
