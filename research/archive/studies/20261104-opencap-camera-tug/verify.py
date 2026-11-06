"""Independent source checks, saved-model replay, and person-paired metrics."""
from pathlib import Path
import csv, hashlib, json, collections, sys
import joblib
import numpy as np
from sklearn.base import clone
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment as e
HERE=e.HERE;HEAVY=e.HEAVY
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def metric(y,p):
    a=np.abs(p-y)
    return {'n':len(y),'MAE':float(a.mean()),'RMSE':float(np.sqrt(np.mean((p-y)**2))),
       'bias_pred_minus_true':float(np.mean(p-y)),'median_AE':float(np.median(a)),
       'p95_AE':float(np.quantile(a,.95)),'fraction_AE_over3':float(np.mean(a>3)),
       'spearman':float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>0 else None}
def paired(a,b):
    dd=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(20261104)
    boot=np.concatenate([dd[rng.integers(0,len(dd),(1000,len(dd)))].mean(1) for _ in range(10)])
    return {'feature_minus_baseline_MAE':float(dd.mean()),'CI95':np.quantile(boot,[.025,.975]).tolist()}
def main():
    prep=json.loads((HERE/'preparation.json').read_text());prot=json.loads((HERE/'protocol.json').read_text())
    assert sha(HEAVY/'features.npz')==prep['features_sha256'];assert sha(HERE/'protocol.json')==prep['protocol_sha256']
    assert sha(HERE/'splits.json')==prot['splits_sha256'];assert sha(HERE/'experiment.py')==prot['code_sha256']
    d=np.load(HEAVY/'features.npz');X=d['X'];y=d['y'];ids=d['ids'];names=d['names'].tolist()
    assert len(ids)==len(set(ids))==prep['people']
    schema=json.loads((HERE/'feature-schema.json').read_text());audit=json.loads((HERE/'source-audit.json').read_text())
    acq={x['member']:x for x in json.loads((HERE/'acquisition.json').read_text())['members']}
    lab={(x['subid'],int(x['visit'])):x for x in csv.DictReader(e.LABEL.open())}
    import zlib
    for name,receipt in acq.items():
        b=(HEAVY/'raw'/name).read_bytes()
        assert len(b)==receipt['bytes'] and hashlib.sha256(b).hexdigest()==receipt['sha256']
        assert zlib.crc32(b)==receipt['crc32']
    sourceids=np.load(e.OLD/'input.npz')['ids']
    olds=json.loads((HERE/'splits.json').read_text())
    effective=json.loads((HERE/'effective-splits.json').read_text())
    for oldrep,newrep in zip(olds,effective):
      for of,nf in zip(oldrep['splits'],newrep['splits']):
        assert set(sourceids[of['test']])&set(ids)==set(ids[nf['test']])
        for oi,ni in zip(of['inner'],nf['inner']):
          assert set(sourceids[oi['validation']])&set(ids)==set(ids[ni['validation']])
    window_selection=json.loads((HERE/'window-selection.json').read_text())
    assert len(window_selection)==298 and sum(r['included'] for r in window_selection)==len(ids)
    for i,a in enumerate(audit):
        p=HEAVY/'raw'/a['member'];b=p.read_bytes()
        assert hashlib.sha256(b).hexdigest()==a['sha256']==acq[a['member']]['sha256']
        assert zlib.crc32(b)==acq[a['member']]['crc32']
        assert ids[i]==a['id'] and abs(float(lab[(a['id'],a['visit'])]['tft_tug'])-y[i])<1e-5
        start,qa=e.walking_start(p);assert start==a['start_time'] and qa['longest_active_span_seconds']>=3-1e-6
        f,q=e.extract(p,a['start_time'])
        np.testing.assert_allclose([f[n] for n in names],X[i],rtol=0,atol=1e-12,equal_nan=True)
    rows=json.loads((HEAVY/'predictions.json').read_text());selection=json.loads((HERE/'selection.json').read_text())
    assert sha(HERE/'effective-splits.json')==prep['effective_splits_sha256']
    splits=json.loads((HERE/'effective-splits.json').read_text());pool=e.candidates()
    assert len(selection)==100 and len(rows)==20*len(y)
    pred=np.full((5,4,len(y)),np.nan);replaymax=0.;innermax=0.
    lookup={(r['repeat'],r['fold'],r['view']):r for r in selection}
    for repeat in splits:
      rr=repeat['repeat']
      seen=[]
      for fold in repeat['splits']:
        ff=fold['fold'];tr=np.array(fold['train']);te=np.array(fold['test']);seen.extend(te)
        assert not set(ids[tr])&set(ids[te]) and set(tr)|set(te)==set(range(len(y)))
        ivall=[]
        for inner in fold['inner']:
            it=inner['train'];iv=inner['validation'];ivall.extend(iv)
            assert not set(it)&set(iv) and set(it)|set(iv)==set(tr)
        assert sorted(ivall)==sorted(tr.tolist())
        for vv,view in enumerate(e.VIEWS):
          s=lookup[rr,ff,view];assert s['train']==tr.tolist() and s['test']==te.tolist()
          xx=X[:,schema['views'][view]];stem=f'r{rr}-f{ff}-{view}'
          path=HEAVY/'models'/(stem+'.joblib');assert sha(path)==s['model_sha256']
          model=joblib.load(path);pp=model.predict(xx[te]);recs=[r for r in rows if (r['repeat'],r['fold'],r['view'])==(rr,ff,view)]
          assert [r['index'] for r in recs]==te.tolist()
          for r,i in zip(recs,te): assert r['id']==ids[i] and r['true']==y[i]
          delta=float(np.max(np.abs(pp-np.array([r['pred'] for r in recs]))));replaymax=max(replaymax,delta);assert delta<1e-9
          # Audit train-only preprocessing from the actual saved model.
          imp=model.steps[0][1];sc=model.steps[1][1]
          expected=np.array([np.nanmedian(xx[tr,j]) if np.isfinite(xx[tr,j]).any() else 0 for j in range(xx.shape[1])])
          np.testing.assert_allclose(imp.statistics_,expected,atol=1e-12,rtol=0)
          fitx=np.where(np.isnan(xx[tr]),expected,xx[tr]);np.testing.assert_allclose(sc.mean_,fitx.mean(0),atol=1e-12,rtol=0)
          np.testing.assert_allclose(sc.var_,fitx.var(0),atol=1e-9,rtol=1e-10)
          inn=np.load(HEAVY/'models'/(stem+'-inner.npz'))
          losses={n:float(np.mean(np.abs(inn[n]-y[tr]))) for n in pool}
          for n in losses: assert abs(losses[n]-s['inner_MAE'][n])<1e-12
          assert min(losses,key=losses.get)==s['selected']
          # Refit selected inner models: verifies selection values against training boundaries.
          ip=np.full(len(y),np.nan)
          for inner in fold['inner']:
            it=np.array(inner['train']);iv=np.array(inner['validation'])
            mm=clone(pool[s['selected']]).fit(xx[it],y[it]);ip[iv]=mm.predict(xx[iv])
          di=float(np.max(np.abs(ip[tr]-inn[s['selected']])));innermax=max(innermax,di);assert di<1e-9
          pred[rr,vv,te]=pp
      assert sorted(seen)==list(range(len(y)))
      print('VERIFIED_REPEAT',rr,flush=True)
    assert np.isfinite(pred).all()
    # Each person has one OOF prediction per repeat/view; no concatenated pseudo-N.
    meta=json.loads((HERE/'metadata.json').read_text());age=np.array([r['age'] for r in meta]);diag=np.array([r['diagnosis'] for r in meta])
    groups={'all':np.ones(len(y),bool),'age65plus':age>=65,'CTL':diag=='CTL','TUG_under12':y<12,'TUG12to20':(y>=12)&(y<20),'TUG20plus':y>=20}
    primary={};repeated={};byrepeat={};sub={}
    for v,view in enumerate(e.VIEWS):
        primary[view]=metric(y,pred[0,v]);ae=np.abs(pred[:,v]-y)
        repeated[view]={'MAE_mean_per_person_across5':float(ae.mean()),'MAE_each_repeat':ae.mean(1).tolist()}
        sub[view]={name:metric(y[g],pred[0,v,g]) for name,g in groups.items() if g.any()}
    comparisons={}
    for baseline in ['speed_only','speed_dynamics']:
        b=e.VIEWS.index(baseline);f=e.VIEWS.index('speed_plus_landmarks')
        comparisons[baseline]={'primary':paired(np.abs(pred[0,f]-y),np.abs(pred[0,b]-y)),
          'repeated_person_mean':paired(np.abs(pred[:,f]-y).mean(0),np.abs(pred[:,b]-y).mean(0)),
          'difference_each_repeat':(np.abs(pred[:,f]-y).mean(1)-np.abs(pred[:,b]-y).mean(1)).tolist()}
    selected=dict(collections.Counter(s['selected'] for s in selection))
    worst=np.argsort(-np.abs(pred[0,3]-y))[:15]
    result={'people':len(y),'primary_repeat':primary,'repeated':repeated,'comparisons':comparisons,'subgroups':sub,
      'selected_models':selected,'worst_feature_predictions':[{'id':str(ids[i]),'true_TUG':float(y[i]),'speed_pred':float(pred[0,0,i]),'feature_pred':float(pred[0,3,i]),'age':float(age[i]),'diagnosis':str(diag[i])} for i in worst],
      'development_gain_supported':all(comparisons[b]['primary']['CI95'][1]<0 and comparisons[b]['repeated_person_mean']['CI95'][1]<0 for b in comparisons),
      'limits':['synthetic ideal lateral projection of multicamera reconstruction, no original RGB or actual single-phone measurement',
        'same previously developed-on cohort, not an independent external holdout','short-window gait timing proxies are not labelled contacts',
        'clinical TUG visit-level linkage, not exact task repetition timing','few high-TUG and older healthy people; no general-elderly claim']}
    save(HERE/'results.json',result)
    with (HERE/'person-predictions.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow(['id','true_TUG','age','diagnosis']+[f'r{r}_{v}' for r in range(5) for v in e.VIEWS])
        for i in range(len(y)):w.writerow([ids[i],y[i],age[i],diag[i]]+pred[:,:,i].reshape(-1).tolist())
    verification={'passed':True,'source_CRC_SHA_files':len(acq),'label_visit_matches':len(y),'feature_replay_people':len(y),
      'person_disjoint_outer_folds':25,'person_disjoint_inner_folds':75,'saved_outer_models_replayed':len(selection),
      'selected_inner_models_refitted':len(selection)*3,'all_candidate_inner_loss_checks':len(selection)*len(pool),
      'prediction_rows':len(rows),'outer_replay_max_abs':replaymax,'inner_refit_max_abs':innermax,
      'train_only_imputation_and_scaling_verified':True,'results_sha256':sha(HERE/'results.json'),
      'protocol_sha256':sha(HERE/'protocol.json'),'features_sha256':sha(HEAVY/'features.npz'),
      'verification_code_sha256':sha(__file__)}
    save(HERE/'verification.json',verification)
    print(json.dumps({'verification':verification,'primary':primary,'comparisons':comparisons}),flush=True)
if __name__=='__main__':main()
