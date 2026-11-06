"""Reconstruct saved-context models and independently evaluate every held-out person."""
from foundation import *
import argparse, time, gc, warnings
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
warnings.filterwarnings('ignore',message='auto_scale_n_estimators is deprecated')
BASE=Path('D:/Projects/D_P/studies/20261103-sppb-lightgbm')
def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y;low=y<=8;pl=p<=8
    out=dict(n=len(y),MAE=float(np.mean(abs(e))),RMSE=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),R2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)) if np.ptp(y)>0 else None,spearman=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None,within1=float(np.mean(abs(e)<=1)),errors_gt2=int(np.sum(abs(e)>2)),TP=int(sum(low&pl)),FN=int(sum(low&~pl)),TN=int(sum(~low&~pl)),FP=int(sum(~low&pl)),bands={})
    for label,mask in [('0-4',y<=4),('5-8',(y>=5)&(y<=8)),('9-10',(y>=9)&(y<=10)),('11-12',y>=11)]:
        out['bands'][label]=dict(n=int(sum(mask)),MAE=float(np.mean(abs(e[mask]))) if mask.any() else None,bias=float(np.mean(e[mask])) if mask.any() else None)
    return out
def contrast(y,a,b):
    difference=abs(a-y)-abs(b-y);rng=np.random.RandomState(20261103);draws=[]
    for _ in range(20):draws.extend(difference[rng.randint(0,len(y),(500,len(y)))].mean(axis=1))
    return dict(MAE_difference=float(difference.mean()),CI95=np.quantile(draws,[.025,.975]).tolist())
def walk(node,row):
    while 'leaf_value' not in node:
        value=row[node['split_feature']]
        left=node['default_left'] if np.isnan(value) else value<=float(node['threshold'])
        assert node['decision_type']=='<='
        node=node['left_child'] if left else node['right_child']
    return node['leaf_value']
def main():
    parser=argparse.ArgumentParser();parser.add_argument('scope',choices=['small','large']);args=parser.parse_args()
    protocol=read(HERE/'protocol.json');receipt=read(HERE/'preparation-receipt.json');ph=sha(HERE/'protocol.json')
    assert ph==receipt['protocol_sha256'] and sha(BIG/'cohorts.json')==receipt['cohorts_sha256']
    for p,h in protocol['sources'].items():assert sha(p)==h,p
    for w in read(HERE/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    cohorts=read(BIG/'cohorts.json');base=read(BASE/'comparison.json');results={};checks=[];start=time.monotonic()
    (BIG/'replay').mkdir(exist_ok=True)
    for name,pp in cohorts.items():
        if (name=='mobilised')!=(args.scope=='large'):continue
        ids=[p['id'] for p in pp];lookup={s:i for i,s in enumerate(ids)};y=np.array([p['sppb'] for p in pp]);folds=np.array([p['fold'] for p in pp]);preds={};counts={}
        if name in base:
            b=base[name];assert b['ids']==ids;np.testing.assert_array_equal(b['y'],y);np.testing.assert_array_equal(b['folds'],folds)
            for key,value in b['predictions'].items():preds[key]=np.array(value)
        else:
            preds['Mean']=np.array([y[folds!=f].mean() for f in folds])
        expected=0
        for condition,(X,names) in matrices(name,pp).items():
            if name in base:assert base[name]['feature_names'][condition+'_LGBM']==names
            for kind in ['TabICLv2','TabPFNv2']+(['LGBM'] if name=='mobilised' else []):
                key=condition+'_'+kind;preds[key]=np.full(len(y),np.nan);counts[key]=np.zeros(len(y),int)
                for f in sorted(set(folds)):
                    stem=f'{name}-{condition}-{f}';path=BIG/'predictions'/(stem+'-'+kind+'.json');rec=read(path);expected+=1
                    assert rec['protocol_sha256']==ph and rec['factory_sha256']==sha(HERE/'foundation.py')
                    tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f)
                    assert rec['train_ids']==[ids[i] for i in tr] and rec['test_ids']==[ids[i] for i in te]
                    assert not set(rec['train_ids'])&set(rec['test_ids'])
                    assert rec['feature_names']==names and rec['context_sha256']==sha(rec['context'])
                    with np.load(rec['context'],allow_pickle=False) as c:
                        np.testing.assert_array_equal(c['train_ids'],np.array(ids)[tr]);np.testing.assert_array_equal(c['test_ids'],np.array(ids)[te]);np.testing.assert_allclose(c['X_train'],X[tr],equal_nan=True);np.testing.assert_allclose(c['X_test'],X[te],equal_nan=True);np.testing.assert_array_equal(c['y_train'],y[tr]);assert c['feature_names'].tolist()==names
                        replayfile=BIG/'replay'/(stem+'-'+kind+'.json')
                        if replayfile.exists():
                            replay=read(replayfile);assert replay['prediction_sha256']==sha(path) and replay['context_sha256']==sha(rec['context']) and replay['factory_sha256']==sha(HERE/'foundation.py');p=np.array(replay['prediction'])
                        else:
                            t=time.monotonic()
                            if kind=='LGBM':
                                assert rec['model_sha256']==sha(rec['model']);model=lgb.Booster(model_file=rec['model']);p=model.predict(c['X_test'],num_threads=2)
                                trees=read(Path(rec['model']).with_suffix('.trees.json'))['tree_info'];ind=np.array([sum(walk(t['tree_structure'],r) for t in trees) for r in c['X_test']]);np.testing.assert_allclose(ind,p,atol=1e-9,rtol=0)
                            else:
                                model=factory(kind);model.fit(c['X_train'],c['y_train']);p=model.predict(c['X_test'])
                            save(replayfile,dict(prediction=p.tolist(),seconds=time.monotonic()-t,prediction_sha256=sha(path),context_sha256=sha(rec['context']),factory_sha256=sha(HERE/'foundation.py')));del model;gc.collect()
                    err=float(np.max(abs(p-np.array(rec['prediction']))));np.testing.assert_allclose(p,rec['prediction'],atol=1e-5,rtol=1e-6)
                    if kind=='LGBM':
                        inner=read(BIG/'inner'/(stem+'.json'));losses=[]
                        for ci in range(16):
                            es=[]
                            for ii,(it,iv) in enumerate(KFold(3,shuffle=True,random_state=20261031).split(tr)):
                                r=next(r for r in inner if r['candidate']==ci and r['inner']==ii)
                                assert r['train']==[ids[i] for i in tr[it]] and r['evaluation']==[ids[i] for i in tr[iv]]
                                es.extend(abs(np.array(r['prediction'])-y[tr[iv]]))
                            losses.append(float(np.mean(es)))
                        np.testing.assert_allclose(losses,rec['selection']['losses'],atol=1e-12);assert int(np.argmin(losses))==rec['selection']['candidate']
                    preds[key][te]=p;counts[key][te]+=1;checks.append(dict(model=stem+'-'+kind,max_error=err,prediction_sha256=sha(path)))
                    print('verified',stem,kind,'max_error',err,flush=True)
                    save(HERE/f'review-progress-{args.scope}.json',dict(status='replaying',verified=len(checks),last=stem+'-'+kind))
        assert all(np.all(v==1) for v in counts.values());assert all(np.isfinite(v).all() for v in preds.values())
        contrasts={}
        for kind in ['TabICLv2','TabPFNv2','LGBM']:
            for a,b in [('features','speed'),('features_GDI','features'),('speed_profile','speed'),('features','speed_profile')]:
                ka=a+'_'+kind;kb=b+'_'+kind
                if ka in preds and kb in preds:contrasts[ka+' minus '+kb]=contrast(y,preds[ka],preds[kb])
        for cond in matrices(name,pp):
            for a,b in [('TabICLv2','LGBM'),('TabPFNv2','LGBM'),('TabICLv2','TabPFNv2')]:
                ka=cond+'_'+a;kb=cond+'_'+b;contrasts[ka+' minus '+kb]=contrast(y,preds[ka],preds[kb])
        groups={'age65plus':np.array([p['age']>=65 for p in pp])}
        if name=='mobilised':
            for disease in sorted(set(p['cohort'] for p in pp)):groups[disease]=np.array([p['cohort']==disease for p in pp])
            incomplete={str(r['participantid']) for r in read(HERE/'inventory.json')['official_total_with_incomplete_components']};groups['complete_SPPB_components']=np.array([p['id'] not in incomplete for p in pp])
            groups['representative_speed_observed']=np.array([p['features']['ws_30_p90_w'] is not None for p in pp])
        subgroup_contrasts={g:{key:contrast(y[m],preds[key.split(' minus ')[0]][m],preds[key.split(' minus ')[1]][m]) for key in contrasts} for g,m in groups.items()} if name=='mobilised' else {}
        results[name]=dict(ids=ids,y=y.tolist(),folds=folds.tolist(),predictions={k:v.tolist() for k,v in preds.items()},metrics={k:metrics(y,p) for k,p in preds.items()},contrasts=contrasts,subgroups={g:{k:metrics(y[m],p[m]) for k,p in preds.items()} for g,m in groups.items()},subgroup_contrasts=subgroup_contrasts)
        save(HERE/f'comparison-{args.scope}.json',results)
    save(HERE/f'verification-{args.scope}.json',dict(status='passed',reconstructed_models=len(checks),maximum_replay_error=max(c['max_error'] for c in checks),models=checks,elapsed_seconds=time.monotonic()-start,protocol_sha256=ph,comparison_sha256=sha(HERE/f'comparison-{args.scope}.json'),review_code_sha256=sha(Path(__file__)),notes='Fixed foundation weights + stored training contexts reconstructed for every model; LightGBM saved models reloaded and independently tree-walked. Metrics and intervals calculated only after full held-out predictions; conditional intervals do not include full training uncertainty.'))
    save(HERE/f'review-progress-{args.scope}.json',dict(status='passed',verified=len(checks)))
if __name__=='__main__':main()
