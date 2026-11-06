"""Select using inner people only; evaluation is a separate invocation."""
import argparse
from common import *

def cached(family,f,j):
    p=CACHE/family/('fold%d_%s.npz'%(f,'refit' if j is None else 'inner%d'%j));proof=read(p.with_suffix('.json'))
    assert proof['cache_sha256']==sha(p) and proof['protocol_sha256']==sha(H/'protocol.json')
    assert proof['outer']==f and proof['inner']==j
    return arrays(p),p,proof

def inner(f):
    old=arrays(MD/('analysis/inner_outer%d.npz'%f));sf=read(MD/'analysis/selection.json')['folds'][f]
    assert sf['inner_sha256']==sha(MD/('analysis/inner_outer%d.npz'%f)) and (old['fold']!=f).all()
    results={};hashes={}
    for family in ('CNN','STT'):
        pred=np.full((6,len(old['y']),2),np.nan)
        for j in range(5):
            if j==f:continue
            v,p,proof=cached(family,f,j);ix=v['validation_mask'];pos=old['fold']==j
            for k in ('y','person_id','video_id','fold','row_index'):np.testing.assert_array_equal(v[k][ix],old[k][pos])
            assert not set(v['person_id'][v['train_mask']])&set(v['person_id'][ix])
            assert (v['fold']!=f).all()
            pred[:,pos]=v['corrected'][:,ix];hashes[str(p)]=sha(p);hashes[str(p.with_suffix('.json'))]=sha(p.with_suffix('.json'))
        assert np.isfinite(pred).all();results[family]=pred
    return old,sf,results,hashes

def select():
    proof=read(MD/'analysis/verification.json');assert proof['passed'] and proof['selection_sha256']==sha(MD/'analysis/selection.json')
    selections=[];hashes={str(MD/'analysis/selection.json'):sha(MD/'analysis/selection.json')}
    for f in range(5):
        old,sf,p,h=inner(f);hashes.update(h)
        pair,scores=choose(p['CNN'],p['STT'],old['bilateral'],old['y'],old['person_id'],sf['cnn_weight'],sf['fraction'])
        selections.append(dict(outer=f,pair=list(pair),methods=[read(H/'protocol.json')['methods'][i] for i in pair],
            scores=scores.tolist(),cnn_weight=sf['cnn_weight'],difference_fraction=sf['fraction'],outer_evaluation_used=False))
    save(O/'selection.json',dict(folds=selections,source_hashes=hashes,protocol_sha256=sha(H/'protocol.json'),outer_predictions_opened=False))
    print(json.dumps(selections),flush=True)

def evaluate():
    sel=read(O/'selection.json');assert sel['protocol_sha256']==sha(H/'protocol.json');cfg=read(H/'protocol.json')
    assert read(MD/'analysis/verification.json')['predictions_sha256']==sha(MD/'analysis/predictions.npz')
    old=arrays(MD/'analysis/predictions.npz');n=len(old['y'])
    models={'existing_CNN_STT':old['existing_blend'],'existing_mean_difference':old['inner_selected_difference']}
    for method in cfg['methods']:
        for prefix in ('fixed_','plain_'):models[prefix+method]=np.full((n,2),np.nan)
    models['inner_selected']=np.full((n,2),np.nan);models['inner_selected_plain']=np.full((n,2),np.nan)
    proofs=[]
    for sf in sel['folds']:
        f=sf['outer'];v,meta,p,h=inner(f);pair,scores=choose(p['CNN'],p['STT'],v['bilateral'],v['y'],v['person_id'],sf['cnn_weight'],sf['difference_fraction'])
        assert list(pair)==sf['pair'];np.testing.assert_allclose(scores,sf['scores'],atol=1e-12,rtol=0)
        te=old['fold']==f;parts={}
        for family in ('CNN','STT'):
            x,path,proof=cached(family,f,None);ix=x['validation_mask']
            for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(x[key][ix],old[key][te])
            assert not set(x['person_id'][x['train_mask']])&set(old['person_id'][te])
            parts[family]=x['corrected'][:,ix];proofs.append(dict(path=str(path),sha256=sha(path)))
        w=sf['cnn_weight'];t=sf['difference_fraction'];bilateral=old['bilateral_STT'][te]
        for k,method in enumerate(cfg['methods']):
            models['fixed_'+method][te]=combine(parts['CNN'][k],parts['STT'][k],bilateral,w,t)
            models['plain_'+method][te]=w*parts['CNN'][k]+(1-w)*parts['STT'][k]
        models['inner_selected'][te]=combine(parts['CNN'][pair[0]],parts['STT'][pair[1]],bilateral,w,t)
        models['inner_selected_plain'][te]=w*parts['CNN'][pair[0]]+(1-w)*parts['STT'][pair[1]]
    # A calibrated mean re-export must reproduce the existing pipelines.
    deltas=[float(np.abs(models['fixed_mean']-models['existing_mean_difference']).max()),float(np.abs(models['plain_mean']-models['existing_CNN_STT']).max())]
    assert max(deltas)<1e-4
    people=np.unique(old['person_id']);y=old['y'];ids=old['person_id'];es={k:errors(y,v,ids) for k,v in models.items()}
    report=dict(mae={k:float(v.mean()) for k,v in es.items()},fold_mae={},bands10={},high90={},paired={},
        mean_reproduction_max=deltas,people=len(people),records=n,selection_sha256=sha(O/'selection.json'),outer_caches=proofs,
        selected_methods=[x['methods'] for x in sel['folds']],limitations=cfg['limitations'])
    high=np.array([y[ids==i].mean()>=90 for i in people])
    for k,p in models.items():
        report['fold_mae'][k]=[float(errors(y[old['fold']==f],p[old['fold']==f],ids[old['fold']==f]).mean()) for f in range(5)]
        banderror=np.abs(np.floor(p/10)-np.floor(y/10))
        report['bands10'][k]={name:float(np.mean([v[ids==i].mean() for i in people])) for name,v in [('exact',banderror==0),('within_one',banderror<=1),('two_or_more',banderror>=2)]}
        report['high90'][k]=dict(people=int(high.sum()),mae=float(es[k][high].mean()))
    delta=es['inner_selected']-es['existing_mean_difference'];rng=np.random.default_rng(20261031)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(10000)])
    report['paired']=dict(delta=float(delta.mean()),better_people=int((delta<0).sum()),worse_people=int((delta>0).sum()),descriptive_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    CACHE.mkdir(parents=True,exist_ok=True);dest=CACHE/'evaluation.npz';np.savez_compressed(dest,**models,**{k:old[k] for k in ('y','person_id','video_id','fold')})
    report.update(predictions_path=str(dest),predictions_sha256=sha(dest),optimizer_steps=0,protected_final_data_opened=False)
    save(O/'comparison.json',report);print(json.dumps(report),flush=True)

def control():
    cfg=read(H/'protocol.json');ref=arrays(S/'cnn-record-loss692/analysis/window_control/unpacked/results/oof_predictions.npz')
    pred=np.full((6,len(ref['y']),2),np.nan);proofs=[]
    for f in range(5):
        x,p,proof=cached('control',f,None);te=ref['fold']==f;ix=x['validation_mask'];np.testing.assert_array_equal(x['video_id'][ix],ref['video_id'][te])
        pred[:,te]=x['corrected'][:,ix];proofs.append(dict(path=str(p),sha256=sha(p)))
    assert np.abs(pred[0]-ref['predictions'][:,0]).max()<1e-4
    worst=np.unravel_index(np.abs(ref['predictions'][:,0]-ref['y']).argmax(),ref['y'].shape)
    result=dict(mae={k:float(errors(ref['y'],p,ref['person_id']).mean()) for k,p in zip(cfg['methods'],pred)},
        worst_original_case=dict(row=int(worst[0]),side=int(worst[1]),truth=float(ref['y'][worst]),predictions={k:float(p[worst]) for k,p in zip(cfg['methods'],pred)}),
        fixed_diagnostic_only=True,no_method_selected_on_outer=True,sources=proofs)
    save(O/'control_diagnostic.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate','control']);a=p.parse_args();globals()[a.stage]()
