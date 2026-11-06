"""Pre-declared single comparison; no selection on outer people."""
from common import *
cfg=read(H/'protocol.json')
assert read(MD/'analysis/verification.json')['predictions_sha256']==sha(MD/'analysis/predictions.npz')
old=arrays(MD/'analysis/predictions.npz');sel=read(MD/'analysis/selection.json')['folds'];n=len(old['y'])
models={k:old[k] for k in ('CNN','STT','existing_blend','inner_selected_difference')}
for k in ('CNN_bag','STT_bag','bagged','refit_reproduced'):models[k]=np.full((n,2),np.nan)
members=[];sources={}
for sf in sel:
    f=sf['outer'];te=old['fold']==f;parts={};refit={}
    for family in ('CNN','STT'):
        x,p=racache(family,f,None);ix=x['validation_mask'];sources[str(p)]=sha(p)
        for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(x[key][ix],old[key][te])
        stack=[x['corrected'][0][ix]];used=['refit']
        for j in [j for j in range(5) if j!=f]:
            q=CACHE/family/('fold%d_inner%d.npz'%(f,j));proof=read(q.with_suffix('.json'))
            assert proof['cache_sha256']==sha(q) and proof['protocol_sha256']==sha(H/'protocol.json');sources[str(q)]=sha(q)
            v=arrays(q)
            for key in ('y','person_id','video_id','fold'):np.testing.assert_array_equal(v[key],old[key][te])
            members.append(dict(family=family,outer=f,inner=j,excluded=proof['excluded']))
            if not proof['excluded']:stack.append(v['prediction']);used.append('inner%d'%j)
        refit[family]=stack[0];parts[family]=np.mean(stack,axis=0)
    w=sf['cnn_weight'];t=sf['fraction'];b=old['bilateral_STT'][te]
    models['CNN_bag'][te]=parts['CNN'];models['STT_bag'][te]=parts['STT']
    models['bagged'][te]=combine(parts['CNN'],parts['STT'],b,w,t)
    models['refit_reproduced'][te]=combine(refit['CNN'],refit['STT'],b,w,t)
reproduction=float(np.abs(models['refit_reproduced']-old['inner_selected_difference']).max());assert reproduction<1e-4,reproduction
y=old['y'];ids=old['person_id'];people=np.unique(ids);es={k:errors(y,v,ids) for k,v in models.items()}
rng=np.random.default_rng(20261031);idx=rng.integers(0,len(people),(10000,len(people)))
high=np.array([y[ids==i].mean()>=90 for i in people])
def paired(a,b):
    d=es[a]-es[b];boot=d[idx].mean(1)
    return dict(delta=float(d.mean()),better_people=int((d<0).sum()),worse_people=int((d>0).sum()),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
report=dict(mae={k:float(v.mean()) for k,v in es.items()},
    mae_bootstrap95={k:np.quantile(es[k][idx].mean(1),[.025,.975]).tolist() for k in ('inner_selected_difference','bagged')},
    fold_mae={k:[float(errors(y[old['fold']==f],models[k][old['fold']==f],ids[old['fold']==f]).mean()) for f in range(5)] for k in models},
    high90={k:dict(people=int(high.sum()),mae=float(es[k][high].mean())) for k in models},
    paired=dict(primary=paired('bagged','inner_selected_difference'),CNN=paired('CNN_bag','CNN'),STT=paired('STT_bag','STT')),
    members=members,refit_reproduction_max=reproduction,sources=sources,protocol_sha256=sha(H/'protocol.json'),optimizer_steps=0,protected_final_data_opened=False)
fb=sum(a<b for a,b in zip(report['fold_mae']['bagged'],report['fold_mae']['inner_selected_difference']))
pr=report['paired']['primary'];report['folds_improved']=int(fb)
report['success_rule_met']=bool(pr['delta']<0 and pr['bootstrap95'][1]<0 and fb>=4)
CACHE.mkdir(parents=True,exist_ok=True);dest=CACHE/'evaluation.npz';np.savez_compressed(dest,**models,**{k:old[k] for k in ('y','person_id','video_id','fold')})
report.update(predictions_path=str(dest),predictions_sha256=sha(dest))
save(O/'comparison.json',report)
print(json.dumps({k:report[k] for k in ('mae','mae_bootstrap95','paired','folds_improved','success_rule_met','refit_reproduction_max')},indent=1))
print(json.dumps({k:report['fold_mae'][k] for k in ('inner_selected_difference','bagged')}),json.dumps({k:report['high90'][k]['mae'] for k in ('inner_selected_difference','bagged')}))
