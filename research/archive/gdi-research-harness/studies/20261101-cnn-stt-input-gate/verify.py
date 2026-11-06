"""Independent metric, gate reconstruction and leakage/symmetry checks."""
import copy
from study import *

def independent_prediction(model,d,cfg):
    outputs=[]
    for i in range(len(d['x'])):
        x=[1.]+[max(-cfg['feature_clip'],min(cfg['feature_clip'],(float(a)-b)/c))
                  for a,b,c in zip(d['x'][i],model['mean'],model['sd'])]
        g=[1/(1+np.exp(-(sum(a*b for a,b in zip(x,beta))+model['offset']))) for beta in model['beta']]
        c,s=d['cnn'][i],d['stt'][i]
        if model['family']=='shared': p=[g[0]*c[k]+(1-g[0])*s[k] for k in range(2)]
        else:
            mean=g[0]*sum(c)/2+(1-g[0])*sum(s)/2
            diff=g[1]*(c[0]-c[1])+(1-g[1])*(s[0]-s[1])
            p=[mean+diff/2,mean-diff/2]
        outputs.append(p)
    return np.array(outputs)

def independent_metrics(y,p,ids):
    records=[]
    for person in sorted(set(ids.tolist())):
        ix=[i for i,q in enumerate(ids) if q==person];n=len(ix)
        records.append(dict(high=sum(float(y[i].sum()) for i in ix)/(2*n)>=90,
            mae=sum(abs(float(p[i,k]-y[i,k])) for i in ix for k in range(2))/(2*n),
            mean_mae=sum(abs(float(p[i].mean()-y[i].mean())) for i in ix)/n,
            side_difference_mae=sum(abs(float(p[i,0]-p[i,1]-y[i,0]+y[i,1])) for i in ix)/n,
            bias=sum(float(p[i,k]-y[i,k]) for i in ix for k in range(2))/(2*n),
            underprediction=sum(max(float(y[i,k]-p[i,k]),0) for i in ix for k in range(2))/(2*n)))
    result={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='high'}
    high=[r for r in records if r['high']]
    result['high90']={k:sum(r[k] for r in high)/len(high) for k in records[0] if k!='high'}
    result['high90']['people']=len(high)
    return result

def main():
    start=time.perf_counter();cfg=read(H/'protocol.json');tr=read(O/'training.json');report=read(O/'comparison.json')
    assert tr['protocol_sha256']==report['protocol_sha256']==sha(H/'protocol.json')
    assert report['training_sha256']==sha(O/'training.json')
    for p,digest in {**tr['source_hashes'],**report['source_hashes']}.items(): assert sha(p)==digest,p
    assert report['predictions_sha256']==sha(O/'predictions.npz')
    assert report['path_predictions_sha256']==sha(O/'path-predictions.npz')
    z=arrays(O/'predictions.npz');pp=arrays(O/'path-predictions.npz')
    metric_error=0.;prediction_error=0.;reconstruction_error=0.;selection_refits=0;outer_labels_unchanged=True
    audit_refits=0;audit_error=0.
    for name in report['metrics']:
        check=independent_metrics(z['y'],z[name],z['person_id'])
        for m in ['mae','mean_mae','side_difference_mae','bias','underprediction']:
            metric_error=max(metric_error,abs(check[m]-report['metrics'][name][m]),
                             abs(check['high90'][m]-report['metrics'][name]['high90'][m]))
        assert check['high90']['people']==68
    for e in tr['paths']:
        f,j=e['outer'],e['inner']; d=inner_data(f,j,{})
        boundary(e['base_people'],*e['split_people'],e['outer_people'])
        assert sorted(sum(e['split_people'],[]))==e['meta_people']
        assert e['meta_people']==np.unique(d['ids']).tolist()
        masks=[np.isin(d['ids'],g) for g in e['split_people']];a,b,c=[subset(d,m) for m in masks]
        refit=subset(d,masks[0]|masks[1]);audit_predictions={}
        aw=fixed_weight(refit);audit_predictions['constant']=aw*c['cnn']+(1-aw)*c['stt']
        choices=[];cw=fixed_weight(a)
        choices.append(dict(family='constant',lambda_=None,selection_mae=mae(b['y'],cw*b['cnn']+(1-cw)*b['stt'],b['ids'])))
        for family in cfg['families']:
            options=[]
            for lam in cfg['l2_grid']:
                model,fun=fit(a,family,lam,cfg);selection_refits+=1
                p,_=predict(model,b,cfg)
                r=dict(family=family,lambda_=lam,selection_mae=mae(b['y'],p,b['ids']))
                choices.append(r);options.append(r)
                # Central finite differences check the actual smooth-MAE gate gradients.
                if f==0 and j==1 and lam==cfg['l2_grid'][0]:
                    t=np.array(model['beta']).ravel()+np.linspace(-.03,.03,len(np.array(model['beta']).ravel()))
                    val,grad=fun(t)
                    for index in [0,len(t)//2,len(t)-1]:
                        hi=t.copy();lo=t.copy();hi[index]+=1e-5;lo[index]-=1e-5
                        numerical=(fun(hi)[0]-fun(lo)[0])/2e-5
                        assert abs(numerical-grad[index])<1e-6,(family,index,numerical,grad[index])
            best=min(options,key=lambda q:(q['selection_mae'],-q['lambda_']))
            audit_model,_=fit(refit,family,best['lambda_'],cfg);audit_refits+=1
            audit_predictions[family]=independent_prediction(audit_model,c,cfg)
            final,_=fit(d,family,best['lambda_'],cfg);selection_refits+=1
            np.testing.assert_allclose(final['beta'],e['models'][family]['beta'],rtol=0,atol=1e-10)
        for r,stored in zip(choices,e['candidates']):
            assert r['family']==stored['family'] and r['lambda_']==stored['lambda_']
            assert abs(r['selection_mae']-stored['selection_mae'])<1e-10
        chosen=min(choices,key=lambda r:(r['selection_mae'],r['family']!='constant',-(r['lambda_'] or 0)))
        assert chosen==e['selected']
        audit_predictions['selected']=audit_predictions[chosen['family']]
        saved_audit=next(r for r in tr['audit'] if r['outer']==f and r['inner']==j)
        for family,p in audit_predictions.items():
            independent=[]
            for person in sorted(set(c['ids'].tolist())):
                ix=np.flatnonzero(c['ids']==person)
                independent.append(sum(abs(float(p[i,k]-c['y'][i,k])) for i in ix for k in range(2))/(2*len(ix)))
            audit_error=max(audit_error,float(np.max(abs(np.array(independent)-saved_audit['errors'][family]))))
        assert abs(fixed_weight(d)-e['fixed_weight'])<1e-12
        od=outer_data(e,{});fixed=e['fixed_weight']*od['cnn']+(1-e['fixed_weight'])*od['stt']
        np.testing.assert_allclose(fixed,pp[f'f{f}_j{j}_matched_fixed'],rtol=0,atol=1e-12)
        for family in cfg['families']:
            model=e['models'][family];p,g=predict(model,od,cfg);q=independent_prediction(model,od,cfg)
            prediction_error=max(prediction_error,float(abs(p-q).max()))
            np.testing.assert_allclose(p,pp[f'f{f}_j{j}_{family}'],rtol=0,atol=1e-12)
            poison=copy.deepcopy(od);poison['y']=np.full_like(p,999999.)
            np.testing.assert_array_equal(p,predict(model,poison,cfg)[0])
            swapped=copy.deepcopy(od);swapped['cnn']=od['cnn'][:,::-1];swapped['stt']=od['stt'][:,::-1]
            np.testing.assert_allclose(p[:,::-1],predict(model,swapped,cfg)[0],rtol=0,atol=1e-12)
        family=chosen['family'];expected=fixed if family=='constant' else pp[f'f{f}_j{j}_{family}']
        np.testing.assert_array_equal(expected,pp[f'f{f}_j{j}_selected'])
    for f in range(5):
        rows=np.flatnonzero(z['fold']==f)
        for family in ['matched_fixed','shared','mean_difference','selected']:
            p=sum(pp[f'f{f}_j{j}_{family}'] for j in range(5) if j!=f)/4
            reconstruction_error=max(reconstruction_error,float(abs(p-z[family][rows]).max()))
    # Duplicating records within one person must not change the overall person-weighted metric.
    y=np.array([[10.,20.],[30.,50.]]);p=np.array([[15.,22.],[40.,48.]]);ids=np.array(['a','b'])
    assert abs(mae(y,p,ids)-mae(y[[0,0,0,1]],p[[0,0,0,1]],ids[[0,0,0,1]]))<1e-12
    try: boundary(['a'],['a'])
    except ValueError: pass
    else: raise AssertionError('Leaked person accepted')
    assert metric_error<1e-10 and prediction_error<1e-10 and reconstruction_error<1e-12 and audit_error<1e-10
    save(O/'verification.json',dict(passed=True,paths=20,selection_and_final_refits=selection_refits,
        metric_max_error=metric_error,independent_prediction_max_error=prediction_error,
        path_average_max_error=reconstruction_error,gradient_checked=True,all_source_hashes_unchanged=True,
        audit_refits=audit_refits,audit_metric_max_error=audit_error,
        base_meta_outer_people_disjoint=True,poisoned_outer_labels_invariant=outer_labels_unchanged,
        side_swap_equivariant=True,person_weighting_checked=True,leaked_person_rejected=True,
        comparison_sha256=sha(O/'comparison.json'),study_code_sha256=sha(H/'study.py'),
        seconds=time.perf_counter()-start))
    print(json.dumps(read(O/'verification.json'),indent=2),flush=True)

if __name__=='__main__': main()
