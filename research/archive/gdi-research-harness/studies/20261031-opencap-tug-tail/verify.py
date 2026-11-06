from common import *
import pandas as pd
import joblib

def main():
    d=load();y=d['y'];views=features(d);plan=json.loads((HERE/'splits.json').read_text())
    assert sha(HERE/'input.npz')==sha(PREV/'input.npz') and sha(HERE/'splits.json')==sha(PREV/'splits.json')
    for rep in plan:
        seen=[]
        for f in rep['splits']:
            tr=set(f['train']);te=set(f['test']);seen+=f['test'];assert not tr&te and tr|te==set(range(298))
            innerseen=[]
            for i in f['inner']:
                a=set(i['train']);b=set(i['validation']);innerseen+=i['validation'];assert not a&b and a|b==tr
            assert sorted(innerseen)==sorted(tr)
        assert sorted(seen)==list(range(298))
    # Independent order-statistic construction for the robust features.
    v=d['x'][:,17,:].astype(float);ordered=np.sort(v,axis=1);changes=np.sort(abs(v[:,1:]-v[:,:-1]),axis=1)
    manual=np.column_stack([ordered[:,45],(ordered[:,67]+ordered[:,68]-ordered[:,22]-ordered[:,23])/2,(changes[:,44]+changes[:,45])/2])
    assert np.allclose(manual,views['robust'],atol=1e-7,rtol=1e-6)
    completion=json.loads((HERE/'completion.json').read_text());assert completion['completed'] and completion['new_models']==425 and completion['reused_models']==75
    assert completion['code_sha256']==sha(HERE/'train.py')
    pred=pd.read_csv(HERE/'predictions.csv');assert len(pred)==29800
    method_map=METHODS.copy();new_count=425
    inner_rows=json.loads((HERE/'inner-evaluations.json').read_text())
    selections=json.loads((HERE/'selections.json').read_text());assert len(selections)==500
    if (HERE/'followup-completion.json').exists():
        followup=json.loads((HERE/'followup-completion.json').read_text());assert followup['completed'] and followup['new_models']==150
        assert followup['code_sha256']==sha(HERE/'followup.py') and followup['protocol_sha256']==sha(HERE/'followup-protocol.json')
        extra=pd.read_csv(HERE/'followup-predictions.csv');assert len(extra)==8940
        pred=pd.concat([pred,extra],ignore_index=True);method_map.update(FOLLOWUP_METHODS);new_count+=150
        inner_rows+=json.loads((HERE/'followup-inner-evaluations.json').read_text())
        extra_selection=json.loads((HERE/'followup-selections.json').read_text());assert len(extra_selection)==150;selections+=extra_selection
    assert not pred.duplicated(['repeat','seed','method','sample']).any()
    for _,g in pred.groupby(['repeat','seed','method']):
        assert sorted(g['sample'])==list(range(298));assert np.isfinite(g.prediction).all()
        assert np.array_equal(g.id,d['ids'][g['sample']]) and np.allclose(g.target,y[g['sample']],atol=1e-12)
    evaluations={}
    for r in inner_rows:
        f=plan[r['repeat']]['splits'][r['fold']];tr=np.array(f['train']);assert r['train_order']==tr.tolist()
        scores=[]
        for values,recorded in zip(r['inner_predictions'],r['scores']):
            error=abs(np.array(values)-y[tr]);masks=[y[tr]<10,(y[tr]>=10)&(y[tr]<20),y[tr]>=20]
            s={'MAE':float(error.mean()),'macro':float(np.mean([error[m].mean() for m in masks if m.any()]))}
            for k in s:assert np.isclose(s[k],recorded[k],atol=1e-12)
            scores.append(s)
        evaluations[(r['repeat'],r['fold'],r['seed'],r['view'],r['weight'],r['model'])]=(r['grid'],scores)
    maxdiff=0.
    for r in selections:
        view,weight,objective,kind=method_map[r['method']];f=plan[r['repeat']]['splits'][r['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=views[view]
        grid,scores=evaluations[(r['repeat'],r['fold'],r['seed'],view,weight,kind)]
        assert r['setting']==grid[int(np.argmin([s[objective] for s in scores]))]
        path=Path(r['model_path']) if r['reused_baseline'] else HERE/r['model_path'];assert sha(path)==r['model_sha256']
        scale,m=joblib.load(path)
        assert np.allclose(scale.mean_,x[tr].astype(float).mean(0),atol=1e-8) and np.allclose(scale.var_,x[tr].astype(float).var(0),atol=1e-8)
        if kind=='RF':
            assert m.random_state==r['seed'] and m.min_samples_leaf==r['setting'] and m.n_estimators==128
            assert np.array_equal(m._sample_weight,np.where(y[tr]>=20,weight,1.))
        else:assert m.alpha==r['setting']
        pp=m.predict(scale.transform(x[te]));pp=np.maximum(pp,0) if kind=='Ridge' else pp
        g=pred[(pred.repeat==r['repeat'])&(pred.seed==r['seed'])&(pred.method==r['method'])].set_index('sample')
        assert np.all(g.loc[te,'fold']==r['fold'])
        delta=float(np.max(abs(pp-g.loc[te,'prediction'].to_numpy())));assert delta<1e-8;maxdiff=max(maxdiff,delta)
    original=pd.read_csv(PREV/'cpu-predictions.csv');original=original[(original.model=='RF')&(original.view=='speed_summary')]
    base=pred[pred.method=='BaselineRF'].merge(original,on=['repeat','seed','sample'],suffixes=('','_old'),validate='one_to_one')
    assert len(base)==4470 and np.allclose(base.prediction,base.prediction_old,atol=1e-10,rtol=0)
    mechanisms=pd.read_csv(HERE/'baseline-tail-mechanisms.csv').set_index(['repeat','seed','sample'])
    for w in json.loads((HERE/'forest-training-label-weights.json').read_text()):
        tr=np.array(w['train_indices']);a=np.array(w['weights']);assert w['sample'] not in tr and (a>=0).all() and np.isclose(a.sum(),1)
        row=mechanisms.loc[(w['repeat'],w['seed'],w['sample'])]
        assert np.isclose(a@y[tr],row.prediction,atol=1e-10) and np.isclose(a[y[tr]>=20].sum(),row.long_label_weight_in_prediction,atol=1e-10)
    save('verification.json',{'passed':True,'new_models_replayed':new_count,'baseline_models_replayed':75,'maximum_prediction_delta':maxdiff,'same_previous_people_inputs_splits':True,'nested_objectives_recomputed':True,'training_weights_checked':True,'all_methods_predict_all_people':True,'baseline_exactly_reproduced':True,'forest_label_influence_reconstructed':True,'robust_features_independently_reconstructed':True})
    print('TAIL_MODEL_AND_MECHANISM_VERIFICATION_PASSED',maxdiff)

if __name__=='__main__':main()
