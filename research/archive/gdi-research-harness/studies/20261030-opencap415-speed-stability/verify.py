"""Audit frozen cohorts/splits, reproduce saved predictions and nested selections."""
from common import *
import ast
import importlib.util
import pandas as pd
import joblib

def prepare_checks():
    d=load();source=dict(np.load(SOURCE/'all3s/input.npz'))
    for key in source:assert np.array_equal(d[key],source[key])
    prep=json.loads((HERE/'preparation.json').read_text())
    assert sha(HERE/'splits.json')==prep['splits_sha256']
    splits=json.loads((HERE/'splits.json').read_text());assert len(splits)==5
    assert splits[0]['splits']==json.loads((SOURCE/'all3s/splits.json').read_text())
    signatures=[]
    for rep in splits:
        seen=[];signature=np.empty(len(d['ids']),dtype=int)
        for f in rep['splits']:
            tr=set(f['train']);te=set(f['test']);seen+=f['test'];signature[f['test']]=f['fold']
            assert not tr&te and tr|te==set(range(298))
            assert not set(d['ids'][list(tr)])&set(d['ids'][list(te)])
            innerseen=[]
            for i in f['inner']:
                a=set(i['train']);b=set(i['validation']);innerseen+=i['validation']
                assert not a&b and a|b==tr and not (a|b)&te
            assert sorted(innerseen)==sorted(tr)
        assert sorted(seen)==list(range(298));signatures.append(tuple(signature))
    assert len(set(signatures))==5
    for name in json.loads((HERE/'cnn-runs.json').read_text()):
        p=HERE/'cnn'/name;dd=dict(np.load(p/'input.npz'));config=json.loads((p/'run-config.json').read_text())
        for key in ['x','y','ids','frames']:assert np.array_equal(d[key],dd[key])
        assert json.loads((p/'splits.json').read_text())==splits[config['repeat']]['splits']
        for f in splits[config['repeat']]['splits']:assert np.all(dd['folds'][f['test']]==f['fold'])
        audit=json.loads((p/'input-audit.json').read_text())
        assert sha(p/'input.npz')==audit['input_sha256'] and sha(p/'protocol.json')==audit['protocol_sha256']
        ast.parse((p/'gpu.py').read_text())
    # Independently form the 57 statistics, then check every view.
    x=d['x'];manual=[]
    for stat in ['mean','std','change']:
        for channel in range(19):
            values=x[:,channel,:]
            manual.append(values.mean(1) if stat=='mean' else values.std(1) if stat=='std' else np.abs(values[:,1:]-values[:,:-1]).mean(1))
    full=np.stack(manual,axis=1);views=features(x)
    for name,ix in [('mean_speed',[17]),('speed_summary',[17,36,55]),('pelvis_motion',[17,18,36,37,55,56]),('full',list(range(57)))]:
        assert np.array_equal(views[name],full[:,ix])
    return d,splits,views

def main(mode):
    d,plan,views=prepare_checks();y=d['y'];audit={'prepared_inputs_and_views':True,'all_people_separated':True,'original_split_preserved':True,'different_partitions':5}
    if mode=='prepare':
        save('preparation-verification.json',audit);print(json.dumps(audit));return
    complete=json.loads((HERE/'cpu-completion.json').read_text());assert complete['completed'] and complete['models']==400
    assert complete['code_sha256']==sha(HERE/'cpu.py')
    pred=pd.read_csv(HERE/'cpu-predictions.csv');selections=json.loads((HERE/'cpu-selections.json').read_text());assert len(selections)==400
    assert len(pred)==26820 and not pred.duplicated(['repeat','seed','view','model','sample']).any()
    for keys,g in pred.groupby(['repeat','seed','view','model']):
        assert sorted(g['sample'])==list(range(298)) and np.isfinite(g.prediction).all()
        assert np.array_equal(g.id.to_numpy(),d['ids'][g['sample']]) and np.array_equal(g.target.to_numpy(),y[g['sample']])
    maxdiff=0.
    for r in selections:
        f=plan[r['repeat']]['splits'][r['fold']];tr=np.array(f['train']);te=np.array(f['test']);x=views[r['view']]
        assert r['inner_train_order']==tr.tolist()
        scores=[float(np.mean(abs(y[tr]-np.array(p)))) for p in r['inner_predictions']]
        assert np.allclose(scores,r['scores'],rtol=0,atol=1e-12)
        assert r['selected']==r['grid'][int(np.argmin(scores))]
        scale,model=joblib.load(HERE/r['model_path'])
        assert np.allclose(scale.mean_,x[tr].astype(float).mean(0),atol=1e-8)
        assert np.allclose(scale.var_,x[tr].astype(float).var(0),atol=1e-8)
        if r['model']=='RF':assert model.random_state==r['seed'] and model.min_samples_leaf==r['selected']
        else:assert model.alpha==r['selected']
        g=pred[(pred.repeat==r['repeat'])&(pred.seed==r['seed'])&(pred.view==r['view'])&(pred.model==r['model'])].set_index('sample')
        assert np.all(g.loc[te,'fold']==r['fold'])
        err=np.max(abs(model.predict(scale.transform(x[te]))-g.loc[te,'prediction'].to_numpy()));maxdiff=max(maxdiff,float(err));assert err<1e-8
    for rep in plan:
        for f in rep['splits']:
            tr=np.array(f['train']);te=f['test']
            for name,value in [('Mean',y[tr].mean()),('Median',np.sort(y[tr])[(len(tr)-1)//2])]:
                g=pred[(pred.repeat==rep['repeat'])&(pred.model==name)].set_index('sample');assert np.allclose(g.loc[te,'prediction'],value,atol=1e-12)
    original=pd.read_csv(SOURCE/'all3s/cpu-predictions.csv').sort_values('sample')
    for model,seed in [('RF',20261030),('Ridge',-1)]:
        g=pred[(pred.repeat==0)&(pred.seed==seed)&(pred.view=='full')&(pred.model==model)].sort_values('sample')
        assert np.allclose(g.prediction,original[model],atol=1e-5,rtol=0)
    audit.update(CPU_models_replayed=400,CPU_max_prediction_delta=maxdiff,original_RF_Ridge_reproduced=True,nested_selection_recalculated=True)
    if mode=='all':
        completion=json.loads((HERE/'cnn-completion.json').read_text())
        assert completion['completed'] and len(completion['runs'])==15 and completion['device']=='cpu'
        for run in completion['runs']:
            assert run['code_sha256']==sha(HERE/'cnn'/run['run']/'cnn_cpu.py')
        sys.path.insert(0,str(ROOT/'outputs/motionbert-smoke-20261015/deps'));import torch
        torch.set_num_threads(2);nmodels=0;gpumax=0.;allgpu=[]
        for name in json.loads((HERE/'cnn-runs.json').read_text()):
            p=HERE/'cnn'/name;config=json.loads((p/'run-config.json').read_text());runplan=plan[config['repeat']]['splits']
            summary=json.loads((p/'gpu-summary.json').read_text());assert summary['folds_finished']==5 and summary['models_finished']==1
            assert summary['environment']['device']=='cpu'
            original_ast=ast.parse((p/'gpu.py').read_text());cpu_ast=ast.parse((p/'cnn_cpu.py').read_text())
            assert ast.dump(next(n for n in original_ast.body if isinstance(n,ast.ClassDef) and n.name=='Net'))==ast.dump(next(n for n in cpu_ast.body if isinstance(n,ast.ClassDef) and n.name=='Net'))
            gp=pd.read_csv(p/'gpu-predictions.csv').sort_values('sample').reset_index(drop=True)
            assert gp['sample'].tolist()==list(range(298)) and np.array_equal(gp.target.to_numpy(),y)
            sys.path.insert(0,str(p));saved_common=sys.modules.pop('common',None)
            spec=importlib.util.spec_from_file_location('cnn_for_verification',p/'gpu.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            for f in runplan:
                ck=torch.load(p/f'runs/main/CNN/fold{f["fold"]}-refit/model.pt',map_location='cpu',weights_only=False)
                tr=np.array(f['train']);te=np.array(f['test']);n=ck['normalization'];x=d['x'].astype(float)
                assert ck['train_indices']==tr.tolist() and ck['validation_indices'] is None and ck['input_sha256']==sha(p/'input.npz')
                assert np.allclose(n['xmean'],x[tr].mean((0,2))[None,:,None])
                assert np.allclose(n['xstd'],np.maximum(x[tr].std((0,2))[None,:,None],.001))
                assert np.isclose(n['ymean'],y[tr].mean()) and np.isclose(n['ystd'],max(y[tr].std(),.1))
                ff=np.log1p(d['frames']).astype(float)[:,None]
                assert np.allclose(n['fmean'],ff[tr].mean(0)) and np.allclose(n['fstd'],np.maximum(ff[tr].std(0),.001))
                model=module.Net('CNN');model.load_state_dict(ck['state']);model.eval()
                with torch.no_grad():
                    pp=model(torch.tensor((x[te]-n['xmean'])/n['xstd'],dtype=torch.float32),torch.tensor((ff[te]-n['fmean'])/n['fstd'],dtype=torch.float32)).numpy()*n['ystd']+n['ymean']
                err=float(np.max(abs(pp-gp.loc[te,'prediction'].to_numpy())));assert err<1e-4;gpumax=max(gpumax,err);nmodels+=1
                selection=json.loads((p/'gpu-selection.json').read_text())[f['fold']]
                assert selection['refit_epochs']==int(np.median(selection['inner_best_epochs']))==ck['epoch']
                for i,inner in enumerate(f['inner']):
                    history=json.loads((p/f'runs/main/CNN/fold{f["fold"]}-inner{i}/history.json').read_text())
                    best=min(history,key=lambda h:h['validation_person_MAE'])['epoch']
                    assert best==selection['inner_best_epochs'][i]
            sys.path.remove(str(p));sys.modules.pop('common',None)
            if saved_common is not None:sys.modules['common']=saved_common
            err=gp.prediction.to_numpy()-y
            assert np.isclose(abs(err).mean(),summary['metrics']['CNN']['person_MAE'])
            gp['repeat']=config['repeat'];gp['seed']=config['model_seed'];gp['view']='full';gp['id']=d['ids'];allgpu.append(gp)
        assert nmodels==75
        pd.concat(allgpu).to_csv(HERE/'cnn-predictions.csv',index=False)
        audit.update(CNN_models_replayed=nmodels,CNN_training_device='cpu',CNN_max_prediction_delta=gpumax)
    audit['passed']=True;save('verification.json',audit);print(json.dumps(audit))

if __name__=='__main__':main(sys.argv[1])
