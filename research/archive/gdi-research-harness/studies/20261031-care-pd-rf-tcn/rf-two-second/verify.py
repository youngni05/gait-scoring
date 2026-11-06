"""Independent reconstruction of splits, selection, fit weights and metrics."""
import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;VIDEO=BASE/'video-stability/straight-followup'
sys.path.insert(0,str(BASE/'feature-analysis'));import study
import numpy as np
import joblib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def personmean(values,person):return float(np.mean([np.mean(values[person==s]) for s in np.unique(person)]))
def main():
    collection=json.loads((HERE/'collection-verification.json').read_text());assert collection['passed']
    for name,digest in collection['files'].items():assert sha(HERE/'received'/name)==digest
    p=json.loads((HERE/'protocol.json').read_text());assert p['seeds']==study.SEEDS and p['expected_final_fits']==90
    for path,digest in p['sources'].items():assert sha(Path(path))==digest
    complete=json.loads((HERE/'received/main/complete.json').read_text());assert complete['passed'] and complete['final_models']==90 and complete['inner_fits']==180
    for name,key in [('experiment.py','script_sha256'),('cpu_runner.py','runner_sha256'),('input.npz','input_sha256'),('protocol.json','protocol_sha256')]:assert sha(HERE/name)==complete[key]
    with np.load(HERE/'input.npz') as z:d={k:z[k] for k in z.files}
    with np.load(HERE/'predictions.npz') as z:pred={k:z[k] for k in z.files}
    with np.load(VIDEO/'movi-predictions.npz') as z:movi={k:z[k] for k in z.files}
    report=json.loads((HERE/'analysis.json').read_text());source=np.load(BASE/'data.npz');original=np.load(VIDEO/'care-predictions.npz');short=np.load(VIDEO/'care-features.npz');assert np.array_equal(d['indices'],np.flatnonzero(short['length']>=50))
    for key in ['y','person','cohort','fold']:assert np.array_equal(d[key],source[key][d['indices']])
    assert len(d['y'])==2914 and len(np.unique(d['person']))==110
    splitchecks=0;replay=[];selection=[];roots=[]
    for name in ['full','first2','center2']:
        assert np.array_equal(d[name],(np.load(BASE/'feature-analysis/inputs.npz')['motion3d'] if name=='full' else short[name])[d['indices']])
        for f in range(6):
            outer=np.where(d['fold']!=f)[0];test=np.where(d['fold']==f)[0];train=np.where((d['fold']!=f)&(d['fold']!=(f+1)%6))[0];valid=np.where(d['fold']==(f+1)%6)[0]
            assert set(d['person'][outer]).isdisjoint(d['person'][test]) and set(d['person'][train]).isdisjoint(d['person'][valid])
            for j,seed in enumerate(study.SEEDS):
                target=HERE/'received/main/fits'/name/('fold%d_seed%d'%(f,seed));a=np.load(target.with_suffix('.npz'));marker=json.loads(target.with_suffix('.json').read_text());model=joblib.load(target.with_suffix('.joblib'));model.n_jobs=1
                assert sha(target.with_suffix('.joblib'))==marker['model_sha256'] and sha(target.with_suffix('.npz'))==marker['prediction_sha256']
                for key,expected in [('test',test),('outer',outer),('inner_train',train),('validation',valid)]:assert np.array_equal(a[key],expected)
                losses=[personmean(abs(v-d['y'][valid]),d['person'][valid]) for v in a['inner_predictions']];selection.append(float(abs(np.array(losses)-a['choices'][:,0]).max()));chosen=[2,8][int(np.argmin(losses))]
                assert chosen==a['selected_leaf']==model.min_samples_leaf
                assert model.random_state==seed and model.n_estimators==200 and model.max_features==.5 and model.bootstrap and model.criterion=='squared_error'
                vals=model.predict(d[name][test]);replay.append(float(abs(vals-a['prediction']).max()));replay.append(float(abs(vals-pred['new_'+name][j,test]).max()))
                replay.append(float(abs(model.predict(movi['features'])-pred['movi_new_'+name][f*5+j]).max()))
                person=d['person'][outer];_,inv,cnt=np.unique(person,return_inverse=True,return_counts=True);weights=1/cnt[inv];weights/=weights.mean();tree=model.estimators_[0];bootstrap=np.random.RandomState(tree.random_state).randint(0,len(outer),len(outer));bw=np.bincount(bootstrap,minlength=len(outer))*weights;expected=np.sum(bw*d['y'][outer])/bw.sum();roots.append(abs(float(tree.tree_.value[0,0,0])-expected));splitchecks+=1
    errors=[]
    for name,groups in report['clinical'].items():
        for group,mask in [('all',np.ones(2914,bool)),('high',d['y']>=2),('score3',d['y']==3)]:
            observed=[]
            for j in range(5):
                q=pred[name][j,mask];y=d['y'][mask];person=d['person'][mask];e=q-y;v=[personmean(abs(e),person),personmean(e,person),personmean(abs(e)>=2,person),personmean((q>=.5)&(q<1.5),person),personmean(np.floor(q+.5)==y,person)];observed.append(v);expected=[groups[group]['seeds'][j][k] for k in ['MAE','bias','large_error','score1_fraction','rounded_accuracy']];errors.append(float(np.max(abs(np.array(v)-expected))))
            expected=[groups[group][k] for k in ['MAE','bias','large_error','score1_fraction','rounded_accuracy']];errors.append(float(np.max(abs(np.mean(observed,axis=0)-expected))))
            assert groups[group]['people']==len(np.unique(d['person'][mask])) and groups[group]['records']==int(mask.sum())
    for name,groups in report['movi'].items():
        for condition,entry in groups.items():
            a=pred[name+'_'+condition];assert a.shape==(14,2,30);delta=a[:,1]-a[:,0];errors.extend([abs(float(abs(delta).mean())-entry['view_abs']),abs(float(delta.mean())-entry['view_bias']),abs(float(((a>=.5)&(a<1.5)).mean())-entry['score1_fraction'])])
            for j in range(5):errors.append(abs(float(abs(delta[:,j::5]).mean())-entry['by_seed'][j]['view_abs']))
    assert splitchecks==90 and max(replay)<=1e-12 and max(selection)<=1e-12 and max(roots)<=1e-10 and max(errors)<=1e-12
    result=dict(passed=True,models=90,inner_fits=180,people=110,records=2914,seeds=study.SEEDS,outer_folds=6,movi_people=14,selected_leaves_verified=90,replay_max_error=max(replay),inner_metric_max_error=max(selection),root_weight_max_error=max(roots),independent_metric_max_error=max(errors),analysis_sha256=sha(HERE/'analysis.json'),collection_sha256=sha(HERE/'collection-verification.json'),verifier_sha256=sha(Path(__file__)))
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
