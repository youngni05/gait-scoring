"""Matched-record RF retraining: frozen folds/seeds, train-only selection."""
import os,sys,json,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;FEATURE=BASE/'feature-analysis';VIDEO=BASE/'video-stability/straight-followup'
sys.path.insert(0,str(FEATURE));import study
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load():
    d=study.load();short=np.load(VIDEO/'care-features.npz');ids=np.flatnonzero(short['length']>=50)
    return {k:v[ids] for k,v in d.items() if k in ['y','person','cohort','fold']},ids,dict(full=np.load(FEATURE/'inputs.npz')['motion3d'][ids],first2=short['first2'][ids],center2=short['center2'][ids])
def prepare():
    d,ids,x=load();assert len(ids)==2914 and len(set(d['person']))==110
    for a in x.values():assert a.shape==(2914,128) and np.isfinite(a).all()
    sources={str(p):sha(p) for p in [BASE/'data.npz',FEATURE/'inputs.npz',VIDEO/'care-features.npz',VIDEO/'care-predictions.npz',VIDEO/'movi-predictions.npz',VIDEO/'protocol.json',Path(__file__)]}
    save('protocol.json',dict(purpose='Separate short-input training mismatch from camera and population shift. No MoVi label assumptions.',
        variants=['full','first2','center2'],primary='first2',secondary='center2',control='full: refit full-walk features on identical eligible records',
        people=110,records=2914,excluded_short_records=39,frames=50,fps=25,features=128,
        seeds=study.SEEDS,folds=6,inner_validation='(outer_fold+1)%6',selection='Person-equal inner MAE; leaves2/8; tie lower leaf. Refit outer training only.',
        rf=dict(n_estimators=200,max_features=.5,bootstrap=True,criterion='squared_error',n_jobs=2),
        weighting='Each person same total fitting weight; no normalization or calibration. One input per source record in each variant, no sample inflation.',
        expected_inner_fits=180,expected_final_fits=90,scope='Same 14 MoVi people, 2sec straight and turn, paired cameras,56clips. No new HMR2 inference. Compare same-record OOF CARE error and same-clip camera sensitivity. No clinical cutoff optimized.',code_sha256=sha(Path(__file__)),sources=sources))
    np.savez_compressed(HERE/'input.npz',**d,indices=ids,**x)
    print('PREPARED',len(ids),len(set(d['person'])),flush=True)
def make(seed,leaf):return RandomForestRegressor(n_estimators=200,max_features=.5,min_samples_leaf=leaf,random_state=seed,n_jobs=2)
def train():
    p=json.loads((HERE/'protocol.json').read_text());assert sha(Path(__file__))==p['code_sha256']
    root=Path(os.environ.get('RF2_OUTPUT',str(HERE)));root.mkdir(exist_ok=True)
    names=[sys.argv[2]] if len(sys.argv)>2 else p['variants'];folds=[int(sys.argv[3])] if len(sys.argv)>3 else list(range(6));seeds=p['seeds'][:1] if os.environ.get('RF2_PILOT') else p['seeds'];tag='-'.join(names)+'-'+'-'.join(map(str,folds))
    z=np.load(HERE/'input.npz');d={k:z[k] for k in ['y','person','cohort','fold','indices']};start=time.time();completed=0
    for name in names:
        x=z[name]
        for f in folds:
            tr=np.flatnonzero((d['fold']!=f)&(d['fold']!=(f+1)%6));va=np.flatnonzero(d['fold']==(f+1)%6);outer=np.flatnonzero(d['fold']!=f);te=np.flatnonzero(d['fold']==f)
            assert not set(d['person'][outer])&set(d['person'][te]) and not set(d['person'][tr])&set(d['person'][va])
            for seed in seeds:
                folder=root/'fits'/name;folder.mkdir(parents=True,exist_ok=True);target=folder/('fold%d_seed%d'% (f,seed))
                if target.with_suffix('.json').exists():completed+=1;continue
                choices=[];inner=[]
                for leaf in [2,8]:
                    m=make(seed,leaf);m.fit(x[tr],d['y'][tr],sample_weight=study.weights(d['person'][tr]));pred=m.predict(x[va]);choices.append([study.metrics(d['y'][va],pred,d['person'][va])['MAE'],leaf]);inner.append(pred)
                leaf=min(choices,key=lambda r:r[0])[1];m=make(seed,leaf);m.fit(x[outer],d['y'][outer],sample_weight=study.weights(d['person'][outer]));pred=m.predict(x[te]);joblib.dump(m,target.with_suffix('.joblib'))
                np.savez_compressed(target.with_suffix('.npz'),test=te,outer=outer,inner_train=tr,validation=va,inner_predictions=np.array(inner),choices=choices,selected_leaf=leaf,prediction=pred)
                replay=joblib.load(target.with_suffix('.joblib')).predict(x[te]);error=float(abs(replay-pred).max());assert error<=1e-12
                marker=dict(variant=name,fold=f,seed=seed,replay_max_error=error,model_sha256=sha(target.with_suffix('.joblib')),prediction_sha256=sha(target.with_suffix('.npz')),elapsed_seconds=time.time()-start)
                target.with_suffix('.json').write_text(json.dumps(marker,indent=2));completed+=1;(root/('progress-'+tag+'.json')).write_text(json.dumps(dict(completed=completed,expected=len(names)*len(folds)*len(seeds),last=marker)));print(name,f,seed,round(time.time()-start,1),flush=True)
    expected=len(names)*len(folds)*len(seeds);assert completed==expected;(root/('complete-'+tag+'.json')).write_text(json.dumps(dict(passed=True,final_models=expected,inner_fits=2*expected,elapsed_seconds=time.time()-start,script_sha256=sha(Path(__file__)),input_sha256=sha(HERE/'input.npz'),protocol_sha256=sha(HERE/'protocol.json'))))
if __name__=='__main__':{'prepare':prepare,'train':train}[sys.argv[1]]()
