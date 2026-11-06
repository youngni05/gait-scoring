"""Controlled calculation comparison; fixed historical estimator, no test tuning."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,hashlib
import numpy as np,joblib
HERE=Path(__file__).resolve().parent;BIG=Path('D:/gait-research-data/D_P/harmonized-gait-video-validation-20261104')
sys.path.insert(0,str(HERE.parent/'20261104-planar-tug-improvement'))
import sensor_loop as s
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):s.save(p,x)
def data():
    fd,FX,sets=s.load();gd=np.load(s.HEAVY/'gstride/short-window-input.npz')
    ff={r['id']:r for r in json.loads((BIG/'F-features.json').read_text()) if r['status']=='ok'}
    gg={r['id']:r for r in json.loads((BIG/'G-features.json').read_text())}
    fi=np.array([i for i,pid in enumerate(fd['ids']) if int(pid) in ff]);gi=np.array([i for i,pid in enumerate(gd['ids']) if str(pid) in gg])
    ids=np.r_[[f'F{int(pid)}' for pid in fd['ids'][fi]],gd['ids'][gi]]
    old=np.concatenate([FX['core3'][fi],gd['X'][gi]])
    new=np.array([ff[int(pid)]['X'] for pid in fd['ids'][fi]]+[gg[str(pid)]['X'] for pid in gd['ids'][gi]])
    return dict(ids=ids,y=np.r_[fd['y'][fi],gd['y'][gi]],fold=np.r_[fd['fold'][fi],gd['fold'][gi]],
      cohort=np.r_[['F']*len(fi),['G']*len(gi)],old=old,new=new,
      F_excluded_ids=fd['ids'][[i for i in range(len(fd['ids'])) if i not in fi]],
      F_excluded_y=fd['y'][[i for i in range(len(fd['ids'])) if i not in fi]])
def freeze():
    d=data();assert not(HERE/'comparison-protocol.json').exists()
    save(HERE/'comparison-protocol.json',dict(code_sha256=sha(__file__),helper_sha256=sha(Path(s.__file__)),
      feature_protocol_sha256=sha(HERE/'harmonization-protocol.json'),feature_hashes={str(BIG/f):sha(BIG/f) for f in ['F-features.json','G-features.json']},
      estimator='Historical ExtraTrees128 leaf10 seed20261104, one CPU; train-only median+standardization,shared inverse-speed; no tuning/innerselection/stacking. Fixed before fitting this batch.',
      modes=['F_only','G_only','pooled','F_to_G_zero_adaptation'],views=['old_speed','old_core3','new_speed','new_core3'],
      matched_people={'F':int(sum(d['cohort']=='F')),'G':int(sum(d['cohort']=='G'))},
      splitting='Unchanged original F and G fold assignments,subset without reseeding. Pooled fold excludes its same-index heldout from both cohorts. All external G predictions mean fiveFtrainedmodels,noGlabels in thosemodels.',
      evaluation='Samepeople before/after and speed/core3; pairedbootstrap5000person95CI,MAE/bias/p95/TUG>=20. All outputs retained. Development diagnostic,notconfirmatory evidence; not comparable directly to historical12candidate stacks on594.'
    ))
def train():
    p=json.loads((HERE/'comparison-protocol.json').read_text());assert p['code_sha256']==sha(__file__) and p['helper_sha256']==sha(Path(s.__file__))
    for f,h in p['feature_hashes'].items():assert sha(f)==h
    d=data();folder=BIG/'models';folder.mkdir(exist_ok=True);np.savez_compressed(BIG/'input.npz',**d)
    records=[]
    for mode in ['F_only','G_only','pooled']:
      include=d['cohort']==('F' if mode=='F_only' else 'G') if mode!='pooled' else np.ones(len(d['y']),bool)
      for version in ['old','new']:
       for view in ['speed','core3']:
        X=d[version][:,:1] if view=='speed' else d[version]
        for f in range(5):
          tr=np.flatnonzero(include&(d['fold']!=f));te=np.flatnonzero(include&(d['fold']==f))
          assert not set(d['ids'][tr])&set(d['ids'][te])
          m=s.fit(X[tr],d['y'][tr],{'kind':'trees','leaf':10},s.feature_names()[view]);pred=s.predict(m,X[te])
          tag=f'{mode}-{version}-{view}-{f}';path=folder/(tag+'.joblib');joblib.dump(m,path,compress=3)
          context=folder/(tag+'.npz');np.savez_compressed(context,train=tr,test=te,pred=pred)
          records.append(dict(mode=mode,version=version,view=view,fold=f,model=str(path),model_sha256=sha(path),context=str(context),context_sha256=sha(context)))
        print('FITTED',mode,version,view,flush=True)
    save(HERE/'fit-receipt.json',dict(code_sha256=sha(__file__),input_sha256=sha(BIG/'input.npz'),models=records))
    external={}
    for version in ['old','new']:
     for view in ['speed','core3']:
      X=d[version][:,:1] if view=='speed' else d[version];te=np.flatnonzero(d['cohort']=='G')
      pp=[s.predict(joblib.load(r['model']),X[te]) for r in records if r['mode']=='F_only' and r['version']==version and r['view']==view]
      assert len(pp)==5;external[version+'_'+view]=np.mean(pp,axis=0)
    np.savez_compressed(BIG/'external.npz',**external)
    print('COMPARISON_COMPLETE',len(records),'models',flush=True)
if __name__=='__main__':{'freeze':freeze,'train':train}[sys.argv[1]]()
