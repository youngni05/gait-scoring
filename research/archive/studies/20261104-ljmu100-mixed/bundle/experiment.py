"""Frozen original training functions, with audited mixed inputs and separate-source evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser()
ap.add_argument('action',choices=['check','smoke','pilot','train','baseline'])
ap.add_argument('--family',choices=['CNN','STT','bilateral'],required=True)
ap.add_argument('--data-root',type=Path,required=True)
ap.add_argument('--folds',default='0,1,2,3,4')
ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
a=ap.parse_args()
sys.path[:0]=[str(HERE/a.family),str(HERE/a.family/'vendor')]
spec=importlib.util.spec_from_file_location('original_training',HERE/a.family/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
torch=r.torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):r.save(p,v)
def load_data():
    seal=read(HERE/'manifest.json')
    for name,digest in seal.items():assert sha(HERE/name)==digest,name
    protocol=read(HERE/'study-protocol.json')
    assert sha(a.data_root/'data.npz')==protocol['data_sha256']
    with np.load(a.data_root/'data.npz') as z:d={k:z[k].copy() for k in z.files}
    x=d.pop('x')
    if a.family!='CNN':x=np.ascontiguousarray(x[...,:8].reshape(-1,2,124,4,2))
    d['x']=r.WindowSet(x,d.pop('window_record'),len(d['y']),d.pop('window_start'),d.pop('canonical_fallback'))
    d['x'].validate()
    assert len(d['y'])==1942 and len(set(d['person_id']))==756
    assert np.all(d['y'][d['source']=='LJMU']==100)
    for f in range(5):
        for j in range(5):
            if j!=f:r.split(d,f,j)
    return d

def identity(d,c,f,j,epochs,purpose):
    tr,va,te=r.split(d,f,j)
    seed=c['seed']+100*f+(50 if j is None else j)
    return {'bundle_sha256':sha(HERE/'manifest.json'),'family':a.family,'arm':'Stanford_plus_LJMU100',
            'purpose':purpose,'outer_fold':f,'inner_fold':j,'epochs':epochs,
            ('fit_seed' if a.family=='CNN' else 'seed'):seed,
            'train_people':np.unique(d['person_id'][tr]).tolist(),
            'validation_people':np.unique(d['person_id'][va]).tolist(),
            'outer_evaluation_people':np.unique(d['person_id'][te]).tolist()}

def new_model(c):return r.GdiCNN(c['dropout']) if a.family=='CNN' else r.new_model()
def predict_checkpoint(path,x,c):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    model=new_model(c);model.load_state_dict(ck['model'],strict=True);model.to(a.device)
    raw=r.predict(model,x,ck['lo'],ck['span'],a.device)
    return ck,raw

def summarize_sources(d,mask,pred):
    results={};ix=np.flatnonzero(mask)
    for source in ['Stanford','LJMU']:
        use=d['source'][ix]==source
        if not use.any():continue
        ids=d['person_id'][ix][use];y=d['y'][ix][use];p=pred[use]
        errors=np.array([np.abs(p[ids==q]-y[ids==q]).mean() for q in np.unique(ids)])
        results[source]={'people':len(errors),'records':int(use.sum()),'mae':float(errors.mean())}
    return results

def train_fold(d,c,f):
    dest=a.data_root/'runs/mixed'/a.family/f'fold{f}'
    if (dest/'complete.json').exists():
        done=read(dest/'complete.json');assert done['bundle_sha256']==sha(HERE/'manifest.json')
        assert sha(dest/'predictions.npz')==done['prediction_sha256'];return
    histories=[];member_predictions=[];members=[]
    te=d['fold']==f
    for j in range(5):
        if j==f:continue
        tr,va,_=r.split(d,f,j);folder=dest/f'inner{j}'
        ident=identity(d,c,f,j,c['max_epochs'],'inner_epoch_selection')
        model,lo,span,h=r.fit(d['x'][tr],d['y'][tr],d['person_id'][tr],d['x'][va],d['y'][va],d['person_id'][va],
                            c,c['max_epochs'],folder,ident,a.device)
        histories.append(h)
        train_raw=r.predict(model,d['x'][tr],lo,span,a.device)
        outer_raw=r.predict(model,d['x'][te],lo,span,a.device)
        coef=np.array(h[-1]['calibration']);excluded=bool((np.std(train_raw,axis=0)<.01).any())
        np.savez_compressed(folder/'replay.npz',train_row=np.flatnonzero(tr),train_raw=train_raw,
                            validation_row=np.flatnonzero(va),validation_raw=r.predict(model,d['x'][va],lo,span,a.device),
                            evaluation_row=np.flatnonzero(te),evaluation_raw=outer_raw)
        members.append({'tag':f'inner{j}','excluded_from_bag':excluded,'train_raw_sd':np.std(train_raw,axis=0).tolist(),
                        'checkpoint_sha256':sha(folder/'latest.pt'),'calibration':coef.tolist()})
        if not excluded:member_predictions.append(outer_raw*coef[0]+coef[1])
        del model
    epoch,means=r.select_epoch(histories)
    save(dest/'selection.json',{'selected_epoch':epoch,'inner_mean_person_mae_by_epoch':means,'outer_evaluation_used_for_selection':False})
    tr,_,te=r.split(d,f,None);folder=dest/'refit';ident=identity(d,c,f,None,epoch,'outer_refit')
    model,lo,span,h=r.fit(d['x'][tr],d['y'][tr],d['person_id'][tr],None,None,None,c,epoch,folder,ident,a.device)
    train_raw=r.predict(model,d['x'][tr],lo,span,a.device);raw=r.predict(model,d['x'][te],lo,span,a.device)
    coef=np.array(h[-1]['calibration']);pred=raw*coef[0]+coef[1]
    np.savez_compressed(folder/'replay.npz',train_row=np.flatnonzero(tr),train_raw=train_raw,evaluation_row=np.flatnonzero(te),evaluation_raw=raw)
    excluded=bool((np.std(train_raw,axis=0)<.01).any())
    members.append({'tag':'refit','excluded_from_bag':excluded,'train_raw_sd':np.std(train_raw,axis=0).tolist(),
                    'checkpoint_sha256':sha(folder/'latest.pt'),'calibration':coef.tolist()})
    if not excluded:member_predictions.append(pred)
    assert member_predictions,'All members collapsed'
    bag=np.mean(member_predictions,axis=0)
    np.savez_compressed(dest/'predictions.npz',row_index=np.flatnonzero(te),refit=pred,refit_raw=raw,bag=bag)
    save(dest/'complete.json',{'bundle_sha256':sha(HERE/'manifest.json'),'family':a.family,'fold':f,'fits':5,
                              'selected_epoch':epoch,'members':members,'prediction_sha256':sha(dest/'predictions.npz'),
                              'sources_refit':summarize_sources(d,te,pred),'sources_bag':summarize_sources(d,te,bag)})
    print('FOLD_COMPLETE',a.family,f,flush=True)

def baseline(d,c,folds):
    allmembers=read(HERE/'baseline-models.json')
    for f in folds:
        dest=a.data_root/'runs/baseline'/a.family/f'fold{f}';dest.mkdir(parents=True,exist_ok=True)
        te=d['fold']==f;preds=[];audit=[];refit=None;refit_raw=None
        for member in allmembers:
            if member['family']!=a.family or member['fold']!=f:continue
            path=a.data_root/member['checkpoint'];assert sha(path)==member['checkpoint_sha256']
            ck,raw=predict_checkpoint(path,d['x'][te],c)
            orig=ck['identity'];j=orig['inner_fold']
            tr=(d['source']=='Stanford')&(d['fold']!=f)
            if j is not None:tr&=d['fold']!=j
            assert set(orig['train_people'])==set(d['source_person_id'][tr])
            assert not set('Stanford:'+str(q) for q in orig['train_people'])&set(d['person_id'][te])
            coef=np.array(member['calibration']);pred=raw*coef[0]+coef[1];preds.append(pred)
            if member['tag']=='refit':refit=pred;refit_raw=raw
            audit.append({'tag':member['tag'],'checkpoint_sha256':sha(path),'calibration':coef.tolist()})
        assert refit is not None
        np.savez_compressed(dest/'predictions.npz',row_index=np.flatnonzero(te),refit=refit,refit_raw=refit_raw,bag=np.mean(preds,axis=0))
        save(dest/'complete.json',{'family':a.family,'fold':f,'members':audit,'prediction_sha256':sha(dest/'predictions.npz'),
                                  'sources_refit':summarize_sources(d,te,refit),'sources_bag':summarize_sources(d,te,np.mean(preds,axis=0))})
        print('BASELINE_COMPLETE',a.family,f,flush=True)

def pilot(d,c,synthetic=False):
    tr,va,te=r.split(d,0,1)
    if synthetic:
        # Numerical plumbing only; these are explicitly synthetic labels and features.
        rng=np.random.default_rng(20261104);shape=(12,)+d['x'].values.shape[1:]
        x=r.WindowSet(rng.normal(size=shape).astype(np.float32),np.repeat(np.arange(6),2),6)
        y=rng.uniform(45,105,(6,2));ids=np.array(['synthetic%d'%i for i in range(6)])
        tx,vx=x[np.arange(4)],x[np.arange(4,6)];ty,vy=y[:4],y[4:];ti,vi=ids[:4],ids[4:];epochs=2
    else:tx,vx=d['x'][tr],d['x'][va];ty,vy=d['y'][tr],d['y'][va];ti,vi=d['person_id'][tr],d['person_id'][va];epochs=3
    ident={'family':a.family,'purpose':'synthetic_smoke' if synthetic else 'real_pilot',
           ('fit_seed' if a.family=='CNN' else 'seed'):20261029,'epochs':epochs,
           'train_people':np.unique(ti).tolist(),'validation_people':np.unique(vi).tolist()}
    out=a.data_root/'runs'/('smoke' if synthetic else 'pilot')/a.family
    model,lo,span,h=r.fit(tx,ty,ti,vx,vy,vi,c,epochs,out,ident,a.device)
    raw=r.predict(model,vx,lo,span,a.device);ck,replay=predict_checkpoint(out/'latest.pt',vx,c)
    error=float(np.max(np.abs(raw-replay)));assert error<.001,error
    np.savez_compressed(out/'pilot_predictions.npz',validation_raw=raw,train_raw=r.predict(model,tx,lo,span,a.device))
    save(out/'verification.json',{'passed':True,'synthetic':synthetic,'epochs':epochs,'family':a.family,
                                 'mean_epoch_seconds':float(np.mean([v['seconds'] for v in h])),
                                 'checkpoint_replay_max_error':error,'checkpoint_sha256':sha(out/'latest.pt'),
                                 'outer_evaluation_used':False,'torch':str(torch.__version__),
                                 'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None})

d=load_data();c=read(HERE/a.family/'protocol.json');r.setup(20261028,a.device)
if a.action=='check':
    save(a.data_root/'runs/checks'/f'{a.family}.json',{'passed':True,'all20_person_splits':True,'records':len(d['y']),
                                                  'people':len(set(d['person_id'])),'windows':len(d['x'].values),'optimizer_steps':0})
elif a.action in ('smoke','pilot'):pilot(d,c,a.action=='smoke')
elif a.action=='baseline':baseline(d,c,[int(x) for x in a.folds.split(',')])
else:
    assert read(a.data_root/'runs/pilot'/a.family/'verification.json')['passed']
    for f in [int(x) for x in a.folds.split(',')]:train_fold(d,c,f)
