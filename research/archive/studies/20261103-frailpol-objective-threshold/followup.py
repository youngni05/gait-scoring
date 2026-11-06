from pathlib import Path
import sys,json,hashlib,time,itertools,gc,warnings,datetime
OLD=Path('D:/Projects/D_P/studies/20261103-frailpol-short-walk');sys.path.insert(0,str(OLD))
from models import factory,sha,read,save,np,torch
import lightgbm as lgb
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
H=Path('D:/Projects/D_P/studies/20261103-frailpol-objective-threshold');B=Path('D:/gait-research-data/D_P/frailpol-objective-threshold-20261103');SOURCE=Path('D:/gait-research-data/D_P/frailpol-short-walk-20261103')
warnings.filterwarnings('ignore',message='auto_scale_n_estimators is deprecated')
def threshold(y,p,budget):
    candidates=np.r_[0.,np.nextafter(np.unique(p),np.inf)]
    ok=[t for t in candidates if np.mean(p[y!=2]>=t)<=budget]
    return float(min(ok))
def weights(y,target,mode):
    if mode!='weighted':return np.ones(len(y))
    if target=='frailty':return len(y)/(3*np.bincount(y.astype(int),minlength=3)[y.astype(int)])
    return np.where(y>=20,4.,np.where(y>=12,2.,1.))
def fit_lgb(X,y,target,mode,cfg):
    p=read(OLD/'protocol.json')['LGBM']['fixed'].copy();p.update(cfg);rounds=p.pop('num_boost_round')
    if target=='frailty':p.update(objective='multiclass',num_class=3)
    return lgb.train(p,lgb.Dataset(X,label=y,weight=weights(y,target,mode)),num_boost_round=rounds)
def main():
    H.mkdir(parents=True,exist_ok=True);B.mkdir(parents=True,exist_ok=True)
    for d in ['contexts','predictions','models','inner']:(B/d).mkdir(exist_ok=True)
    oldp=read(OLD/'protocol.json');rr=read(SOURCE/'cohorts.json')
    protocol=dict(owner_thread='01a1a1c2-655b-79a3-87bc-e325af86028e',frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_cohort_sha256=sha(SOURCE/'cohorts.json'),source_run_sha256=sha(OLD/'run-receipt.json'),code_sha256=sha(Path(__file__)),scope='Same people/features/folds. TUG raw seconds LGBM/TabICLv2/TabPFNv2; weighted raw seconds LGBM. Frailty original models threshold-only, plus inverse-frequency weighted multiclass LGBM.',tug_weight='fixed2 if12<=y<20,4 ify>=20,else1; select weighted model by inner weighted MAE,raw model by inner MAE',threshold='P(frail)>=threshold; smallest threshold satisfying pooled inner OOF nonfrail FPR<=0.10(primary) or0.20(secondary). No external labels. Also report original argmax and P(frail)>=0.5.',selection='LGBM same16grid/inner3splits. Weighted frailty choose mean per-class logloss. Foundation fixed8ensembles. No new probability calibration. Original LGBM selected inner predictions reused for threshold. Foundation inner3 new fits on outertrain only.',metrics='TUG paired MAE,bias,slow>=20 n9,12..20,under12; stratified frailty rates,ROC-AUC,AP,recall,FPR,PPV and confusion; paired bootstrap10000 conditional fixed OOF. Compare speed/features and same-input new/original. Report actual held-out FPR; target not guaranteed. Exploratory reused development cohort.',verification='Hash source/checkpoints, train/test IDs, inner partitions, independently reconstruct threshold and selection, save/reload new LGBM final+inner, reconstruct all new foundation final+inner from contexts; atol/rtol1e-6; independent outcome metrics.',storage=dict(expected_additional_GB=1,small=str(H),large=str(B)))
    if not (H/'protocol.json').exists():save(H/'protocol.json',protocol)
    p=read(H/'protocol.json');assert p['source_cohort_sha256']==sha(SOURCE/'cohorts.json') and p['code_sha256']==sha(Path(__file__))
    for w in read(OLD/'weights-manifest.json'):assert sha(w['path'])==w['sha256']
    for name,h in read(OLD/'run-receipt.json')['predictions'].items():assert sha(SOURCE/'predictions'/name)==h
    grid=oldp['LGBM']['grid'];cfgs=[dict(zip(grid,v)) for v in itertools.product(*grid.values())];done=0
    for target in ['frailty','tug']:
      rows=[r for r in rr if r['status']=='ok' and r[target] is not None];ids=np.array([r['id'] for r in rows]);y=np.array([r[target] for r in rows]);fold=np.array([r['fold'] for r in rows]);names=list(rows[0]['features']);xx=np.array([[r['features'][c] for c in names] for r in rows]);cl=target=='frailty'
      for cond in ['speed','features']:
       X=xx[:,:1] if cond=='speed' else xx
       for outer in range(5):
        tr=np.flatnonzero(fold!=outer);te=np.flatnonzero(fold==outer);splitter=StratifiedKFold(3,shuffle=True,random_state=20261031) if cl else KFold(3,shuffle=True,random_state=20261031);splits=list(splitter.split(tr,y[tr]))
        cases=[('LGBM','original'),('TabICLv2','original'),('TabPFNv2','original'),('LGBM','weighted')] if cl else [('LGBM','raw'),('TabICLv2','raw'),('TabPFNv2','raw'),('LGBM','weighted')]
        for kind,mode in cases:
          stem=f'{target}-{cond}-{outer}-{kind}-{mode}';out=B/'predictions'/(stem+'.json')
          if out.exists():done+=1;continue
          t=time.monotonic();ctx=B/'contexts'/(stem+'.npz');np.savez_compressed(ctx,X_train=X[tr],y_train=y[tr],X_test=X[te],train_ids=ids[tr],test_ids=ids[te]);inner=[];selection=None;replays=[]
          if kind=='LGBM':
            if mode=='original':
              orig=read(SOURCE/'predictions'/f'{target}-{cond}-{outer}-LGBM.json');selection=orig['selection'];inner=read(SOURCE/'inner'/f'{target}-{cond}-{outer}.json');pred=np.array(orig['prediction']);model_path=orig['model'];assert sha(model_path)==orig['model_sha256']
            else:
              losses=[]
              for ci,cfg in enumerate(cfgs):
                errs=[];ww=[]
                for k,(it,iv) in enumerate(splits):
                  m=fit_lgb(X[tr[it]],y[tr[it]],target,mode,cfg);ip=m.predict(X[tr[iv]],num_threads=2);e=-np.log(np.clip(ip[np.arange(len(iv)),y[tr[iv]].astype(int)],1e-15,1)) if cl else abs(ip-y[tr[iv]])
                  # Validation class weights use validation class counts, yielding macro logloss.
                  w=weights(y[tr[iv]],target,mode);errs.extend(e);ww.extend(w);inner.append(dict(candidate=ci,inner=k,train=ids[tr[it]].tolist(),evaluation=ids[tr[iv]].tolist(),prediction=ip.tolist()))
                losses.append(float(np.average(errs,weights=ww)))
              best=int(np.argmin(losses));selection=dict(candidate=best,params=cfgs[best],losses=losses)
              for k,(it,iv) in enumerate(splits):
                m=fit_lgb(X[tr[it]],y[tr[it]],target,mode,cfgs[best]);mp=B/'models'/f'{stem}-inner{k}.txt';m.save_model(str(mp));ip=lgb.Booster(model_file=str(mp)).predict(X[tr[iv]],num_threads=2);saved=next(v['prediction'] for v in inner if v['candidate']==best and v['inner']==k);delta=float(np.max(abs(ip-np.array(saved))));assert delta<1e-6;replays.append(delta)
              m=fit_lgb(X[tr],y[tr],target,mode,cfgs[best]);pred=m.predict(X[te],num_threads=2);model_path=str(B/'models'/(stem+'.txt'));m.save_model(model_path)
            replay=lgb.Booster(model_file=model_path).predict(X[te],num_threads=2);replays.append(float(np.max(abs(replay-pred))));assert np.allclose(replay,pred,atol=1e-6,rtol=1e-6)
          else:
            if cl:
              for k,(it,iv) in enumerate(splits):
                m=factory(kind,True);m.fit(X[tr[it]],y[tr[it]]);ip=m.predict_proba(X[tr[iv]]);assert np.array_equal(m.classes_,[0,1,2]);del m;gc.collect()
                m=factory(kind,True);m.fit(X[tr[it]],y[tr[it]]);replay=m.predict_proba(X[tr[iv]]);replays.append(float(np.max(abs(replay-ip))));assert np.allclose(replay,ip,atol=1e-6,rtol=1e-6)
                inner.append(dict(candidate=0,inner=k,train=ids[tr[it]].tolist(),evaluation=ids[tr[iv]].tolist(),prediction=ip.tolist()));del m;gc.collect()
              pred=np.array(read(SOURCE/'predictions'/f'{target}-{cond}-{outer}-{kind}.json')['prediction'])
            else:
              m=factory(kind,False);m.fit(X[tr],y[tr]);pred=m.predict(X[te]);del m;gc.collect()
              m=factory(kind,False);m.fit(X[tr],y[tr]);replay=m.predict(X[te]);replays.append(float(np.max(abs(replay-pred))));assert np.allclose(replay,pred,atol=1e-6,rtol=1e-6);del m;gc.collect()
          thresholds={}
          if cl:
            picked=[v for v in inner if v['candidate']==(selection['candidate'] if selection else 0)];lookup={int(i):j for j,i in enumerate(ids)};ii=[];pp=[]
            for v in picked:ii.extend(v['evaluation']);pp.extend(np.array(v['prediction'])[:,2].tolist());assert not set(v['train'])&set(v['evaluation']) and set(v['train'])|set(v['evaluation'])==set(ids[tr])
            assert len(ii)==len(tr) and set(ii)==set(ids[tr]);yy=np.array([y[lookup[i]] for i in ii]);pp=np.array(pp)
            thresholds={str(b):threshold(yy,pp,b) for b in [.1,.2]}
          save(B/'inner'/(stem+'.json'),inner)
          rec=dict(target=target,condition=cond,fold=outer,kind=kind,mode=mode,prediction=pred.tolist(),thresholds=thresholds,selection=selection,context=str(ctx),context_sha256=sha(ctx),protocol_sha256=sha(H/'protocol.json'),replay_max=max(replays or [0]),seconds=time.monotonic()-t)
          if kind=='LGBM':rec.update(model=model_path,model_sha256=sha(model_path))
          save(out,rec);done+=1;save(H/'progress.json',dict(status='running',complete=done,expected=80,last=stem));print(done,stem,round(rec['seconds'],1),'s',flush=True)
    save(H/'run-receipt.json',dict(status='fits_and_replays_complete_pending_independent_review',cases=done,hashes={str(f):sha(f) for folder in ['contexts','predictions','models','inner'] for f in (B/folder).glob('*')},protocol_sha256=sha(H/'protocol.json')))
if __name__=='__main__':main()
