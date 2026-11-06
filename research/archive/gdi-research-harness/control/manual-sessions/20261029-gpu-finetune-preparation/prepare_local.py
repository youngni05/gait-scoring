from pathlib import Path
import csv,json,hashlib,shutil,sys
R=Path('C:/Users/jun/Desktop/D_P');H=R/'gdi-research-harness';S=H/'control/manual-sessions/20261029-gpu-finetune-preparation';B=S/'bundle';B.mkdir(exist_ok=False)
sys.path.insert(0,str(R/'outputs/gdi-ridge-experiment-20261016/deps'));import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sources={}
def checked(p,receipt):
 a={x['path']:x['sha256'] for x in load(receipt)['artifacts']}; assert sha(p)==a[p.relative_to(H).as_posix()];sources[p.relative_to(R).as_posix()]=sha(p)
old=H/'studies/20261028-window-cause-controls';models=H/'studies/20261028-missing-augmentation-learning/models.npz';lab=H/'control/desktop-sessions/20261026T130506Z-e876e7/length_internal_oof.csv';cv=H/'control/desktop-sessions/20261026T041807Z-f050df/internal_cv_protocol.json'
for p,receipt in [(old/'features.npz',old/'study_receipt.json'),(models,models.parent/'study_receipt.json'),(lab,lab.parent/'receipt.json'),(cv,cv.parent/'receipt.json')]:checked(p,receipt)
with np.load(old/'features.npz',allow_pickle=False) as z:ids=z['patient_ids'].copy();vid=z['video_ids'].copy();features=np.stack([z['front'],z['last']],axis=1)
labels={x['video_id']:x for x in csv.DictReader(lab.open(encoding='utf-8'))};assert set(labels)==set(map(str,vid));assert len(ids)==203 and len(set(ids))==175
fold=np.array([int(labels[str(v)]['fold']) for v in vid]);y=np.array([[float(labels[str(v)]['actual_L']),float(labels[str(v)]['actual_R'])] for v in vid]);protocol=load(cv)
for i,v in enumerate(vid):assert str(ids[i])==labels[str(v)]['patient_id']
for k in range(5):assert set(map(str,ids[fold==k]))==set(protocol['fold_people'][k]);assert not set(ids[fold==k])&set(ids[fold!=k])
inputs=[]
for i in range(203):
 p=old/f'records/row_{i:04d}.npz';checked(p,old/'study_receipt.json')
 with np.load(p,allow_pickle=False) as z:inputs.append(np.stack([z['front_input'],z['last_input']]))
x=np.stack(inputs);assert x.shape==(203,2,124,17,3) and x.dtype==np.float32 and np.isfinite(x).all()
np.savez_compressed(B/'train_only.npz',x=x,y=y,person_id=ids,video_id=vid,fold=fold,features=features)
with np.load(models,allow_pickle=False) as z:
 baseline={f'fold{k}__{key}':z[f'fold{k}__{key}'].copy() for k in range(5) for key in ['scaler_mean','scaler_scale','coef','intercept']}
np.savez_compressed(B/'baseline_heads.npz',**baseline)
pred=np.empty((203,2,2))
for k in range(5):
 ix=fold==k;prefix=f'fold{k}__';pred[ix]=((features[ix]-baseline[prefix+'scaler_mean'])/baseline[prefix+'scaler_scale'])@baseline[prefix+'coef'].T+baseline[prefix+'intercept']
macro=lambda a:float(np.mean([a[ids==p].mean() for p in np.unique(ids)]))
metrics={'front_mae':macro(abs(pred[:,0]-y).mean(1)),'back_mae':macro(abs(pred[:,1]-y).mean(1)),'front_back_gap':macro(abs(pred[:,0]-pred[:,1]).mean(1))}
assert abs(metrics['front_mae']-8.275834)<1e-5;assert abs(metrics['front_back_gap']-3.960483)<1e-5
np.savez_compressed(B/'baseline_predictions.npz',predictions=pred)
mb=R/'outputs/motionbert-smoke-20261015';(B/'lib/model').mkdir(parents=True)
for src,dest in [(mb/'assets/MB_lite.bin',B/'MB_lite.bin'),(mb/'vendor/LICENSE',B/'MOTIONBERT_LICENSE'),(mb/'vendor/lib/model/DSTformer.py',B/'lib/model/DSTformer.py'),(mb/'vendor/lib/model/drop.py',B/'lib/model/drop.py')]:shutil.copyfile(src,dest);sources[src.relative_to(R).as_posix()]=sha(src)
for p in [B/'lib/__init__.py',B/'lib/model/__init__.py']:p.write_text('',encoding='utf-8')
config={'status':'draft_not_for_real_training','variants':['head_only','tail','last_stage'],'folds':list(range(5)),'seeds':[20261028,20261029,20261030],'steps':None,'batch_size':None,'head_lr':None,'encoder_lr':None,'weight_decay':None,'loss':'record_equal_MSE_left_right','train_window':'front','evaluate_windows':['front','back'],'optimizer':'AdamW','dropout':'disabled_all_arms','protocol_note':'GPU shape-only profiling then predeclare fixed budget; do not choose using outer fold results.'}
(B/'protocol.draft.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
(S/'sources.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
(S/'preparation.json').write_text(json.dumps({'people':175,'records':203,'folds':5,'input_shape':list(x.shape),'baseline_replay':metrics,'protected_data_read':False,'real_data_optimizer_steps':0,'source_checks':len(sources),'actual_gpu_used':False,'source_torch':'2.14.1+cpu','source_numpy':'2.3.5'},indent=2),encoding='utf-8')
print(json.dumps({'bundle':str(B),'verified_sources':len(sources),'baseline':metrics,'input_shape':list(x.shape)}))
