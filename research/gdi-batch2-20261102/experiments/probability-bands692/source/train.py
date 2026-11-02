"""All head fitting/selection/calibration happens within one base-held-out fold."""
from core import *
cfg=read(H/'protocol.json');proof=read(R/'analysis/verification.json');assert proof['passed']
assert proof['comparison_sha256']==sha(R/'analysis/comparison.json')
manifest=[]
for f in range(5):
 for j in range(5):
  if j==f:continue
  d=inner_data(f,j);groups=partition(d['ids'],f,j);disjoint(d['base_people'],*groups)
  masks=[np.isin(d['ids'],g) for g in groups];tr,va,ca=masks;refit=tr|va
  models={};choices=[];candidates=[];cal={}
  for family in cfg['families']:
   family_candidates=[]
   for lam in cfg['ridge_grid']:
    model=fit_head(d['x'][tr],d['base'][tr],d['y'][tr],d['ids'][tr],family,lam)
    loss=nll(predict(model,d['x'][va],d['base'][va]),d['y'][va],weights(d['ids'][va]))
    row=dict(family=family,lambda_=lam,selection_nll=loss);family_candidates.append(row);candidates.append(row)
   best=min(family_candidates,key=lambda a:a['selection_nll']);choices.append(best)
   model=fit_head(d['x'][refit],d['base'][refit],d['y'][refit],d['ids'][refit],family,best['lambda_'])
   raw=predict(model,d['x'][ca],d['base'][ca]);temp=fit_temperature(raw,d['y'][ca],weights(d['ids'][ca]))
   model['temperature']=temp;models[family]=model;cal[family+'_raw']=raw;cal[family+'_calibrated']=temperature(raw,temp)
  selected=min(choices,key=lambda a:a['selection_nll'])['family']
  residual_scale=fit_residual(d['base'][refit],d['y'][refit],d['ids'][refit])
  raw=probabilities(d['base'][ca],np.full(sum(ca),residual_scale));temp=fit_temperature(raw,d['y'][ca],weights(d['ids'][ca]))
  cal['residual_raw']=raw;cal['residual_calibrated']=temperature(raw,temp)
  path=CACHE/('fold%d_inner%d.json'%(f,j));path.parent.mkdir(parents=True,exist_ok=True)
  calpath=path.with_suffix('.npz');np.savez_compressed(calpath,**cal,y=d['y'][ca],person_id=d['ids'][ca],video_id=d['video'][ca])
  row=dict(outer=f,inner=j,base_people=d['base_people'].tolist(),train_people=groups[0].tolist(),selection_people=groups[1].tolist(),calibration_people=groups[2].tolist(),
   selected_family=selected,choices=choices,candidates=candidates,models=models,residual_scale=residual_scale,residual_temperature=temp,
   source_hashes=d['hashes'],protocol_sha256=sha(H/'protocol.json'),calibration_predictions_sha256=sha(calpath),outer_predictions_opened=False)
  save(path,row);manifest.append(dict(path=str(path),sha256=sha(path),outer=f,inner=j,selected_family=selected,
      group_counts=[len(g) for g in groups],temperatures={k:v['temperature'] for k,v in models.items()}))
  print(json.dumps(manifest[-1]),flush=True)
save(O/'training.json',dict(complete=True,experts=manifest,protocol_sha256=sha(H/'protocol.json'),outer_predictions_opened=False,
                         source_audit_sha256=sha(R/'analysis/verification.json'),neural_optimizer_steps=0))
