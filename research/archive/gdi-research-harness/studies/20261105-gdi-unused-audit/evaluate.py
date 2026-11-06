"""Descriptive remaining-data evaluation, separately by identity/history/quality."""
from prepare_inputs import *
import csv,io,itertools,math
sys.path.insert(0,str(CASE));metrics=module('case_metrics',CASE/'analyze.py')
def main():
 seal=read(HERE/'code-seal.json')
 for n,dig in seal['code'].items():assert sha(HERE/n)==dig,n
 lock=read(HERE/'predictions-locked.json');assert lock['passed'] and sha(BIG/'predictions.npz')==lock['predictions_sha256'] and sha(HERE/'prediction-membership.json')==lock['membership_sha256'];assert not (HERE/'results.json').exists()
 with np.load(BIG/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 membership=read(HERE/'prediction-membership.json')['records'];inv=read(HERE/'inventory.json');audit=read(HERE/'input-audit.json');policy=read(HERE/'label-policy.json');assert policy['old38_numeric_labels'] in [False,True];save(HERE/'evaluation-started.json',dict(protocol_sha256=sha(HERE/'protocol.json'),prediction_sha256=lock['predictions_sha256'],label_policy=policy,started_at=datetime.datetime.now().astimezone().isoformat()))
 rows=membership;allowed=[r for r in rows if r['person_id'] and (r['history']!='old38_test' or policy['old38_numeric_labels'])];keys={(r['person_id'],r['exam_id'],side) for r in allowed for side in ['L','R']};link=read(S/'20261027-gdi-linkage/linkage_summary.json');content=prep.arc.read_member(link['archive_member']);assert hashlib.sha256(content).hexdigest()==link['archive_member_sha256'];labels={}
 for r in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))):
  key=(r['Patient_ID'],r['examid'],r['side'])
  if key in keys:labels.setdefault(key,[]).append(r['GDI'])
 y=np.full((len(rows),2),np.nan);labelstatus=[]
 for i,r in enumerate(rows):
  if not r['person_id']:reason='identity_unresolved_accuracy_unavailable'
  elif r['history']=='old38_test' and not policy['old38_numeric_labels']:reason='old38_labels_kept_closed'
  else:
   reason=None;values=[]
   for side in ['L','R']:
    a=labels.get((r['person_id'],r['exam_id'],side),[])
    if len(a)!=1:reason='missing_or_duplicate_GDI';break
    try:v=float(a[0])
    except (ValueError,TypeError):reason='nonnumeric_GDI';break
    if not math.isfinite(v):reason='nonfinite_GDI';break
    values.append(v)
   if reason is None:y[i]=values
  labelstatus.append(reason or 'finite_unique_bilateral')
 names=[k[6:] for k in d if k.startswith('full__')];pred={n:d['full__'+n] for n in names};complete=np.isfinite(y).all(1)&np.logical_and.reduce([np.isfinite(p).all(1) for p in pred.values()]);cohort=np.array([r['cohort'] for r in rows]);quality=np.array([r['quality'] for r in rows]);history=np.array([r['history'] for r in rows]);ids=d['person_id'];groups={};group_masks={};summary=[]
 for co in ['primary_remaining','additional_final','additional_development']:
  for qu in ['all','standard_window_gate_only','high_missing_exploratory']:
   mask=complete&(cohort==co)
   if qu!='all':mask&=quality==qu
   key=co+'__'+qu;group_masks[key]=mask;persons=set(ids[mask]);pm={p:float(y[(ids==p)&mask].mean()) for p in persons};persontruth=np.array([pm.get(p,np.nan) for p in ids]);bands={g:mask&mm for g,mm in [('all',np.ones(len(rows),bool)),('below60',persontruth<60),('90plus',persontruth>=90),('100plus',persontruth>=100)]};groups[key]={n:{g:metrics.metric(p,y,ids,m) for g,m in bands.items()} for n,p in pred.items()}
   for n in names:
    z=groups[key][n];summary.append(dict(cohort=co,quality=qu,model=n,people=z['all']['people'],records=z['all']['records'],mae=z['all']['mae'],bias=z['all']['bias'],high90_people=z['90plus']['people'],high90_mae=z['90plus']['mae'],high90_bias=z['90plus']['bias']))
 historical={}
 for hi in ['old37_validation','old38_test','not_old37_38']:
  m=complete&(cohort=='primary_remaining')&(history==hi);historical[hi]={n:metrics.metric(p,y,ids,m) for n,p in pred.items()}
 agreement=[]
 for co in ['primary_remaining','additional_final','additional_development','unlinked']:
  m=(cohort==co)&np.logical_and.reduce([np.isfinite(p).all(1) for p in pred.values()])
  for a,b in itertools.combinations(names,2):
   if not m.any():value=None
   elif co=='unlinked':value=float(abs(pred[a][m]-pred[b][m]).mean())
   else:_,vv=metrics.people_values(abs(pred[a]-pred[b]).mean(1),ids,m);value=float(vv.mean())
   agreement.append(dict(cohort=co,model_a=a,model_b=b,MAD=value,records=int(m.sum()),unit='record because identity unresolved' if co=='unlinked' else 'person'))
 stack=np.stack([pred[n] for n in names]);spread=np.mean(np.max(stack,axis=0)-np.min(stack,axis=0),axis=1);numeric_failures={n:int((~np.isfinite(p).all(1)).sum()) for n,p in pred.items()};save(HERE/'label-audit.json',dict(status_counts=dict(Counter(labelstatus)),statuses=[dict(index=r['index'],status=status) for r,status in zip(rows,labelstatus)],annotation_sha256=hashlib.sha256(content).hexdigest(),old38_numeric_opened=policy['old38_numeric_labels']))
 np.savez_compressed(BIG/'scored.npz',y=y,person_id=ids,video_id=d['video_id'],cohort=cohort,quality=quality,history=history,common_accuracy_mask=complete,spread=spread,**pred)
 result=dict(protocol_sha256=sha(HERE/'protocol.json'),predictions_sha256=lock['predictions_sha256'],scored_sha256=sha(BIG/'scored.npz'),models=names,inventory_counts=inv['counts'],input_records=audit['records'],available_input_records=audit['available'],unavailable_input_reasons=audit['unavailable_reasons'],input_quality_counts=audit['quality_counts'],label_status_counts=dict(Counter(labelstatus)),model_numeric_failure_records=numeric_failures,groups=groups,historical_primary_subsets=historical,agreement=agreement,cohort_availability={co:dict(inventory_records=sum(r['cohort']==co for r in inv['records']),input_available_records=int((cohort==co).sum()),accuracy_records=int((complete&(cohort==co)).sum()),identified_people=len({r['person_id'] for r in inv['records'] if r['cohort']==co and r['person_id']}),input_available_people=len(set(ids[(cohort==co)&(ids!='')])),disagreement_range_mean=float(np.mean(spread[(cohort==co)&np.isfinite(spread)])) if np.any((cohort==co)&np.isfinite(spread)) else None) for co in ['primary_remaining','additional_final','additional_development','unlinked']},new_training=False,scope='Existing source, additional flawed records, descriptive audit. Current-model-unseen people not necessarily untouched by older studies. Additional visits of seen people and unknown identities kept separate. No model selection or calibration.')
 save(HERE/'results.json',result);metrics.writecsv(HERE/'model-comparison.csv',summary);metrics.writecsv(HERE/'model-agreement.csv',agreement)
 fields=['source_index','person_id','video_id','exam_id','cohort','history','quality','label_status','selected_windows','missing_min','missing_max','any_coordinate_entire_window_missing','model_disagreement_range','true_left','true_right']+[n+side for n in names for side in ['_left','_right']]
 with (BIG/'all-record-predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();lookup={r['index']:i for i,r in enumerate(rows)}
  for r in inv['records']:
   q=audit['quality'][r['index']];i=lookup.get(r['index']);out={k:r.get(k,'') for k in ['person_id','video_id','exam_id','cohort']};out['source_index']=r['index']
   if i is None:out.update(quality='unavailable',label_status=q['reason'],history='not_scored');wr.writerow(out);continue
   out.update(history=rows[i]['history'],quality=rows[i]['quality'],label_status=labelstatus[i],selected_windows=q['selected_windows'],missing_min=q['selected_missing_min'],missing_max=q['selected_missing_max'],any_coordinate_entire_window_missing=q['selected_all_absent_coordinate_windows'],model_disagreement_range=float(spread[i]) if math.isfinite(spread[i]) else '',true_left=float(y[i,0]) if math.isfinite(y[i,0]) else '',true_right=float(y[i,1]) if math.isfinite(y[i,1]) else '')
   for n in names:
    for j,side in enumerate(['_left','_right']):out[n+side]=float(pred[n][i,j]) if math.isfinite(pred[n][i,j]) else ''
   wr.writerow(out)
 save(HERE/'evaluation-receipt.json',dict(passed=True,results_sha256=sha(HERE/'results.json'),scored_sha256=sha(BIG/'scored.npz'),all_record_csv_sha256=sha(BIG/'all-record-predictions.csv'),label_policy=policy,optimizer_steps=0));print(json.dumps(dict(evaluation_ready=True,cohorts=result['cohort_availability'],label_counts=result['label_status_counts'])),flush=True)
if __name__=='__main__':main()
