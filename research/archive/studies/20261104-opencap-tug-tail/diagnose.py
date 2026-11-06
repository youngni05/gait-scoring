from pathlib import Path
import json, numpy as np, csv, hashlib, collections
SRC=Path('D:/Projects/D_P/studies/20261104-opencap-camera-tug')
HERE=Path('D:/Projects/D_P/studies/20261104-opencap-tug-tail')
DATA=Path('D:/gait-research-data/D_P/opencap-camera-tug-20261104')
def main():
    HERE.mkdir(parents=True,exist_ok=True)
    d=np.load(DATA/'features.npz');X=d['X'];y=d['y'];ids=d['ids'];names=d['names'].tolist()
    meta=json.loads((SRC/'metadata.json').read_text());audit=json.loads((SRC/'source-audit.json').read_text())
    records=json.loads((DATA/'predictions.json').read_text());selections=json.loads((SRC/'selection.json').read_text())
    pred=np.full((5,len(y)),np.nan)
    for r in records:
        if r['view']=='speed_plus_landmarks':pred[r['repeat'],r['index']]=r['pred']
    high=y>=20;moderate=y>=12;groups={}
    for name,g in [('under12',y<12),('12to20',(y>=12)&(y<20)),('20plus',high)]:
        err=pred[:,g]-y[g];groups[name]={'n':int(g.sum()),'MAE':float(abs(err).mean()),'bias':float(err.mean()),'TUG_range':[float(y[g].min()),float(y[g].max())],
          'prediction_range':[float(pred[:,g].min()),float(pred[:,g].max())],'share_of_total_absolute_error':float(abs(err).sum()/abs(pred-y).sum())}
    gates={}
    for threshold in [10,12,15,20]:
        per=[]
        for p in pred:
            flag=p>=threshold
            per.append({'referred':int(flag.sum()),'severe_caught':int((flag&high).sum()),'severe_missed':int((~flag&high).sum()),
                'true_TUG_under12_referred':int((flag&(y<12)).sum()),'precision_TUG20plus':float((flag&high).sum()/flag.sum()) if flag.any() else None})
        gates[str(threshold)]=per
    severe=[]
    for i in np.flatnonzero(high):
        neighbors=np.argsort(abs(X[:,0]-X[i,0]));neighbors=neighbors[neighbors!=i][:10]
        folds=[]
        for s in selections:
            if s['view']=='speed_plus_landmarks' and i in s['test']:
                train=y[s['train']];folds.append({'repeat':s['repeat'],'model':s['selected'],'training_n_ge20':int((train>=20).sum()),'training_max':float(train.max()),'prediction':float(pred[s['repeat'],i])})
        severe.append({'id':str(ids[i]),'true_TUG':float(y[i]),'age':meta[i]['age'],'diagnosis':meta[i]['diagnosis'],
          'speed_mps':float(X[i,0]),'walking_6m_component_seconds':float(6/max(X[i,0],.1)),
          'mean_prediction':float(pred[:,i].mean()),'range_prediction':[float(pred[:,i].min()),float(pred[:,i].max())],
          'matched_speed_neighbor_TUGs':y[neighbors].tolist(),'matched_speed_neighbor_speeds':X[neighbors,0].tolist(),
          'extrema_count':audit[i]['extrema_count'],'timing_available':audit[i]['timing_available'],
          'leg_length':audit[i]['projected_leg_length_m'],'active_window_start':audit[i]['start_time'],'folds':folds})
    pairs=[]
    for i in range(len(y)):
      for j in range(i):
        if abs(X[i,0]-X[j,0])<.03 and abs(y[i]-y[j])>=10:
          pairs.append({'id1':str(ids[i]),'id2':str(ids[j]),'speed1':float(X[i,0]),'speed2':float(X[j,0]),'TUG1':float(y[i]),'TUG2':float(y[j])})
    idx=json.loads(Path('D:/Projects/D_P/gdi-research-harness/studies/20261030-opencap415-tug-timeseries/archive-index.json').read_text())
    members={x['name']:x for x in idx};availability={};total_selected={}
    for task in ['5xsts','tug_cone','tug_line']:
        found=[]
        for a in audit:
            prefix=a['member'].rsplit('/Markers/',1)[0]+'/'
            hits=[x for n,x in members.items() if n.startswith(prefix) and n.endswith('_task-'+task+'.trc')]
            if hits:
                found.append({'id':a['id'],'member':hits[0]['name']});total_selected[hits[0]['name']]=hits[0]
        availability[task]={'people':len(found),'severe_people':int(sum(next(y[k] for k,pid in enumerate(ids) if pid==r['id'])>=20 for r in found)),'members':found}
    out={'people':len(y),'group_errors_repeated':groups,'predicted_TUG_gates':gates,'severe_cases':severe,'same_speed_different_TUG_pairs':sorted(pairs,key=lambda r:-abs(r['TUG1']-r['TUG2']))[:20],
      'additional_task_availability_same_session':availability,'additional_task_total_compressed_bytes':sum(r['compressed'] for r in total_selected.values()),
      'interpretation_limits':['6/speed is an illustrative decomposition, not measured TUG walking time','No raw video or independently labelled subtask timings','All subgroup and neighbor analyses are descriptive; not training inputs or clinical thresholds']}
    (HERE/'diagnosis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'groups':groups,'gates':gates,'severe':[{k:r[k] for k in ['id','true_TUG','speed_mps','mean_prediction','matched_speed_neighbor_TUGs']} for r in severe],
      'extra_tasks':{k:{kk:vv for kk,vv in v.items() if kk!='members'} for k,v in availability.items()}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
