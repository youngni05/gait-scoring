"""Build an isolated multiwindow variant; never modify the running single-window bundle."""
from pathlib import Path
import json, shutil
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'cnn-paper-input695'
B=HERE/'bundle'

def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old))
    return text.replace(old,new)

def main():
    B.mkdir(parents=True,exist_ok=True)
    shutil.copytree(BASE/'bundle/sources',B/'sources',dirs_exist_ok=True)
    for name in ['model.py','reference_metrics.json','reference_predictions.npz']:
        shutil.copy2(BASE/'bundle'/name,B/name)
    source=(BASE/'prepare_inputs.py').read_text(encoding='utf-8')
    source=replace(source,"proofdata={**", "fallback = reason is None and not multi\n        if fallback: multi=[int(ids[0])-1]\n        proofdata={'canonical_fallback':fallback, **")
    source=replace(source,"source_raw_matched=raw[ids-1].astype(np.float64))", "source_raw_matched=raw[ids-1].astype(np.float64),\n                                multi_x=np.stack([np.stack([processed[k:k+124,left],processed[k:k+124,right]]) for k in multi]).astype(np.float32),\n                                window_start=np.array(multi,dtype=np.int64))")
    source=replace(source,"x=np.stack([np.load(cache/('%04d.npz'%i),allow_pickle=False)['x'] for i in idx])", "parts=[];owners=[];starts=[];fallback_flags=[]\n    for record,i in enumerate(idx):\n        with np.load(cache/('%04d.npz'%i),allow_pickle=False) as z:\n            parts.append(z['multi_x'].copy());starts.extend(z['window_start'].tolist())\n            owners.extend([record]*len(z['multi_x']));fallback_flags.extend([proofs[i]['canonical_fallback']]*len(z['multi_x']))\n        with np.load(STUDY/'cnn-paper-input695/records'/('%04d.npz'%i),allow_pickle=False) as old, np.load(cache/('%04d.npz'%i),allow_pickle=False) as now:\n            assert np.array_equal(old['x'],now['x']), 'Preprocessing changed'\n    x=np.concatenate(parts)")
    source=replace(source,'result.update(x=x,original_row_index=idx)', 'result.update(x=x,original_row_index=idx,window_record=np.asarray(owners,dtype=np.int64),window_start=np.asarray(starts,dtype=np.int64),canonical_fallback=np.asarray(fallback_flags,dtype=bool))')
    source=replace(source,"'actual_training_steps':0", "'windows':len(x),'fallback_records':sum(p['canonical_fallback'] for p in proofs), 'actual_training_steps':0")
    (HERE/'prepare_inputs.py').write_text(source,encoding='utf-8')

    source=(BASE/'bundle/run.py').read_text(encoding='utf-8')
    source=replace(source,'from model import GdiCNN, channels','from model import GdiCNN, channels\nfrom windows import WindowSet, weighted_mse')
    start=source.index('def validate_data(d):'); end=source.index('\n\ndef split(',start)
    source=source[:start]+'''def validate_data(d):
    n=len(d['y']); x=d['x']
    require(isinstance(x,WindowSet) and len(x)==n,'Wrong window container')
    x.validate()
    require(d['y'].shape==(n,2) and np.isfinite(d['y']).all(),'Invalid labels')
    require(n==1061 and len(set(d['person_id']))==692 and len(set(d['video_id']))==n,'Wrong cohort')
    for key in ('person_id','video_id','fold','cohort','original_row_index'):
        require(d[key].shape==(n,),'Wrong record metadata')
    require(len(np.unique(d['original_row_index']))==n,'Duplicate source rows')
    require(len(x.values)==11261 and int(x.fallback.sum())==4,'Wrong window count')
    for record in range(n):
        starts=x.starts[x.owner==record]
        require(len(np.unique(starts))==len(starts),'Duplicate window')
        regular=~x.fallback[x.owner==record]
        require(np.isin(starts[regular],np.arange(0,500-124,31)).all(),'Invalid regular start')
        if not regular.all():require(len(starts)==1,'Fallback must be a sole canonical window')
    for f in range(5):
        tr,_,te=split(d,f)
        require(len(set(d['person_id'][te]))==[138,138,138,139,139][f],'Wrong fold membership')
        for j in range(5):
            if j!=f:split(d,f,j)
''' + source[end:]
    source=replace(source,'    validate_data(d)\n    return d', "    d['x']=WindowSet(d['x'],d.pop('window_record'),len(d['y']),d.pop('window_start'),d.pop('canonical_fallback'))\n    validate_data(d)\n    return d")
    source=source.replace('paper_feature_matched_window_no_search','paper_feature_multiwindow_no_search').replace('matched124_official12_separate_sides','sliding124_stride31_official12_separate_sides')
    source=replace(source,'def step(model, opt, x, target, c):','def step(model, opt, x, target, c, weights=None):')
    source=replace(source,"mse = (model(x)[:,0] - target).square().mean()", "mse = weighted_mse(model(x)[:,0],target,weights)")
    start=source.index('def predict(');end=source.index('\n\ndef calibrate(',start)
    source=source[:start]+'''def predict(model,x,lo,span,device,batch=128):
    # Average windows within each record BEFORE training-only calibration / metrics.
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(x.values),batch):
            chunk=channels(torch.from_numpy(x.values[start:start+batch]))
            p=model(chunk.reshape(-1,12,124).to(device))[:,0]*span.to(device)+lo.to(device)
            result.append(p.reshape(-1,2).cpu().numpy())
    return x.aggregate(np.concatenate(result))
''' + source[end:]
    source=replace(source,"xt = channels(torch.from_numpy(x))\n    yt = ((torch.from_numpy(y.astype(np.float32)) - lo) / span).reshape(-1)\n    train_examples=xt.reshape(-1,12,124)\n    vx = channels(torch.from_numpy(valid_x)) if valid_x is not None else None", "xt = x\n    yt = ((torch.from_numpy(y[x.owner].astype(np.float32)) - lo) / span).reshape(-1)\n    train_examples=channels(torch.from_numpy(x.values)).reshape(-1,12,124)\n    example_weights=torch.from_numpy(x.side_weights())\n    vx = valid_x")
    source=replace(source,"yt[idx].to(device), c))", "yt[idx].to(device), c, example_weights[idx].to(device)))")
    source=replace(source,"channels(torch.from_numpy(d['x'][tr]))","d['x'][tr]")
    source=replace(source,"channels(torch.from_numpy(d['x'][te]))","d['x'][te]")
    source=replace(source,"'input_shape': list(d['x'].shape)","'input_shape': list(d['x'].values.shape), 'window_counts': [int(d['x'].counts.min()),int(d['x'].counts.max())], 'windows':len(d['x'].values)")
    source=replace(source,"'cnn_input_shape': [len(d['y'])*2, 12, 124]","'cnn_input_shape': [len(d['x'].values)*2, 12, 124]")
    source=replace(source,"2*int(split(d,f,j)[0].sum())", "2*len(d['x'][split(d,f,j)[0]].values)")
    source=replace(source,"2*int(tr.sum())", "2*len(d['x'][tr].values)")
    source=replace(source,"'primary_output':'training_only_OLS_corrected; raw is diagnostic',", "'primary_output':'mean_windows_then_training_only_OLS_corrected; raw is diagnostic',\n               'windows':len(d['x'].values), 'record_balanced_loss':True, 'canonical_fallback_records':int(d['x'].fallback.sum()),")
    source=replace(source,"ROOT / 'run.py', ROOT / 'model.py'", "ROOT / 'run.py', ROOT / 'model.py', ROOT / 'windows.py', ROOT / 'input_contract.json'")
    source=replace(source,"loss1 = step(m, opt, x, y, c)","probe_weights=torch.linspace(0.2,1.8,len(y),device=device)\n    loss1 = step(m, opt, x, y, c, probe_weights)")
    source=replace(source,"loss2 = step(m, opt, x, y, c)","loss2 = step(m, opt, x, y, c, probe_weights)")
    source=replace(source,"replay_loss = step(restored, opt2, x, y, c)","replay_loss = step(restored, opt2, x, y, c, probe_weights)")
    source=source.replace('Official-feature matched-window CNN','Official-feature multiwindow CNN')
    (B/'run.py').write_text(source,encoding='utf-8')
    config=json.loads((BASE/'bundle/protocol.json').read_text())
    config.update(name='cnn_official12_multiwindow_record_balanced_v1',stage='paper_feature_multiwindow_no_search',
        input='sliding124_stride31_official12_separate_sides',loss='mean_of_global_mean_one_inverse_record_window_count_weighted_side_MSE_plus_L2',
        window_aggregation='uniform_mean_per_record_side_before_training_only_OLS',
        window_starts='range(0,500-124,31), zero based; original canonical window fallback for four records',
        wall_time_seconds=27000,note='Same 692 people and original folds; all windows per epoch; more optimizer steps than single-window. Not exact paper reproduction.')
    (B/'protocol.json').write_text(json.dumps(config,indent=2)+'\n')
    launch=(BASE/'bundle/launch.sh').read_text().replace('PAPER_CNN','MULTIWINDOW_CNN')
    launch=launch.replace('10900s','27100s').replace('900s','1800s')
    (B/'launch.sh').write_text(launch,encoding='utf-8',newline='\n')
    submit=(BASE/'bundle/submit.slurm').read_text().replace('gdi-cnn-paper','gdi-cnn-multi').replace('04:00:00','08:00:00')
    (B/'submit.slurm').write_text(submit,encoding='utf-8',newline='\n')
    print('Isolated multiwindow preparation and runner generated.')

if __name__=='__main__':main()
