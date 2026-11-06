"""Build a new multiwindow STT bundle without editing completed experiments."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
H=Path(__file__).resolve().parent
B=H/'bundle'
BASE=H.parent/'full/bundle'
CNN=H.parent.parent/'cnn-multiwindow695/bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)
def main():
    B.mkdir(parents=True,exist_ok=True)
    for folder in ('model','sources','vendor'):
        shutil.copytree(BASE/folder,B/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(BASE/'capacity.py',B/'capacity.py')
    assert sha(CNN/'data.npz')=='7c1912cd27c63a82ed0d0324f98c0c0fae9a608174d52bd7a502f05e9a8c7a1e'
    with np.load(CNN/'data.npz',allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    original=d['x'];d['x']=original[...,:8].reshape(11261,2,124,4,2).copy()
    np.testing.assert_array_equal(d['x'].reshape(11261,2,124,8),original[...,:8])
    np.savez_compressed(B/'data.npz',**d)
    source=json.loads((BASE/'sources.json').read_text())
    source.update(base_data_sha256=sha(CNN/'data.npz'),data_sha256=sha(B/'data.npz'),
        input_coordinates_exact_existing_cnn_first8=True,windows=11261,records=1061,people=692,
        identical_cnn_window_metadata=['window_record','window_start','canonical_fallback'],
        source_cnn_windows_sha256=sha(CNN/'windows.py'),source_single_STT_run_sha256=sha(BASE/'run.py'))
    save(B/'sources.json',source)
    save(H/'input_verification.json',dict(passed=True,people=692,records=1061,windows=11261,
        coordinates_exact=True,all_other_npz_arrays_unchanged=True,source_cnn_data_sha256=sha(CNN/'data.npz'),
        prepared_data_sha256=sha(B/'data.npz'),fallback_records=int(d['canonical_fallback'].sum()),protected_final_data_used=False))
    text=(CNN/'windows.py').read_text()
    text=replace(text,'(len(self.owner),2,124,12)','(len(self.owner),2,124,4,2)')
    (B/'windows.py').write_text(text,encoding='utf-8')
    c=json.loads((BASE/'protocol.json').read_text())
    c.update(name='official_STT_multiwindow_nested692',input='exact_CNN11261_windows_first8_xy',
        window_aggregation='uniform_mean_per_record_side_before_training_only_OLS',
        loss='global_mean_one_inverse_record_window_count_weighted_side_MSE',
        note='Same CNN windows, record weights, nested splits and evaluation. STT Adam/200epochs/batch128 retained; not compute matched.',
        primary_output='mean_windows_then_training_only_OLS_corrected',parallel_workers=[[0,1,2],[3,4]])
    save(B/'protocol.json',c)
    text=(BASE/'run.py').read_text()
    text=replace(text,'from model.Transformer import SpatioTemporalTransformer','from model.Transformer import SpatioTemporalTransformer\nfrom windows import WindowSet,weighted_mse')
    text=replace(text,"    require(d['x'].shape==(1061,2,124,4,2) and d['x'].dtype==np.float32,'Input shape/dtype')", "    d['x']=WindowSet(d['x'],d.pop('window_record'),len(d['y']),d.pop('window_start'),d.pop('canonical_fallback'))\n    d['x'].validate()\n    require(len(d['x'].values)==11261 and int(d['x'].fallback.sum())==4,'Window count/fallback mismatch')\n    for record in range(len(d['y'])):\n        starts=d['x'].starts[d['x'].owner==record];fallback=d['x'].fallback[d['x'].owner==record]\n        require(len(np.unique(starts))==len(starts),'Duplicate windows')\n        require(np.isin(starts[~fallback],np.arange(0,500-124,31)).all(),'Invalid window start')\n        if fallback.any():require(len(starts)==1,'Fallback record changed')")
    text=replace(text,"np.isfinite(d['x']).all()","np.isfinite(d['x'].values).all()")
    text=replace(text,'for k in range(0,len(x),64):','for k in range(0,len(x.values),64):')
    text=replace(text,'x[k:k+64].reshape(-1,124,4,2)','x.values[k:k+64].reshape(-1,124,4,2)')
    text=replace(text,'return np.concatenate(out)','return x.aggregate(np.concatenate(out))')
    text=replace(text,'def digest_array(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()',"def digest_array(a):\n    if isinstance(a,WindowSet):\n        fields=[digest_array(v) for v in (a.values,a.owner,a.starts,a.fallback)]\n        return hashlib.sha256(json.dumps([a.nrecords,fields]).encode()).hexdigest()\n    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()")
    text=replace(text,"require(x.shape==(len(y),2,124,4,2) and y.shape==(len(ids),2),'Training shape')", "x.validate();require(len(x)==len(y) and y.shape==(len(ids),2),'Training shape')")
    text=replace(text,"if vx is not None:require(vx.shape==(len(vy),2,124,4,2) and vy.shape==(len(vids),2),'Validation shape')", "if vx is not None:\n        vx.validate();require(len(vx)==len(vy) and vy.shape==(len(vids),2),'Validation shape')")
    text=replace(text,"xt=x.reshape(-1,124,4,2);yt=(y.astype(np.float32).reshape(-1)-lo)/span", "xt=x.values.reshape(-1,124,4,2);yt=(y[x.owner].astype(np.float32).reshape(-1)-lo)/span\n    weights=x.side_weights()")
    text=replace(text,"loss=aligned_mse(pred,target);", "loss=weighted_mse(pred,target,torch.from_numpy(weights[ix]).to(device));")
    text=replace(text,"math.ceil(2*int(tr.sum())/c['batch_size'])", "math.ceil(2*len(d['x'][tr].values)/c['batch_size'])",count=2)
    text=replace(text,"primary_output='training_only_OLS_corrected; raw output diagnostic'", "windows=len(d['x'].values),record_balanced_loss=True,primary_output='mean_windows_then_training_only_OLS_corrected; raw output diagnostic'")
    text=replace(text,"'verify_nested.py','sources.json')", "'verify_nested.py','sources.json','windows.py','worker.py','submit.slurm')")
    text=replace(text,"a=parser.parse_args();c=load", "parser.add_argument('--folds',type=int,nargs='+',choices=range(5))\n    a=parser.parse_args();c=load")
    text=replace(text,"save(a.output/'environment.json',env)", "save(a.output/('environment-%s-%d.json'%(os.getenv('SLURM_JOB_ID','local'),time.time_ns())),env)")
    text=replace(text,"for outer in range(5):train_fold(d,c,outer,a.output,a.device)", "require(a.folds is None or len(set(a.folds))==len(a.folds),'Duplicate worker folds')\n            for outer in (range(5) if a.folds is None else a.folds):train_fold(d,c,outer,a.output,a.device)\n            if a.folds is not None:\n                print('STT_WORKER_FOLDS_FINISHED',flush=True);return")
    text=replace(text,"d['x'][:2].reshape(-1,124,4,2)","d['x'].values[:2].reshape(-1,124,4,2)")
    text=replace(text,"input_shape=list(d['x'].shape)","windows=len(d['x'].values),input_shape=list(d['x'].values.shape)")
    # Keep the official model and all existing epoch diagnostics unchanged.
    (B/'run.py').write_text(text,encoding='utf-8')
    text=(BASE/'verify.py').read_text()
    text=replace(text,"y=rng.normal(70,10,(8,2));", "x=run.WindowSet(x,np.arange(8),8)\n    y=rng.normal(70,10,(8,2));")
    text=replace(text,'x[:2].reshape(-1,124,4,2)','x.values[:2].reshape(-1,124,4,2)')
    text=replace(text,'x[:2,s]','x.values[:2,s]')
    text=replace(text,'x[6:]+100',"run.WindowSet(x[6:].values+100,x[6:].owner,2)")
    # Unequal windows in actual fit tests, not only a standalone loss example.
    text=replace(text,"args=(x[:6],y[:6],ids", "training=run.WindowSet(np.concatenate([x.values[:6],x.values[:2]]),np.array([0,1,2,3,4,5,0,1]),6)\n        args=(training,y[:6],ids")
    text=replace(text,'run.fit(x[:6],y[:6],ids','run.fit(training,y[:6],ids',count=2)
    (B/'verify.py').write_text(text,encoding='utf-8')
    text=(BASE/'verify_nested.py').read_text()
    text=replace(text,"# Weighted choice differs", "d['x']=run.WindowSet(np.repeat(d['x'],np.tile([1,2,3,1],5),axis=0),np.repeat(np.arange(n),np.tile([1,2,3,1],5)),n)\n    # Weighted choice differs")
    text=replace(text,"altered={k:v.copy() for k,v in d.items()};altered['x'][d['fold']==0]+=100;altered['y'][d['fold']==0]+=1000", "altered={k:v.copy() for k,v in d.items() if k!='x'}\n        x=d['x'];ax=x.values.copy();ax[d['fold'][x.owner]==0]+=100\n        altered['x']=run.WindowSet(ax,x.owner.copy(),len(x),x.starts.copy(),x.fallback.copy())\n        altered['y'][d['fold']==0]+=1000")
    (B/'verify_nested.py').write_text(text,encoding='utf-8')
    print(json.dumps(dict(prepared=True,windows=11261,data_sha256=sha(B/'data.npz'))))
if __name__=='__main__':main()
