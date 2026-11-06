"""Construct a frozen, standalone pair of pooling experiments from the running reference."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
H=Path(__file__).resolve().parent;B=H/'bundle';SOURCE=H.parent/'multiwindow/bundle'
def change(s,a,b):
    assert a in s,a
    return s.replace(a,b)
def main():
    B.mkdir(parents=True,exist_ok=True)
    keep=['data.npz','run.py','windows.py','sources.json','verify.py','verify_nested.py','capacity.py']
    for name in keep:
        shutil.copyfile(SOURCE/name,B/name)
    for folder in ['model','sources','vendor']:
        shutil.copytree(SOURCE/folder,B/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['bag_model.py','bags.py','verify_pooling.py','pilot_pair.py','review_pilot.py','submit.slurm','README.md']:
        shutil.copyfile(H/name,B/name)
    c=json.loads((SOURCE/'protocol.json').read_text())
    c.update(name='record_loss_STT_pooling_control',pooling='mean',variants=['mean','attention'],batch_records=8,
        loss='record_mean_of_two_side_squared_errors_after_window_pooling',window_aggregation='per_leg_uniform_or_softmax_learned_weights',
        note='Both controls train recording-level loss. Earlier STT trained per-window loss; use it as contextual baseline, not a pooling-only ablation.')
    c.pop('batch_size');c.pop('parallel_workers');c['primary_output']='pooled_record_scores_then_training_only_OLS_corrected'
    (B/'protocol.json').write_text(json.dumps(c,indent=2)+'\n')
    s=(B/'run.py').read_text()
    s=change(s,'from model.Transformer import SpatioTemporalTransformer','from bag_model import RecordingSTT\nfrom bags import pack,predict_details,weight_summary,audit_weights')
    s=change(s,'def new_model():','def new_model(mode="mean"):')
    s=change(s,'return SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)','return RecordingSTT(mode)')
    start=s.index('def predict(');end=s.index('def calibration(',start)
    s=s[:start]+'''def predict(model,x,lo,span,device):
    return predict_details(model,x,lo,span,device)[0]
'''+s[end:]
    s=change(s,'identity=dict(identity,train_x_sha=',"identity=dict(identity,model_variant=c['pooling'],training_config=c,train_x_sha=")
    s=change(s,'model=new_model().to(device)',"model=new_model(c['pooling']).to(device)")
    start=s.index('    xt=x.values.reshape');end=s.index("    history=[];path=out/'latest.pt'",start)
    s=s[:start]+"    yt=(y.astype(np.float32)-lo)/span\n"+s[end:]
    s=change(s,"c['batch_size']","c['batch_records']")
    s=change(s,'pred=model(torch.from_numpy(xt[ix]).to(device));target=torch.from_numpy(yt[ix]).to(device)',"values,valid,_=pack(x,ix,device)\n            pred=model(values,valid).reshape(-1);target=torch.from_numpy(yt[ix]).to(device).reshape(-1)")
    s=change(s,'loss=weighted_mse(pred,target,torch.from_numpy(weights[ix]).to(device))','loss=aligned_mse(pred,target)')
    s=change(s,"ident.update(train_x_sha=","ident.update(model_variant=c['pooling'],training_config=c,train_x_sha=")
    s=change(s,"2*len(d['x'][tr].values)/c['batch_records']","len(d['x'][tr])/c['batch_records']")
    s=change(s,"m=new_model().to(a.device).eval()","m=new_model(c['pooling']).to(a.device).eval()")
    s=change(s,"with torch.inference_mode():p=m(torch.from_numpy(d['x'].values[:2].reshape(-1,124,4,2)).to(a.device))","values,valid,_=pack(d['x'],np.arange(2),a.device)\n        with torch.inference_mode():p=m(values,valid)")
    s=change(s,"require(p.shape==(4,)","require(p.shape==(2,2)")
    s=change(s,"a=parser.parse_args();c=load(ROOT/'protocol.json');setup", "parser.add_argument('--variant',choices=['mean','attention'],required=True)\n    a=parser.parse_args();c=load(ROOT/'protocol.json');c.update(pooling=a.variant,name='record_loss_STT_'+a.variant);setup")
    s=change(s,"windows=len(d['x'].values),record_balanced_loss=True,primary_output='mean_windows_then_training_only_OLS_corrected; raw output diagnostic'","windows=len(d['x'].values),record_balanced_loss=True,primary_output='record_pooled_then_training_only_OLS_corrected; raw output diagnostic'")
    s=change(s,"'verify_nested.py','sources.json','windows.py','worker.py','submit.slurm'","'verify_nested.py','sources.json','windows.py','bag_model.py','bags.py','verify_pooling.py','pilot_pair.py','review_pilot.py','submit.slurm'")
    # Save learned per-window weights for each outer fold and each pilot.
    s=change(s,"coef=h[-1]['calibration'];raw=predict(m,d['x'][te],lo,span,device)","coef=h[-1]['calibration'];raw,detail=predict_details(m,d['x'][te],lo,span,device)\n    np.savez_compressed(dest/'window_weights.npz',**detail,row_index=np.flatnonzero(te))\n    save(dest/'weight_summary.json',weight_summary(d['x'][te],detail))")
    s=change(s,"raw=predict(model,d['x'][va],lo,span,a.device);coef=h[-1]['calibration'];pred=corrected(raw,coef)","raw,detail=predict_details(model,d['x'][va],lo,span,a.device);coef=h[-1]['calibration'];pred=corrected(raw,coef)\n    np.savez_compressed(a.output/'window_weights.npz',**detail,row_index=np.flatnonzero(va))\n    save(a.output/'weight_summary.json',weight_summary(d['x'][va],detail))")
    s=change(s,'result=dict(passed=True,pilot_only=True,epochs=3','result=dict(passed=True,pilot_only=True,variant=a.variant,epochs=3')
    s=change(s,"prediction_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=", "window_weights_sha256=sha(dest/'window_weights.npz'),weight_summary_sha256=sha(dest/'weight_summary.json'),prediction_sha256=sha(dest/'predictions.npz'),checkpoint_sha256=")
    s=change(s,"pred[ix]=z['predictions'][:,0];raw[ix]=z['predictions_raw'][:,0];seen[ix]+=1", """pred[ix]=z['predictions'][:,0];raw[ix]=z['predictions_raw'][:,0];seen[ix]+=1
            require(sha(dest/'window_weights.npz')==done['window_weights_sha256'] and sha(dest/'weight_summary.json')==done['weight_summary_sha256'],'Window weight hash mismatch')
            with np.load(dest/'window_weights.npz') as wd:
                np.testing.assert_array_equal(wd['row_index'],ix)
                audit_weights(d['x'][ix],wd,raw[ix],c['pooling'])
                require(weight_summary(d['x'][ix],wd)==load(dest/'weight_summary.json'),'Weight summary mismatch')""")
    (B/'run.py').write_text(s,encoding='utf8')
    # Existing fitting and nested isolation tests are retained for both variants.
    for name in ['verify.py','verify_nested.py']:
        s=(B/name).read_text()
        s=change(s,"a=p.parse_args();","p.add_argument('--variant',choices=['mean','attention'],required=True)\n    a=p.parse_args();")
        s=change(s,"c=run.load(run.ROOT/'protocol.json')","c=run.load(run.ROOT/'protocol.json');c.update(pooling=a.variant,name='record_loss_STT_'+a.variant)")
        s=s.replace('run.new_model()',"run.new_model(c['pooling'])")
        if name=='verify.py':
            start=s.index('    model=run.new_model');end=s.index('    with tempfile.TemporaryDirectory',start)
            s=s[:start]+'''    model=run.new_model(c['pooling']).to(a.device).eval()
    values,valid,_=run.pack(x,np.array([0,1]),a.device)
    with torch.inference_mode():
        grouped=model(values,valid)
        separate=torch.cat([model(values[i:i+1],valid[i:i+1]) for i in range(2)])
    torch.testing.assert_close(grouped,separate,rtol=1e-5,atol=1e-6)
'''+s[end:]
        (B/name).write_text(s,encoding='utf8')
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='manifest.json']
    manifest={}
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
        manifest[p.relative_to(B).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest['data.npz']==json.loads((SOURCE/'manifest.json').read_text())['data.npz']
    (B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=H/'gdi-stt692-window-attention.zip';prefix='gdi-stt692-window-attention-20261031/'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest)+['manifest.json']:z.write(B/name,prefix+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==digest
    report=dict(passed=True,python38_syntax=True,data_sha256=manifest['data.npz'],manifest_sha256=hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest(),zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,server_submitted=False)
    (H/'package_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
