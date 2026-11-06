"""Build an isolated, fold-local SSL branch from the frozen multiwindow STT."""
from pathlib import Path
import ast, hashlib, json, shutil, zipfile
H=Path(__file__).resolve().parent
B=H/'bundle'; SOURCE=H.parent/'multiwindow/bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):p.write_text(s,encoding='utf-8')
def main():
    B.mkdir(exist_ok=True)
    source_hashes={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for p in SOURCE.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.name in ('manifest.json','worker.py','submit.slurm','README.md'):continue
        dest=B/p.relative_to(SOURCE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for p in (H/'implementation').iterdir():shutil.copyfile(p,B/p.name)
    c=json.loads((B/'protocol.json').read_text())
    c.update(name='STT_multiwindow_fold_local_LC_TC',parallel_workers=[[0,1,2,3,4]],
        ssl=dict(epochs=25,pilot_epochs=3,batch_size=128,learning_rate=0.0003,mask_frames=31,prefix_frames=93,
            objectives=['contiguous_mask_reconstruction','prefix_to_suffix_forecast'],objective_weights=[0.5,0.5],
            input_normalization='existing_fixed_per_record_preprocessing_no_new_dataset_statistics',
            sample_weight='same_record_balanced_window_side_weights_as_STT',selection='fixed_epochs_no_validation',
            transfer='spatial_and_temporal_encoder_only',downstream='all_STT_parameters_finetuned_from_fresh_GDI_head'))
    write(B/'protocol.json',json.dumps(c,indent=2)+'\n')
    s=(B/'run.py').read_text()
    s=s.replace('from windows import WindowSet,weighted_mse','from windows import WindowSet,weighted_mse\nimport pretrain')
    s=s.replace('identity=dict(identity,train_x_sha=', "identity=dict(identity,ssl_config=c['ssl'],train_x_sha=")
    s=s.replace('    opt=torch.optim.Adam(model.parameters()', "    pretrain.initialize(model,x,ids,c,out/'pretrain',device,identity['seed'])\n    setup(identity['seed'],device)\n    opt=torch.optim.Adam(model.parameters()")
    s=s.replace("return dict(bundle_sha256=sha(ROOT/'manifest.json'),purpose=purpose", "return dict(ssl_config=c['ssl'],bundle_sha256=sha(ROOT/'manifest.json'),purpose=purpose")
    s=s.replace("    complete=load(folder/'fit_complete.json');h=load(folder/'curve.json')", "    pretrain.audit(d['x'][tr],d['person_id'][tr],c,folder/'pretrain',ident['seed'])\n    complete=load(folder/'fit_complete.json');h=load(folder/'curve.json')")
    s=s.replace("'worker.py','submit.slurm')", "'worker.py','submit.slurm','pretrain.py','review_pilot.py','verify_ssl.py')")
    s=s.replace("    tr,va,te=split(d,0,1);torch.cuda.reset_peak_memory_stats()", "    c=dict(c,ssl=dict(c['ssl'],epochs=c['ssl']['pilot_epochs']))\n    tr,va,te=split(d,0,1);torch.cuda.reset_peak_memory_stats()")
    s=s.replace("result=dict(passed=True,pilot_only=True,epochs=3", "result=dict(passed=True,pilot_only=True,ssl_epochs=c['ssl']['epochs'],ssl_checkpoint_sha256=sha(a.output/'fit/pretrain/latest.pt'),epochs=3")
    s=s.replace("result=dict(method=c['name'],metrics=", "result=dict(pretraining_fits=25,pretraining_epochs=c['ssl']['epochs'],method=c['name'],metrics=")
    write(B/'run.py',s)
    for name in ('verify.py','verify_nested.py'):
        s=(B/name).read_text()
        s=s.replace("c=run.load(run.ROOT/'protocol.json')", "c=run.load(run.ROOT/'protocol.json');c=dict(c,ssl=dict(c['ssl'],epochs=2))")
        write(B/name,s)
    files=sorted(p for p in B.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts)
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),feature_version=(3,8))
    manifest={p.relative_to(B).as_posix():sha(p) for p in files}
    write(B/'manifest.json',json.dumps(manifest,indent=2)+'\n')
    archive=H/'gdi-stt692-ssl.zip';prefix='gdi-stt692-ssl-20261031/'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[B/'manifest.json']:z.write(p,prefix+p.relative_to(B).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==digest
    assert source_hashes=={p.relative_to(SOURCE).as_posix():sha(p) for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    result=dict(passed=True,sha256=sha(archive),manifest_sha256=sha(B/'manifest.json'),source_hashes=source_hashes,
        source_unchanged=True,python38_syntax=True,zip_hashes_verified=True,data_sha256=sha(B/'data.npz'))
    write(H/'package.json',json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
if __name__=='__main__':main()
