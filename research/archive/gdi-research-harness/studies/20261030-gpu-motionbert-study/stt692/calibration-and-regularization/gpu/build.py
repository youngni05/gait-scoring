"""Prepare isolated factorial GPU bundle; never submit or modify running experiments."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile
H=Path(__file__).resolve().parent;B=H/'bundle';SRC=H.parents[1]/'bilateral/bundle'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new)
def main():
    original={str(p.relative_to(SRC)):sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    B.mkdir(parents=True,exist_ok=True)
    excluded={'manifest.json','worker.py','submit.slurm','capacity.py','README.md'}
    for p in SRC.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name not in excluded:
            dst=B/p.relative_to(SRC);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
    for name in ('recipe.py','verify_recipe.py','pilot_all.py','submit.slurm'):
        shutil.copyfile(H/name,B/name)
    s=(B/'run.py').read_text(encoding='utf8')
    s=replace(s,'from windows import WindowSet,weighted_mse','from windows import WindowSet,weighted_mse\nfrom recipe import configure,recipe_identity,training_weights,make_optimizer,VARIANTS')
    s=replace(s,'identity=dict(identity,train_x_sha=', 'identity=dict(identity,training_recipe=recipe_identity(c),train_x_sha=')
    s=replace(s,"opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=0.)",'opt=make_optimizer(model,c)')
    s=replace(s,'weights=x.side_weights()','weights=training_weights(x,ids,c)')
    s=replace(s,"return dict(bundle_sha256=sha(ROOT/'manifest.json'),purpose=purpose", "return dict(training_recipe=recipe_identity(c),bundle_sha256=sha(ROOT/'manifest.json'),purpose=purpose")
    s=replace(s,"windows=len(d['x'].values),record_balanced_loss=True,primary_output=", "windows=len(d['x'].values),training_recipe=recipe_identity(c),record_balanced_loss=not c['person_weighted'],primary_output=")
    s=replace(s,"'windows.py','worker.py','submit.slurm'", "'windows.py','recipe.py','verify_recipe.py','pilot_all.py','submit.slurm'")
    s=replace(s,"parser.add_argument('--folds',", "parser.add_argument('--variant',choices=VARIANTS,required=True)\n    parser.add_argument('--folds',")
    s=replace(s,"a=parser.parse_args();c=load(ROOT/'protocol.json');", "a=parser.parse_args();a.output=a.output/a.variant;c=configure(load(ROOT/'protocol.json'),a.variant);")
    (B/'run.py').write_text(s,encoding='utf8')
    for name in ('verify.py','verify_nested.py'):
        s=(B/name).read_text(encoding='utf8')
        s=replace(s,"p=argparse.ArgumentParser();", "p=argparse.ArgumentParser();p.add_argument('--variant',choices=run.VARIANTS,required=True);")
        s=replace(s,"a=p.parse_args();a.output.mkdir", "a=p.parse_args();a.output=a.output/a.variant;a.output.mkdir")
        s=replace(s,"c=run.load(run.ROOT/'protocol.json')", "c=run.configure(run.load(run.ROOT/'protocol.json'),a.variant)")
        (B/name).write_text(s,encoding='utf8')
    c=json.loads((B/'protocol.json').read_text());c.update(name='bilateral_person_weight_decay_factorial',preparation_only=True,
        factorial_variants=['baseline','person','decay','person_decay'],candidate_weight_decay=0.01,
        calibration='unchanged training-record OLS for all GPU arms; CPU comparison separate',
        purpose='isolate person-balanced training loss and AdamW weight decay; no dropout/model change',
        pilot_weights_reused=False,full_training_authorized=False)
    (B/'protocol.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf8')
    manifest={p.relative_to(B).as_posix():sha(p) for p in sorted(B.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    for name in manifest:
        if name.endswith('.py'):ast.parse((B/name).read_text(encoding='utf8'),feature_version=(3,8))
    (B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    archive=H/'gdi-stt692-person-regularization.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest)+['manifest.json']:z.write(B/name,'gdi-stt692-person-regularization-20261031/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('gdi-stt692-person-regularization-20261031/'+name)).hexdigest()==digest
    assert original=={name:sha(SRC/name) for name in original}
    assert manifest['data.npz']==original['data.npz']
    result={'passed':True,'python38_syntax':True,'archive_sha256':sha(archive),'bundle_sha256':sha(B/'manifest.json'),
            'data_unchanged':True,'source_bundle_unchanged':True,'source_files_sha256':original,'submitted':False}
    (H/'package.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'}))
if __name__=='__main__':main()
