"""CPU verification of the prepared archive; synthetic fitting only."""
from pathlib import Path
import json,subprocess,sys,tempfile,zipfile,hashlib
H=Path(__file__).resolve().parent;B=H/'bundle';C=H/'checks'
def call(args,log,cwd=B):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf8') as out:
        p=subprocess.run([sys.executable,'-u','-B']+[str(v) for v in args],cwd=cwd,stdout=out,stderr=subprocess.STDOUT,timeout=600)
    if p.returncode:raise RuntimeError('Check failed; see '+str(log))
def main():
    C.mkdir(exist_ok=True);results={}
    call([B/'verify_recipe.py','--output',C/'recipe'],C/'recipe.log')
    for variant in ('baseline','person','decay','person_decay'):
        for script,group in [('verify.py','synthetic'),('verify_nested.py','nested')]:
            call([B/script,'--variant',variant,'--device','cpu','--output',C/group],C/(group+'-'+variant+'.log'))
            results[group+'-'+variant]=json.loads((C/group/variant/'verification.json').read_text())
        print('VERIFIED '+variant,flush=True)
    with tempfile.TemporaryDirectory(dir=C,prefix='zip-') as temp:
        root=Path(temp);archive=H/'gdi-stt692-person-regularization.zip'
        with zipfile.ZipFile(archive) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(root)
        b=root/'gdi-stt692-person-regularization-20261031'
        call([b/'run.py','check','--variant','person_decay','--device','cpu','--output',root/'check'],C/'zip.log',b)
        assert json.loads((root/'check/person_decay/check.json').read_text())['passed']
        model=b/'recipe.py';model.write_bytes(model.read_bytes()+b'\n# corruption fixture\n')
        p=subprocess.run([sys.executable,'-B',str(b/'run.py'),'check','--variant','person','--device','cpu','--output',str(root/'bad')],cwd=b,capture_output=True,text=True,timeout=120)
        assert p.returncode!=0 and 'Bundle changed: recipe.py' in p.stderr
    package=json.loads((H/'package.json').read_text());src=H.parents[1]/'bilateral/bundle'
    assert all(hashlib.sha256((src/name).read_bytes()).hexdigest()==sha for name,sha in package['source_files_sha256'].items())
    out={'passed':True,'checks':results,'weight_checks':json.loads((C/'recipe/verification.json').read_text()),
         'independent_zip_check':True,'corrupt_bundle_rejected':True,'original_source_unchanged':True,
         'real_GDI_optimizer_steps':0,'GPU_executed':False,'submitted':False,'archive_sha256':package['archive_sha256']}
    (H/'verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('ALL_PREPARATION_CHECKS_PASSED',flush=True)
if __name__=='__main__':main()
