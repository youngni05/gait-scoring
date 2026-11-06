"""Run the shipped archive in a fresh process and verify corruption is rejected."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,zipfile
HERE=Path(__file__).resolve().parent
def main():
    checks=HERE/'checks';checks.mkdir(exist_ok=True)
    archive=HERE/'gdi-stt692-pilot.zip'
    with tempfile.TemporaryDirectory(dir=checks,prefix='standalone-') as temp:
        root=Path(temp)
        with zipfile.ZipFile(archive) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(root)
        b=root/'gdi-stt692-pilot-20261031'
        command=[sys.executable,'-B',str(b/'run.py'),'check','--device','cpu','--output',str(root/'result')]
        result=subprocess.run(command,cwd=b,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,(result.stdout,result.stderr)
        check=json.loads((root/'result/check.json').read_text());assert check['passed']
        # A real corrupted model file must be refused before data/model inference.
        p=b/'model/MLP.py';p.write_bytes(p.read_bytes()+b'\n# corruption fixture\n')
        corrupted=subprocess.run(command,cwd=b,capture_output=True,text=True,timeout=180)
        assert corrupted.returncode!=0 and 'Bundle changed: model/MLP.py' in corrupted.stderr
    result=dict(passed=True,standalone_zip_cli_check=True,corrupt_bundle_rejected=True,real_GDI_optimizer_steps=0,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),parameters=check['parameters'])
    (HERE/'launch_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
