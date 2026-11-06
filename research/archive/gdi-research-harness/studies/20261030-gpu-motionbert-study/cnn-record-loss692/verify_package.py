"""Fresh extraction, standalone checks and deliberate corrupted-input rejection."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=H/'gdi-cnn-record-loss692.zip';package=json.loads((H/'package_verification.json').read_text())
    assert sha(p)==package['zip_sha256']
    root=H/'checks'/('extracted-'+package['manifest_sha256'][:12]);root.mkdir(exist_ok=True)
    with zipfile.ZipFile(p) as z:
        for member in z.infolist():
            dest=(root/member.filename).resolve();assert root.resolve() in dest.parents and not member.is_dir()
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(member.filename))
    b=root/'gdi-cnn-record-loss692-20261031';manifest=json.loads((b/'manifest.json').read_text())
    for n,h in manifest.items():assert sha(b/n)==h
    for variant in ('window_control','record_mean'):
        cmd=[sys.executable,'-B',str(b/'run.py'),'check','--variant',variant,'--device','cpu','--output',str(root/variant)]
        cp=subprocess.run(cmd,capture_output=True,text=True)
        (root/(variant+'.log')).write_text(cp.stdout+cp.stderr,encoding='utf8');assert cp.returncode==0,cp.stderr
        check=json.loads((root/variant/'input_check.json').read_text());assert check['people']==692 and check['records']==1061 and check['windows']==11261
    # Only the disposable extracted copy is changed; original baseline and package stay frozen.
    target=b/'protocol.json';original=target.read_bytes();target.write_bytes(original+b' ')
    cp=subprocess.run(cmd,capture_output=True,text=True);assert cp.returncode!=0 and 'Bundle changed: protocol.json' in cp.stderr
    target.write_bytes(original)
    verify=json.loads((H/'checks/cpu-audited/verification.json').read_text())
    assert verify['passed'] and verify['bundle_sha256']==package['manifest_sha256']
    baseline=H.parent/'cnn-multiwindow695/bundle';old=json.loads((baseline/'manifest.json').read_text())
    assert all(sha(baseline/n)==h for n,h in old.items())
    for n in ('data.npz','model.py','windows.py'):assert sha(b/n)==old[n]
    report=dict(passed=True,standalone_extraction_checked=True,two_variants_input_check=True,corrupted_bundle_rejected=True,baseline_files_unchanged=True,
        zip_sha256=sha(p),bundle_sha256=package['manifest_sha256'],cpu_verification=verify,real_GDI_optimizer_steps=0,gpu_execution=False,server_submitted=False)
    (H/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cpu_verification'}))
if __name__=='__main__':main()
