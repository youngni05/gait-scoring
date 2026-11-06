"""Execute fresh archive and audit baseline correspondence without training real labels."""
from pathlib import Path
import hashlib,json,sys,subprocess,tempfile,zipfile
import numpy as np
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    archive=H/'gdi-stt692-window-attention.zip';old=H.parent/'multiwindow';checks=H/'checks';checks.mkdir(exist_ok=True)
    assert sha(old/'gdi-stt692-multiwindow.zip')=='4062f92eb1cddec2e67ca0b4ef0775e344d3a84aabb03f4b5318c455bd18bcd8'
    assert sha(old/'bundle/manifest.json')=='ee52bd3827fd56c6151466a856c6876368a76b02fc4e14b143b8f64cac4c03c5'
    with np.load(H/'bundle/data.npz') as z,np.load(old/'bundle/data.npz') as o:
        assert set(z.files)==set(o.files)
        for key in z.files:np.testing.assert_array_equal(z[key],o[key])
        with np.load(H.parent.parent/'cnn-multiwindow695/bundle/data.npz') as cnn:
            for key in ['y','person_id','video_id','fold','original_row_index','window_record','window_start','canonical_fallback']:np.testing.assert_array_equal(z[key],cnn[key])
            np.testing.assert_array_equal(z['x'].reshape(11261,2,124,8),cnn['x'][...,:8])
    variants={}
    with tempfile.TemporaryDirectory(dir=checks,prefix='standalone-') as tmp:
        root=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(root)
        b=root/'gdi-stt692-window-attention-20261031'
        for variant in ['mean','attention']:
            command=[sys.executable,'-u','-B',str(b/'run.py'),'check','--variant',variant,'--device','cpu','--output',str(root/variant)]
            proc=subprocess.run(command,cwd=b,capture_output=True,text=True,timeout=180)
            assert proc.returncode==0,(proc.stdout,proc.stderr)
            variants[variant]=json.loads((root/variant/'check.json').read_text())
        corrupt=b/'bag_model.py';corrupt.write_bytes(corrupt.read_bytes()+b'\n# corrupt fixture\n')
        proc=subprocess.run(command,cwd=b,capture_output=True,text=True,timeout=180)
        assert proc.returncode!=0 and 'Bundle changed: bag_model.py' in proc.stderr
    report=dict(passed=True,zip_sha256=sha(archive),all_input_arrays_identical=True,cnn_xy_windows_and_metadata_identical=True,
        existing_stt_bundle_unchanged=True,standalone_zip_both_variants_passed=True,corrupt_bundle_rejected=True,
        real_GDI_optimizer_steps=0,server_submitted=False,variants=variants)
    (H/'launch_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
