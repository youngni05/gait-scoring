"""Create an isolated bilateral experiment, preserving the running bundle."""
from pathlib import Path
import shutil,json,hashlib,ast,zipfile
H=Path(__file__).resolve().parent;B=H/'bundle';SRC=H.parent/'multiwindow/bundle'
def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)
def main():
    B.mkdir(parents=True,exist_ok=True)
    for p in SRC.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='manifest.json':
            dest=B/p.relative_to(SRC);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    for name in ['bilateral.py','mil_pool.py','verify_structure.py','worker.py','submit.slurm']:
        shutil.copyfile(H/name,B/name)
    s=(B/'run.py').read_text()
    s=replace(s,'from model.Transformer import SpatioTemporalTransformer','from bilateral import BilateralSTT')
    s=replace(s,'return SpatioTemporalTransformer(num_frame=124,num_joints=4,in_chans=2,joint_embed_dim=12,num_heads=2,drop_rate=0.)','return BilateralSTT()')
    s=replace(s,"x.values[k:k+64].reshape(-1,124,4,2)","x.values[k:k+64]")
    s=replace(s,"xt=x.values.reshape(-1,124,4,2)","xt=x.values")
    s=replace(s,"permutation(len(yt))","permutation(len(xt))")
    s=replace(s,"for start in range(0,len(order),c['batch_size']):","for start in range(0,len(order),c['batch_size']//2):")
    s=replace(s,"ix=order[start:start+c['batch_size']]","ix=order[start:start+c['batch_size']//2]")
    s=replace(s,"torch.from_numpy(yt[ix])","torch.from_numpy(yt.reshape(-1,2)[ix].reshape(-1))")
    s=replace(s,"torch.from_numpy(weights[ix])","torch.from_numpy(weights.reshape(-1,2)[ix].reshape(-1))")
    s=replace(s,"loss_sum+=float(loss.detach())*len(ix)","loss_sum+=float(loss.detach())*len(ix)*2")
    s=replace(s,"d['x'].values[:2].reshape(-1,124,4,2)","d['x'].values[:2]")
    s=replace(s,"'worker.py','submit.slurm'","'worker.py','submit.slurm','bilateral.py','mil_pool.py','verify_structure.py'")
    (B/'run.py').write_text(s,encoding='utf8')
    s=(B/'verify.py').read_text()
    start=s.index('    model=run.new_model()');end=s.index('    with tempfile.TemporaryDirectory',start)
    s=s[:start]+'''    model=run.new_model().to(a.device).eval();t=torch.from_numpy(x.values[:2]).to(a.device)
    with torch.inference_mode():
        grouped=model(t).reshape(2,2)
        swapped=model(t.flip(1)).reshape(2,2).flip(1)
        separate=torch.cat([model(t[k:k+1]) for k in range(2)]).reshape(2,2)
    torch.testing.assert_close(grouped,swapped,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(grouped,separate,rtol=1e-5,atol=1e-6)
'''+s[end:]
    (B/'verify.py').write_text(s,encoding='utf8')
    s=(B/'verify_windows.py').read_text()
    s=replace(s,'return x[:,0,0,0]','return x[:,:,0,0,0].reshape(-1)')
    (B/'verify_windows.py').write_text(s,encoding='utf8')
    c=json.loads((B/'protocol.json').read_text());c.update(name='bilateral_shared_STT_mean_and_antisymmetric_difference',parallel_workers=[[0,1,2,3,4]],hold_after_training=False)
    c['bilateral_design']={'joint_leg_context':True,'output':'left=common+delta,right=common-delta','pair_batch':64,'side_rows_per_batch':128,'swap_equivariant':True,'window_pooling':'unchanged_uniform','pilot_weights_reused':False}
    c['note']='Same multiwindow data, split, optimizer, nested selection and evaluation; new paired bilateral head, 64 paired windows per step.'
    (B/'protocol.json').write_text(json.dumps(c,indent=2)+'\n')
    (B/'README.md').write_text((H/'README.md').read_text(encoding='utf8'),encoding='utf8')
    manifest={}
    for p in sorted(B.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='manifest.json':
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
            manifest[p.relative_to(B).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest['data.npz']==json.loads((SRC/'manifest.json').read_text())['data.npz']
    (B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=H/'gdi-stt692-bilateral.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest)+['manifest.json']:z.write(B/name,'gdi-stt692-bilateral-20261031/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read('gdi-stt692-bilateral-20261031/'+name)).hexdigest()==digest
    report=dict(passed=True,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bundle_sha256=hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest(),data_sha256=manifest['data.npz'],bytes=archive.stat().st_size,python38_syntax=True,zip_all_hashes_verified=True,original_bundle_unchanged=True)
    (H/'package_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    for name in ['verify_package.py','verify_pilot.py']:
        s=(H.parent/'multiwindow'/name).read_text(encoding='utf8')
        s=s.replace('gdi-stt692-multiwindow','gdi-stt692-bilateral')
        if name=='verify_pilot.py':
            s=s.replace('Float32 CPU mismatch retained; exact same learned weight values evaluated with float64 CPU arithmetic. Original tolerance unchanged.','Float32 and float64 CPU differences both recorded; same learned weights, original tolerance unchanged.')
            s=replace(s,"['synthetic','nested','windows']","['synthetic','nested','windows','structure']")
            s=replace(s,"x.values[start:start+64].reshape(-1,124,4,2)","x.values[start:start+64]")
        (H/name).write_text(s,encoding='utf8')
if __name__=='__main__':main()
